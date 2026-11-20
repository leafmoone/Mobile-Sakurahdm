import math

import torch
from torch import nn
from torch.nn import functional as F

from mobileo.model.mobile_block import LayerwiseMobileFusion

from .xut_activations import use_powlu_in_xut


class HDMConditioningProjector(LayerwiseMobileFusion):
    """Mobile-O layer mixing and sequence refinement at the XUT context width."""

    def _project(self, fused):
        tokens = self.input_compress(fused.to(self.input_compress[0].weight))
        for refiner in self.sequence_refiners:
            tokens = refiner(tokens)
        return self.output_proj(tokens)

    def forward(self, hidden_states, attention_mask):
        weights = torch.softmax(self.layer_weights / self.temperature, dim=0)
        fused = sum(
            weight * state
            for weight, state in zip(weights, hidden_states[-self.num_layers :])
        )

        # Neither the convolutional refiners nor XUT joint attention can
        # ignore padding. Group exact token lengths and compact each group.
        mask = attention_mask.bool()
        lengths = mask.sum(dim=1).tolist()
        rows_by_length = {}
        for row, length in enumerate(lengths):
            rows_by_length.setdefault(length, []).append(row)
        if len(rows_by_length) == 1:
            length = lengths[0]
            if length != fused.size(1):
                fused = fused[mask].reshape(fused.size(0), length, fused.size(-1))
            return self._project(fused)

        compact = fused[mask].split(lengths)
        return tuple(
            (
                torch.tensor(rows, device=fused.device),
                self._project(torch.stack([compact[row] for row in rows])),
            )
            for rows in rows_by_length.values()
        )


class MobileSakuraHDM(nn.Module):
    def __init__(self, qwen, dit, vae, connector, source_adapter=False):
        super().__init__()
        self.qwen = qwen
        self.dit = dit
        self.vae = vae.requires_grad_(False).eval()
        self.connector = connector
        channels = dit.config.input_dim
        self.source_adapter = (
            nn.Conv2d(channels, channels, kernel_size=1) if source_adapter else None
        )
        if self.source_adapter is not None:
            nn.init.zeros_(self.source_adapter.weight)
            nn.init.zeros_(self.source_adapter.bias)
        for name in ("latents_mean", "latents_std"):
            self.register_buffer(
                name,
                torch.as_tensor(vae.config[name]).view(1, -1, 1, 1),
                persistent=False,
            )
        self.latent_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def encode_images(self, pixels, generator=None):
        with torch.autocast(device_type=self.vae.device.type, enabled=False):
            posterior = self.vae.encode(
                pixels.to(self.vae.device, self.vae.dtype)
            ).latent_dist
        latents = posterior.sample(generator=generator)
        return (latents.float() - self.latents_mean) / self.latents_std

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents.float() * self.latents_std + self.latents_mean
        return self.vae.decode(latents.to(self.vae.dtype)).sample

    def condition(self, qwen_inputs):
        output = self.qwen.model(
            **qwen_inputs, use_cache=False, output_hidden_states=True
        )
        return self.connector(output.hidden_states, qwen_inputs["attention_mask"])

    def denoise(self, noisy_latents, timesteps, context, source_latents=None, pos_map=None, aspect_ratio=None):
        latents = noisy_latents.to(self.dit.dtype)
        if source_latents is not None and self.source_adapter is not None:
            latents = latents + self.source_adapter(
                source_latents.to(self.source_adapter.weight)
            ).to(latents)
        aspect = (latents.new_full((latents.size(0),), math.log(latents.size(-1)/latents.size(-2)))
                  if aspect_ratio is None else aspect_ratio.to(latents))
        extras = (
            {"addon_info": aspect}
            if self.dit.config.addon_info_embs_dim
            else None
        )
        time = timesteps.to(latents)
        if pos_map is not None:
            pos_map = pos_map.to(device=latents.device, dtype=torch.float32)

        def predict(x, t, tokens, positions, extra):
            return self.dit(
                x,
                t,
                encoder_hidden_states=tokens.to(x),
                added_cond_kwargs=extra,
                pos_map=positions,
            ).sample

        if isinstance(context, torch.Tensor):
            return predict(latents, time, context, pos_map, extras)

        outputs, rows = [], []
        for indices, tokens in context:
            extra = (
                {"addon_info": aspect.index_select(0, indices)}
                if extras is not None
                else None
            )
            outputs.append(
                predict(
                    latents.index_select(0, indices),
                    time.index_select(0, indices),
                    tokens,
                    pos_map.index_select(0, indices) if pos_map is not None else None,
                    extra,
                )
            )
            rows.append(indices)
        return torch.cat(outputs).index_select(0, torch.cat(rows).argsort())

    def language_loss(self, qwen_inputs, labels):
        hidden = self.qwen.model(**qwen_inputs, use_cache=False).last_hidden_state
        targets = labels[:, 1:]
        scored = targets != -100
        # Score answer tokens only; a multimodal prompt may contain many vision tokens.
        logits = self.qwen.lm_head(hidden[:, :-1][scored])
        return F.cross_entropy(logits.float(), targets[scored])

    def forward(
        self,
        qwen_inputs,
        noisy_latents=None,
        timesteps=None,
        labels=None,
        source_latents=None,
        language_inputs=None,
        pos_map=None,
        aspect_ratio=None,
    ):
        if noisy_latents is None:
            return self.language_loss(qwen_inputs, labels)
        velocity = self.denoise(
            noisy_latents, timesteps, self.condition(qwen_inputs), source_latents, pos_map, aspect_ratio
        )
        if labels is not None:
            return velocity, self.language_loss(language_inputs, labels)
        return velocity


def build_processor(config):
    from transformers import AutoProcessor

    qwen = config["qwen"]
    return AutoProcessor.from_pretrained(
        qwen["name_or_path"], **qwen.get("processor_kwargs", {})
    )


def _diffusers_kwargs(config):
    kwargs = dict(config.get("load_kwargs", {}))
    if isinstance(dtype := kwargs.get("torch_dtype"), str):
        kwargs["torch_dtype"] = getattr(torch, dtype)
    return kwargs


def _load_hdm(config):
    from xut import env

    for key, value in config.get("xut", {}).items():
        setattr(env, key, value)
    from diffusers.configuration_utils import ConfigMixin
    from hdm.modules.xut import XUDiTConditionModel

    # The upstream JSON-only classmethod omits keyword arguments required by
    # Diffusers' pretrained loader. Keep the native XUT constructor/weights.
    class PretrainedXUT(XUDiTConditionModel):
        from_config = classmethod(ConfigMixin.from_config.__func__)

    if "architecture" in config:
        dit = PretrainedXUT.from_config(config["architecture"])
    else:
        dit = PretrainedXUT.from_pretrained(
            config["name_or_path"], **_diffusers_kwargs(config)
        )
    # Keep pretrained pre/post TREAD blocks. Zero routing dropout retains every token;
    # use_tread=False would incorrectly bypass those trained transformations.
    if config.get("tread", {}).get("enabled") is False and dit.model.use_tread:
        dit.model.dropout_ratio = 0.0
    m = config.get("powlu_m", 3.0)
    if m is not None:
        use_powlu_in_xut(dit, m)
    return dit


def build_model(config):
    from diffusers import AutoencoderKL
    from transformers import AutoModelForMultimodalLM

    qwen_config = config["qwen"]
    qwen = AutoModelForMultimodalLM.from_pretrained(
        qwen_config["name_or_path"], **qwen_config.get("load_kwargs", {})
    )
    dit = _load_hdm(config["hdm"])
    vae_config = config["vae"]
    vae = AutoencoderKL.from_pretrained(
        vae_config["name_or_path"], **_diffusers_kwargs(vae_config)
    )
    connector = HDMConditioningProjector(
        input_dim=qwen.config.text_config.hidden_size,
        output_dim=dit.config.ctx_dim,
        **config["connector"],
    )
    return MobileSakuraHDM(
        qwen, dit, vae, connector, config.get("source_adapter", False)
    )

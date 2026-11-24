"""Opt-in Mage-Flow VAE / widened HDM XUT; no changes to the stock loader."""
from pathlib import Path
import hashlib
import json
import math
import os
import re

import torch
from torch import nn


PROJECTIONS = {
    'attn.qkv': (3, 'd'), 'attn.out': (1, 'd'),
    'xattn.q': (1, 'd'), 'xattn.kv': (2, 'd'), 'xattn.out': (1, 'd'),
    'mlp.w12': (2, 'm'), 'mlp.w3': (1, 'd'),
}


def transfer_tensor(name, src, dst, old, new, old_mlp, new_mlp):
    """Copy semantically aligned chunks, rather than one top-left rectangle."""
    tail = name.removeprefix('model.')
    if tail == 'time_emb.freqs':
        if src.shape != (1, old // 2) or dst.shape != (1, new // 2):
            raise ValueError(f'{name}: invalid frequency table')
        dst[..., :old // 2].copy_(src)
        return 'frequencies'
    if tail == 'time_emb.proj.0.weight':
        if src.shape != (old, old) or dst.shape != (new, new):
            raise ValueError(f'{name}: invalid time projection')
        dst[:old, old // 2:new // 2].zero_()
        dst[:old, new // 2 + old // 2:].zero_()
        dst[:old, :old // 2].copy_(src[:, :old // 2])
        dst[:old, new // 2:new // 2 + old // 2].copy_(src[:, old // 2:])
        dst[old:].mul_(0.25)
        return 'sin-cos'
    if re.search(r'\.rope\.freqs$', tail):
        if (src.ndim != 3 or dst.ndim != 3 or src.shape[1:] != dst.shape[1:]
                or new * src.shape[0] != old * dst.shape[0]):
            raise ValueError(f'{name}: RoPE head dimensions changed')
        dst[:src.shape[0]].copy_(src)
        return 'rope-heads'
    if re.search(r'(?:\.(?:attn|xattn|mlp)_pre_norm\.norm|^shared_adaln_(?:attn|xattn|ffw)\.0)\.(?:weight|bias)$', tail):
        # For RMSNorm, neutralize the 1/width change under zero extension
        # at ordinary activation scales. Standard LayerNorm stays native;
        # its old affine is copied, not advertised as zero-extension exact.
        if src.shape != (old,) or dst.shape != (new,):
            raise ValueError(f'{name}: invalid normalization width')
        dst[:old].copy_(src)
        if '.norm.weight' in tail:
            dst[:old].mul_(math.sqrt(old / new))
        return 'normalization'
    stem = tail.rsplit('.', 1)[0]
    projection = next((p for suffix, p in PROJECTIONS.items() if stem.endswith('.' + suffix)), None)
    if projection:
        split, width = projection
        old_out, new_out = (old_mlp, new_mlp) if width == 'm' else (old, new)
    elif re.fullmatch(r'shared_adaln_(attn|xattn|ffw)\.3', stem):
        split, old_out, new_out = 3, old, new
    elif re.fullmatch(r'shared_adaln_(attn|xattn|ffw)\.1', stem):
        split, old_out, new_out = 1, 4 * old, 4 * new
    elif stem in ('time_emb.proj.0', 'ctx_proj', 'addon_info_embs_proj.0', 'addon_info_embs_proj.2'):
        split, old_out, new_out = 1, old, new
    else:
        raise ValueError(f'Unhandled widened tensor {name}: {tuple(src.shape)} -> {tuple(dst.shape)}')
    if src.ndim not in (1, 2) or dst.ndim != src.ndim or src.shape[0] != split*old_out or dst.shape[0] != split*new_out:
        raise ValueError(f'Unexpected projection layout: {name}')
    if src.ndim == 2:
        expected_input = (old, new)
        if stem.endswith('.mlp.w3'):
            expected_input = (old_mlp, new_mlp)
        elif split == 3 and stem.startswith('shared_adaln_'):
            expected_input = (4 * old, 4 * new)
        elif stem == 'ctx_proj':
            expected_input = (1024, 1024)
        elif stem == 'addon_info_embs_proj.0':
            expected_input = (1, 1)
        if (src.shape[1], dst.shape[1]) != expected_input:
            raise ValueError(f'Unexpected input axis: {name}')
    for i in range(split):
        a = src[i * old_out:(i + 1) * old_out]
        b = dst[i * new_out:(i + 1) * new_out]
        if src.ndim == 1:
            b.zero_()
            b[:old_out].copy_(a)
        else:
            b[old_out:, :a.shape[1]].mul_(0.25)
            b[:old_out].zero_()
            b[:old_out, :a.shape[1]].copy_(a)
    return 'segmented' if split > 1 else 'leading'


@torch.no_grad()
def transfer_weights(source, target, old=1024, new=1280, old_mlp=3072, new_mlp=4096):
    if source.keys() != target.keys():
        raise ValueError(f'Checkpoint keys mismatch: {sorted(source.keys() ^ target.keys())[:10]}')
    counts = {}
    for k, dst in target.items():
        if k.startswith(('model.in_patch.', 'model.out_patch.')):
            kind = 'fresh-latent-io'
        else:
            kind = transfer_tensor(k, source[k], dst, old, new, old_mlp, new_mlp)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def build_model(config):
    from sakurahdm.models import MobileSakuraHDM, HDMConditioningProjector, _load_hdm
    from sakurahdm.sakuramoon_data import sakuramoon_root
    from transformers import AutoModelForMultimodalLM

    sakuramoon_root()
    from sakuramoon.encoders.mage_vae import FrozenMageVAE, MageVAE
    from safetensors.torch import load_file

    hdm = config['hdm']
    mage = config['vae']['mage']
    if (mage.get('dtype') != 'bfloat16' or mage.get('sample_posterior') is not False
            or mage.get('frozen') is not True):
        raise ValueError('Mage experiment requires BF16 posterior mean')
    if mage.get('latent_channels') != 128 or mage.get('downsample_factor') != 16:
        raise ValueError('Mage latent contract is 128 channels, 16x downsample')
    dit = _load_hdm(hdm)
    if (dit.config.input_dim, dit.config.patch_size, dit.config.dim, dit.config.ctx_dim,
            dit.config.heads, dit.config.dim_head, dit.config.mlp_dim) != (128, 1, 1280, 1024, 20, 64, 4096):
        raise ValueError('Unexpected widened XUT architecture')
    source_path = Path(hdm['name_or_path'], hdm['load_kwargs']['subfolder'], 'diffusion_pytorch_model.safetensors')
    source_arch = json.loads(source_path.with_name('config.json').read_text(encoding='utf-8'))
    keys = ('input_dim', 'patch_size', 'dim', 'ctx_dim', 'heads', 'dim_head', 'mlp_dim')
    if tuple(source_arch[k] for k in keys) != (4, 2, 1024, 1024, 16, 64, 3072):
        raise ValueError('Expected original HDM-340M source checkpoint')
    counts = transfer_weights(load_file(source_path, device='cpu'), dit.state_dict())
    if counts.get('fresh-latent-io') != 4:
        raise ValueError('Expected two newly initialized weight/bias pairs for latent I/O')
    with torch.no_grad():
        # Nonzero, near-zero flow head: trainable trunk receives gradients from step one.
        nn.init.normal_(dit.model.out_patch.proj.weight, std=0.02 / math.sqrt(dit.config.dim))
        nn.init.zeros_(dit.model.out_patch.proj.bias)
        dit.model.in_patch.proj.weight[1024:].mul_(0.25)
        dit.model.in_patch.proj.bias[1024:].mul_(0.25)
    codec_path = Path(mage['checkpoint'])
    with codec_path.open('rb') as f:
        codec_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    if codec_hash != mage['checkpoint_sha256']:
        raise ValueError('Unexpected Mage codec checkpoint digest')
    vae = FrozenMageVAE(MageVAE(codec_path).to(dtype=torch.bfloat16))
    qwen = AutoModelForMultimodalLM.from_pretrained(config['qwen']['name_or_path'], **config['qwen']['load_kwargs'])
    connector = HDMConditioningProjector(input_dim=qwen.config.text_config.hidden_size,
                                       output_dim=dit.config.ctx_dim, **config['connector'])

    class MageHDM(MobileSakuraHDM):
        def __init__(self, source_adapter):
            # No EQ-SDXL statistics, transforms or config on this path.
            nn.Module.__init__(self)
            self.qwen, self.dit, self.vae, self.connector = qwen, dit, vae, connector
            self.latent_scale_factor = 16
            self.latent_cache_namespace = f'mage-{codec_hash}-posterior-mean-native-bf16'
            self.source_adapter = nn.Conv2d(128, 128, 1) if source_adapter else None
            if self.source_adapter is not None:
                nn.init.zeros_(self.source_adapter.weight)
                nn.init.zeros_(self.source_adapter.bias)

        @torch.no_grad()
        def encode_images(self, pixels, generator=None):
            param = next(self.vae.parameters())
            if param.dtype != torch.bfloat16:
                raise ValueError('Mage codec dtype must stay BF16')
            with torch.autocast(param.device.type, enabled=False):
                raw = self.vae.encode(pixels.to(device=param.device, dtype=param.dtype))
                expected = (pixels.shape[0], 128, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
                if raw.shape != expected or raw.dtype != torch.bfloat16 or self.vae.training:
                    raise ValueError('Unexpected Mage latent contract')
                return raw.float()  # Converts to a normal tensor, preserves native latent units.

        @torch.no_grad()
        def decode_latents(self, latents):
            param = next(self.vae.parameters())
            with torch.autocast(param.device.type, enabled=False):
                return self.vae.decode(latents.to(device=param.device, dtype=param.dtype)).float()

    model = MageHDM(config.get('source_adapter', False))
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('MAGE_XUT_TRANSFER', dict(counts), 'xut_parameters', sum(p.numel() for p in dit.parameters()), flush=True)
    return model

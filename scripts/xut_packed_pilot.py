"""Isolated flat-token XUT prototype for the HDM-340M concat-context T2I path.

Uses the original parameters, blocks, patch head and 0/50% TREAD routing.
Not registered in the production launcher. Requires explicitly installing it.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import accumulate
import math

import torch
from torch._C._nn import scaled_dot_product_attention as stock_sdpa


@dataclass(frozen=True)
class Layout:
    lengths: tuple[int, ...]
    group_sizes: tuple[int, ...]  # Flattened token counts for the original dense calls.
    cu: torch.Tensor
    rows: torch.Tensor
    total: int
    max_length: int

    @classmethod
    def create(cls, lengths, group_counts, device):
        if (not lengths or any(n <= 0 for n in lengths) or any(n <= 0 for n in group_counts)
                or sum(group_counts) != len(lengths)):
            raise ValueError('Invalid packed sample lengths')
        ends = (0, *accumulate(lengths))
        sizes, i = [], 0
        for count in group_counts:
            sizes.append(ends[i + count] - ends[i]); i += count
        cu = torch.tensor(ends, dtype=torch.int32, device=device)
        rows = torch.tensor([i for i, n in enumerate(lengths) for _ in range(n)], device=device)
        return cls(tuple(lengths), tuple(sizes), cu, rows, ends[-1], max(lengths))

    def route_indices(self, image_tokens, keep, device):
        """Same RNG draws and token order as stock HDM's mask, no dynamic shape."""
        offset = 0
        selections = []
        if not 0 < keep <= image_tokens or image_tokens > min(self.lengths):
            raise ValueError('Invalid image token bounds')
        for n in self.lengths:
            # Stock mask is randperm(I) < keep; take the corresponding original
            # image positions in increasing order, then preserve all text tokens.
            selected = torch.randperm(image_tokens, device=device).argsort()[:keep].sort().values
            text = torch.arange(image_tokens, n, device=device)
            selections.append(torch.cat((selected, text)).add_(offset))
            offset += n
        return torch.cat(selections)

    def image_indices(self, image_tokens, device):
        return (torch.tensor((0, *accumulate(self.lengths[:-1])), device=device)[:, None]
                + torch.arange(image_tokens, device=device)[None, :]).flatten()


def install(model, *, attention='varlen', grouped_linears=False):
    """Install packed denoising only; grouped_linears is a numerical diagnostic."""
    from xut.modules.adaln import AdaLN
    from xut.modules.attention import CrossAttention, SelfAttention
    from xut.modules.axial_rope import make_axial_pos

    if attention not in ('math', 'serial-flash', 'varlen'):
        raise ValueError('Unknown packed attention backend')
    if grouped_linears not in (False, True, 'attn-out'):
        raise ValueError('Invalid grouped-linear diagnostic')
    if attention == 'varlen':
        from flash_attn import flash_attn_varlen_func
    elif attention == 'serial-flash':
        from flash_attn import flash_attn_func
    cfg = model.dit.config
    xut = model.dit.model
    if getattr(model, '_packed_pilot_installed', False):
        raise ValueError('Packed pilot is already installed')
    if not cfg.concat_ctx or not cfg.shared_adaln or cfg.dec_ctx or cfg.class_cond or cfg.double_t or cfg.use_dyt:
        raise ValueError('The pilot only supports concat-context HDM-340M with shared RMSNorm AdaLN')
    paths = [xut.backbone]
    if xut.use_tread:
        paths += [xut.prev_tread_trns, xut.post_tread_trns]
    if any(path.grad_ckpt for path in paths) or any(m.adaln is not None for m in xut.modules() if isinstance(m, AdaLN)):
        raise ValueError('Packed pilot requires no activation checkpointing and no private AdaLN')
    layers = [m for path in paths for m in path.modules() if isinstance(m, (SelfAttention, CrossAttention))]
    if not layers or any(m.xformers or m.head_dim != 64 for m in layers):
        raise ValueError('Expected stock FP16/FP32 XUT head dimension 64')
    context_state = ContextVar('xut_native_pack_layout', default=None)
    original = model.denoise

    @torch.compiler.disable(reason='Explicit DAS attention boundary and host sequence metadata')
    def execute(layout, q, k, v):
        if not (q.shape == k.shape == v.shape) or q.shape[0] != 1 or q.shape[2] != layout.total:
            raise ValueError('Expected one flat array with identical Q/K/V sequence boundaries')
        if attention != 'math' and (not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16)
                                    or not q.dtype == k.dtype == v.dtype):
            raise ValueError('DAS FlashAttention requires a common device and half precision dtype')
        if attention == 'varlen':
            def flat(t):
                return t[0].transpose(0, 1).contiguous()
            out = flash_attn_varlen_func(flat(q), flat(k), flat(v), layout.cu, layout.cu,
                                       layout.max_length, layout.max_length, dropout_p=0.0, causal=False)
            return out.transpose(0, 1).unsqueeze(0)
        if attention == 'math':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                return torch.cat([stock_sdpa(a, b, c) for a,b,c in zip(
                    *(t.split(layout.lengths, dim=2) for t in (q,k,v)))], dim=2)
        return torch.cat([flash_attn_func(*(t.transpose(1, 2).contiguous() for t in (a,b,c)),
                                         dropout_p=0.0, causal=False).transpose(1, 2) for a,b,c in zip(
            *(t.split(layout.lengths, dim=2) for t in (q,k,v)))], dim=2)

    def attention_wrapper(dense):
        def packed(q, k, v, mask=None):
            layout = context_state.get()
            if layout is None:
                return dense(q, k, v, mask)
            if mask is not None:
                raise ValueError('The pilot does not support explicit attention masks')
            return execute(layout, q, k, v)
        return packed

    def linear_wrapper(dense):
        def group_forward(x):
            layout = context_state.get()
            if layout is None:
                return dense(x)
            return torch.cat(tuple(dense(y) for y in x.split(layout.group_sizes, dim=1)), dim=1)
        return group_forward

    def expand(modulation, layout):
        return [tuple(t.index_select(0, layout.rows).transpose(0, 1) for t in g) for g in modulation]

    def call_block_path(path, layout, x, pos, dummy_t, modulation):
        state = context_state.set(layout)
        try:
            return path(x, pos_map=pos, y=dummy_t, shared_adaln=expand(modulation, layout))
        finally:
            context_state.reset(state)

    def denoise(latents, timesteps, context, source_latents=None, pos_map=None, aspect_ratio=None):
        if isinstance(context, torch.Tensor) or not xut.training:
            return original(latents, timesteps, context, source_latents, pos_map, aspect_ratio)
        if source_latents is not None or xut.grad_ckpt or any(path.grad_ckpt for path in paths):
            raise ValueError('Pilot supports T2I only, without activation checkpointing')
        b, _, h, w = latents.shape
        patch = xut.patch_size
        if h % patch or w % patch:
            raise ValueError('Invalid XUT patch grid')
        device = latents.device
        latents = latents.to(model.dit.dtype)
        time = timesteps.to(latents)
        ratio = aspect_ratio.to(latents) if aspect_ratio is not None else latents.new_full((b,), math.log(w / h))
        if pos_map is not None:
            pos_map = pos_map.to(device=device, dtype=torch.float32)
        nimg = (h // patch) * (w // patch)
        lengths, group_counts = [], []
        token_groups, positions, times, conditions = [], [], [], []
        if not isinstance(context, (list, tuple)) or not context:
            raise ValueError('Expected nonempty groups of equal-length context')
        for inds, text in context:
            if (text.ndim != 3 or not text.shape[0] or not text.shape[1] or text.device != device
                    or inds.shape != (text.shape[0],) or inds.dtype != torch.long or inds.device != device):
                raise ValueError('Invalid context group')
        perm = torch.cat([inds for inds, _ in context])
        if perm.numel() != b:
            raise ValueError('Context must cover the complete batch')
        torch._assert_async(torch.eq(perm.sort().values, torch.arange(b, device=device)).all(),
                            'Context group indices must partition the batch')
        # Match the baseline's shape, arithmetic and group order at the entry;
        # projected text has already been refined without padding by the connector.
        for inds, text in context:
            n, length = text.shape[:2]
            image, pos = xut.in_patch(latents.index_select(0, inds), None if pos_map is None else pos_map.index_select(0, inds))
            image = image.contiguous()
            if pos is None:
                pos = make_axial_pos(h // patch, w // patch, dtype=image.dtype, device=device)[None].expand(n, -1, -1)
            t = xut.time_emb(time.index_select(0, inds).reshape(n, -1))
            if cfg.addon_info_embs_dim:
                t = t + xut.addon_info_embs_proj(ratio.index_select(0, inds)[:, None])[:, None]
            times.append(t)
            conditions.append(tuple(layer(t).chunk(3, dim=-1) for layer in (
                xut.shared_adaln_attn, xut.shared_adaln_xattn, xut.shared_adaln_ffw)))
            token_groups.append(torch.cat((image, xut.ctx_proj(text.to(latents))), dim=1).flatten(0, 1))
            zeros = torch.zeros((n, length, pos.shape[-1]), device=device)
            positions.append(torch.cat((pos, zeros), dim=1).flatten(0, 1))
            lengths.extend([nimg + length] * n)
            group_counts.append(n)
        dummy_t = times[0][:1]  # Shared AdaLN does not read per-block t directly.
        modulation = [tuple(torch.cat([cond[i][j] for cond in conditions], dim=0) for j in range(3)) for i in range(3)]
        full = Layout.create(lengths, group_counts, device)
        x, pos = torch.cat(token_groups).unsqueeze(0), torch.cat(positions).unsqueeze(0)
        if xut.use_tread:
            rate = xut.dropout_ratio
            if not 0 <= rate < 1:
                raise ValueError('Invalid TREAD route ratio')
            x = call_block_path(xut.prev_tread_trns, full, x, pos, dummy_t, modulation)
            keep = nimg - int(nimg * rate)
            # Preserve the image-mask random draws even at rate zero so the next
            # training microbatch starts from the same generator state.
            chosen = full.route_indices(nimg, keep, device)
            short = Layout.create(tuple(n - nimg + keep for n in lengths), group_counts, device)
            y = call_block_path(xut.backbone, short, x.index_select(1, chosen), pos.index_select(1, chosen), dummy_t, modulation)
            x = x.index_copy(1, chosen, y.to(x.dtype))
            x = call_block_path(xut.post_tread_trns, full, x, pos, dummy_t, modulation)
        else:
            x = call_block_path(xut.backbone, full, x, pos, dummy_t, modulation)
        img = x[0].index_select(0, full.image_indices(nimg, device)).reshape(b, nimg, -1)
        return torch.cat(tuple(xut.out_patch(t, h, w) for t in img.split(group_counts)), dim=0).index_select(0, perm.argsort())

    for layer in layers:
        layer.attn = attention_wrapper(layer.attn)
    if grouped_linears:
        linears = ([m.out for m in layers] if grouped_linears == 'attn-out' else
                   [m for path in paths for m in path.modules() if isinstance(m, torch.nn.Linear)])
        for layer in linears:
            layer.forward = linear_wrapper(layer.forward)
    model.denoise = denoise
    model._packed_pilot_installed = True

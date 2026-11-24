#!/usr/bin/env python3
"""Launch Mobile-Sakurahdm on DTK with a mixed SDS socket and vendor FlashAttention.

The vendor ATen Flash SDPA adapter is disabled; the DAS wheel's direct API is
used for supported unmasked layouts. Masked calls keep stock math SDPA.
"""
import os
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F


def use_vendor_flash():
    from flash_attn import flash_attn_func

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    original = F.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if (attn_mask is None and q.is_cuda and q.ndim == k.ndim == v.ndim == 4
                and q.device == k.device == v.device and q.dtype == k.dtype == v.dtype
                and q.dtype in (torch.float16, torch.bfloat16)
                and q.shape[:2] == k.shape[:2] == v.shape[:2] and k.shape[-2] == v.shape[-2]
                and q.shape[-1] in (32, 64, 96, 128) and q.shape[-1] == k.shape[-1] == v.shape[-1]
                and q.shape[-2] > 0 and k.shape[-2] > 0
                and (not is_causal or q.shape[-2] == k.shape[-2])):
            return flash_attn_func(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                                  v.transpose(1, 2).contiguous(), dropout_p=dropout_p,
                                  softmax_scale=scale, causal=is_causal).transpose(1, 2)
        return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                        scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = sdpa


def use_sample_shuffle(buffer_size):
    """Shuffle encoded samples per lease before decode and native length grouping."""
    if buffer_size < 2:
        raise ValueError('shuffle buffer must be at least 2')
    from sakurahdm import sakuramoon_data
    original = sakuramoon_data.leased_samples

    def shuffled(path):
        # PyTorch DataLoader seeds each worker's Python RNG independently.
        seed = random.getrandbits(64)
        return original(path).shuffle(buffer_size, initial=min(16, buffer_size), seed=seed)

    sakuramoon_data.leased_samples = shuffled


def main():
    root = Path.cwd()
    if not (root / 'sakurahdm' / 'train_hdm.py').is_file():
        raise SystemExit('Run from the Mobile-Sakurahdm checkout.')
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit('HIP/DTK PyTorch and a visible DCU are required.')
    sys.path.insert(0, str(root))
    use_vendor_flash()
    use_sample_shuffle(int(os.environ.get('SAKURA_SHUFFLE_BUFFER', '64')))
    from hdm_grad_metrics import with_grad_metrics
    from hdm_training_preview import with_training_previews
    from sakurahdm import train_hdm

    train_hdm.Artifacts = with_grad_metrics(with_training_previews(train_hdm.Artifacts))
    train_hdm.main()


if __name__ == '__main__':
    main()

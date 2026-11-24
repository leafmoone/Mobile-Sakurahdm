"""Opt-in, Lightning-compatible CMuon + AdamW for the HDM XUT block roles.

Adapted from SakuraMoon 08ed12c: semantic chunks, BF16 momentum, NS4,
Nesterov and Moonlight scaling. Uses stock dense AdamW for the other roles.
The RMS check/FP32 rescue here is not SakuraMoon's model-calibrated guard.
"""
import json
import math
import os
import re

import torch
from torch.optim import Optimizer, _functional


_BLOCK = re.compile(r'^dit\.model\.(?:backbone\.(?:enc|dec)_blocks\.\d+\.\d+|(?:prev|post)_tread_trns\.blocks\.\d+)\.(.+)\.weight$')
_CHUNKS = {'attn.qkv': 3, 'attn.out': 1, 'xattn.q': 1, 'xattn.kv': 2, 'xattn.out': 1,
           'mlp.w12': 2, 'mlp.w3': 1}
_ADALN = re.compile(r'^dit\.model\.shared_adaln_(?:attn|xattn|ffw)\.3\.weight$')


def canonical_name(name):
    return '.'.join(part for part in name.split('.') if part != '_orig_mod')


def cmuon_chunks(name):
    name = canonical_name(name)
    if match := _BLOCK.fullmatch(name):
        return _CHUNKS.get(match.group(1), 0)
    return 3 if _ADALN.fullmatch(name) else 0


def ns4(tensor, dtype):
    """SakuraMoon's four quintic Newton–Schulz steps, on a single semantic chunk."""
    return _ns4(tensor, dtype)[0]


def _ns4(tensor, dtype):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = tensor.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    norm = x.norm()
    x = x / norm.clamp_min(1e-7)
    for _ in range(4):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return (x.T if transposed else x), norm


class HybridCMuon(Optimizer):
    def __init__(self, groups, *, betas, eps, weight_decay):
        if not all(0.0 <= b < 1.0 for b in betas) or eps <= 0 or weight_decay < 0:
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(groups, dict(betas=betas, eps=eps, weight_decay=weight_decay, fused=False))
        self.rescued_chunks = 0
        self._validate_routes()

    def _validate_routes(self):
        for group in self.param_groups:
            algo = group['algorithm']
            if algo not in ('cmuon', 'adamw'):
                raise ValueError(f'Unknown optimizer group {algo}')
            if algo == 'cmuon' and (len(group['params']) != len(group['chunk_counts']) or any(
                count < 1 or p.ndim != 2 or p.dtype != torch.float32 or p.shape[0] % count
                for p, count in zip(group['params'],group['chunk_counts'],strict=True)
            )):
                raise ValueError('CMuon requires divisible FP32 matrix weights with semantic chunk counts')

    def load_state_dict(self, state_dict):
        saved = state_dict.get('param_groups', ())
        if len(saved) != len(self.param_groups) or any(
            (a.get('name'), a.get('algorithm'), a.get('chunk_counts')) != (b['name'], b['algorithm'], b.get('chunk_counts'))
            or a.get('param_names') != b['param_names'] for a,b in zip(saved,self.param_groups,strict=True)
        ):
            raise ValueError('Not a compatible HDM hybrid optimizer checkpoint')
        for saved_group, group in zip(saved, self.param_groups, strict=True):
            for key, p in zip(saved_group['params'], group['params'], strict=True):
                slots = state_dict['state'].get(key, {})
                if any(not isinstance(value, torch.Tensor) or value.shape != (torch.Size([]) if name == 'step' else p.shape)
                       for name, value in slots.items()):
                    raise ValueError('Incompatible HDM hybrid optimizer state shape')
        super().load_state_dict(state_dict)
        self._validate_routes()
        for group in self.param_groups:
            if group['algorithm'] == 'cmuon':
                for p in group['params']:
                    if 'momentum_buffer' in self.state[p]:
                        self.state[p]['momentum_buffer'] = self.state[p]['momentum_buffer'].to(torch.bfloat16)

    def _adamw(self, group):
        params, grads, avgs, sqs, steps = [], [], [], [], []
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse or p.is_complex():
                raise ValueError('HDM hybrid AdamW only supports dense real weights')
            s = self.state[p]
            if not s:
                s['step'] = torch.zeros((), dtype=torch.float32, device='cpu')
                s['exp_avg'] = torch.zeros_like(p)
                s['exp_avg_sq'] = torch.zeros_like(p)
            params.append(p); grads.append(p.grad); avgs.append(s['exp_avg']); sqs.append(s['exp_avg_sq']); steps.append(s['step'])
        if params:
            _functional.adamw(params, grads, avgs, sqs, [], steps, foreach=None, fused=False,
                              amsgrad=False, beta1=group['betas'][0], beta2=group['betas'][1], lr=group['lr'],
                              weight_decay=group['weight_decay'], eps=group['eps'], maximize=False)

    def _cmuon(self, group):
        mu, lr, wd = group['momentum'], group['lr'], group['weight_decay']
        # Never allow the training autocast context to lower a guarded FP32 retry.
        device_type = group['params'][0].device.type
        with torch.autocast(device_type, enabled=False):
            for p, count in zip(group['params'], group['chunk_counts'], strict=True):
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError('CMuon only supports dense gradients')
                g = p.grad.to(torch.bfloat16)
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.bfloat16)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - mu)
                direction = g.lerp(buf, mu)
                del g
                patches = []
                for chunk in direction.chunk(count, dim=0):
                    ortho, norm = _ns4(chunk, torch.bfloat16)
                    # RMS envelope for a near-orthogonal matrix. Check finite and
                    # bound extreme NS amplification; do not borrow G1 references.
                    limit = 2.0 * math.sqrt(min(chunk.shape))
                    if not bool(norm.isfinite() & (ortho.float().norm() <= limit)):
                        self.rescued_chunks += 1
                        ortho, norm = _ns4(chunk, torch.float32)
                        if not bool(norm.isfinite() & (ortho.norm() <= limit)):
                            raise FloatingPointError(f'CMuon NS4 exceeded the numerical guard for {tuple(chunk.shape)}')
                    patches.append(ortho * (-lr * 0.2 * math.sqrt(max(chunk.shape))))
                if wd:
                    p.mul_(1 - lr * wd)
                for target, update in zip(p.chunk(count, dim=0), patches, strict=True):
                    target.add_(update)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group['algorithm'] == 'cmuon':
                self._cmuon(group)
            else:
                self._adamw(group)
        return loss


def configure_hybrid(model, training):
    """Keep HDM's scheduler and original XUT/connector LR anchors."""
    from transformers import get_scheduler

    policy = training['hybrid_cmuon']
    original = training['optimizer']
    if original.get('fused') or set(original) - {'lr','betas','eps','weight_decay','fused'}:
        raise ValueError('HDM hybrid requires a supported non-fused optimizer config')
    xut_lr = float(original['lr'])
    connector_lr = float(training.get('connector_lr', xut_lr))
    cmul = float(policy.get('cmuon_lr_multiplier', 1.0))
    amul = float(policy.get('adamw_lr_multiplier', 0.6))
    momentum = float(policy.get('momentum', 0.95))
    if not all(math.isfinite(v) and v > 0 for v in (xut_lr, connector_lr, cmul, amul)) or not 0 <= momentum < 1:
        raise ValueError('Invalid hybrid optimizer LR/momentum')
    groups = [dict(name=name, algorithm=algo, lr=lr, params=[], param_names=[]) for name, algo, lr in (
        ('xut_cmuon','cmuon',xut_lr*cmul), ('xut_adamw','adamw',xut_lr*amul), ('connector_adamw','adamw',connector_lr*amul))]
    groups[0].update(chunk_counts=[], momentum=momentum)
    connector = {id(p) for p in model.connector.parameters()}
    dit = {id(p) for p in model.dit.parameters()}
    source_adapter = {id(p) for p in model.source_adapter.parameters()} if getattr(model, 'source_adapter', None) is not None else set()
    supported = connector | dit | source_adapter
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name = canonical_name(name)
        if id(p) not in supported:
            raise ValueError(f'Unexpected trainable parameter outside HDM XUT/connector/source adapter: {name}')
        count = cmuon_chunks(name) if id(p) in dit else 0
        if count:
            group = groups[0]
            group['chunk_counts'].append(count)
        else:
            group = groups[2] if id(p) in connector else groups[1]
        group['params'].append(p)
        group['param_names'].append(name)
    if not all(g['params'] for g in groups):
        raise ValueError('Missing expected HDM optimizer groups')
    opt = HybridCMuon(groups, betas=tuple(original.get('betas',(0.9,0.999))),
                      eps=float(original.get('eps',1e-8)), weight_decay=float(original.get('weight_decay',0.01)))
    if os.environ.get('LOCAL_RANK','0') == '0':
        print('HYBRID_ROUTING',json.dumps({g['name']:dict(tensors=len(g['params']),parameters=sum(p.numel() for p in g['params']),lr=g['lr']) for g in groups}),flush=True)
    schedule = dict(training['scheduler'])
    minimum = schedule.pop('min_lr_ratio', None)
    if minimum is not None:
        schedule['name'] = 'cosine_with_min_lr'
        schedule['scheduler_specific_kwargs'] = {'min_lr_rate': minimum}
    scheduler = get_scheduler(num_training_steps=training['max_steps'],optimizer=opt,**schedule)
    return {'optimizer':opt, 'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

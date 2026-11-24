"""Check the experimental packed XUT against the stock path on identical TREAD draws."""
import argparse
import json
from pathlib import Path
import runpy
from types import MethodType, SimpleNamespace

import torch


def errors(reference, candidate):
    a, b = reference.float(), candidate.float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError('Invalid comparison tensor shape or nonfinite result')
    return float((a - b).norm() / a.norm().clamp_min(1e-8)), float((a - b).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fp32', 'fp16', 'bf16'), required=True)
    parser.add_argument('--backend', choices=('math', 'serial-flash', 'varlen'), required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tread', type=float, choices=(0.0, 0.5), default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--grouped-linears', action='store_true')
    group.add_argument('--grouped-attn-out', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--precision-casts', action='store_true', help='Preserve eager rounding in all XUT compile regions')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--flash-launcher', type=Path, default=Path(__file__).with_name('hdm-dtk-train.py'))
    parser.add_argument('--models-stage', type=Path, required=True, help='YAML overlay with local model paths')
    args = parser.parse_args()
    dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[args.mode]
    if (dtype == torch.float32) != (args.backend == 'math'):
        parser.error('FP32 uses math; half precision uses a Flash backend')
    if args.precision_casts and not args.compile:
        parser.error('--precision-casts requires --compile')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    for setting in ('flash', 'mem_efficient', 'cudnn'):
        getattr(torch.backends.cuda, f'enable_{setting}_sdp')(False)
    torch.backends.cuda.enable_math_sdp(True)
    if dtype != torch.float32:
        runpy.run_path(str(args.flash_launcher))['use_vendor_flash']()
    from sakurahdm.models import MobileSakuraHDM, _load_hdm
    from sakurahdm.train import read_config
    stages = ['configs/hdm-base.yaml', 'configs/hdm-512.yaml', str(args.models_stage)]
    if args.compile:
        stages.append('configs/dtk-compile.yaml')
    conf = read_config(stages)['model']['hdm']
    conf['tread']['enabled'] = args.tread > 0.0
    if args.precision_casts:
        from torch._inductor import list_mode_options
        opts = conf['xut']['COMPILE_ARGS']
        mode = opts.pop('mode')
        opts['options'] = {**list_mode_options(mode), **opts.get('options', {}), 'emulate_precision_casts': True}
    torch.manual_seed(args.seed)
    dit = _load_hdm(conf).to(device).train()
    assert dit.model.use_tread and dit.model.dropout_ratio == args.tread
    model = SimpleNamespace(dit=dit, source_adapter=None)
    model.denoise = MethodType(MobileSakuraHDM.denoise, model)
    from xut.modules.axial_rope import make_axial_pos
    from xut_packed_pilot import Layout, install
    params = dict(dit.named_parameters())
    selected = {f'w:{n}': p for n, p in params.items() if any(s in n for s in (
        'in_patch.', 'out_patch.', 'ctx_proj.', 'prev_tread_trns.blocks.0.mlp.',
        'enc_blocks.0.0.attn.qkv.', 'dec_blocks.0.0.xattn.kv.', 'enc_blocks.0.0.attn.rope.',
        'prev_tread_trns.blocks.0.attn.out.', 'dec_blocks.0.0.xattn.out.',
        'shared_adaln_attn.3.', 'post_tread_trns.blocks.0.mlp.'))}
    assert len(selected) >= 12
    lengths = (64, 213, 17, 107, 213, 37, 512, 77) if dtype != torch.float32 else (64, 213, 17)
    b = len(lengths)
    latent = torch.randn(b, 4, 64, 64, device=device, requires_grad=True)
    timestep = torch.linspace(.15, .8, b, device=device)
    aspect = torch.linspace(.8, 1.2, b, device=device)
    pos = make_axial_pos(64, 64, device=device).expand(b, -1, -1).clone()
    pos[1::2].mul_(.8)
    # Preserve non-sorted group insertion order, both repeated and unique lengths.
    context = tuple((torch.tensor([i for i, n in enumerate(lengths) if n == l], device=device),
                     torch.randn(lengths.count(l), l, dit.config.ctx_dim, device=device, requires_grad=True))
                    for l in dict.fromkeys(lengths))
    target = torch.randn_like(latent)
    rng = torch.cuda.get_rng_state(device)
    loss_scale = 128 if dtype == torch.float16 else 1
    variables = {'latent': latent, **{f'text:{x.shape[1]}': x for _, x in context}, **selected}

    def forward(ctx, image=latent):
        torch.cuda.set_rng_state(rng, device)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            return model.denoise(image, timestep, ctx, pos_map=pos, aspect_ratio=aspect)

    def grads(fixed=False):
        output = forward(context)
        objective = ((output.float() * target).sum() / (target.numel() ** 0.5)
                     if fixed else (output.float() - target).square().mean())
        values = torch.autograd.grad(objective * loss_scale, list(variables.values()))
        return {'output': output.detach(), **{k: v.float()/loss_scale for k,v in zip(variables, values)}}, torch.cuda.get_rng_state(device)

    def capture_route():
        # Keys are original batch-row IDs. Image coordinates are unique and
        # text coordinates are zero; require the exact stock layout by row.
        seen = []
        def save(_path, _args, kwargs):
            seen.append(kwargs['pos_map'].detach().reshape(-1,2))
        handle = dit.model.backbone.register_forward_pre_hook(save, with_kwargs=True)
        try:
            with torch.no_grad():
                forward(context)
        finally:
            handle.remove()
        per_group_order = [i for idx,_ in context for i in idx.tolist()]
        keep = 1024 - int(1024 * args.tread)
        chunks = torch.cat(seen).split(tuple(keep + lengths[i] for i in per_group_order))
        return dict(zip(per_group_order, chunks, strict=True))

    def report(a, b):
        return {k: errors(v, b[k])[0] for k,v in a.items()}

    def summary(e):
        return {'output': e['output'], 'latent': e['latent'],
                'text_max': max(v for k,v in e.items() if k.startswith('text:')),
                'weight_max': max(v for k,v in e.items() if k.startswith('w:'))}

    def trace():
        prefixes = ('prev_tread_trns.blocks.0', 'backbone.enc_blocks.0.0', 'post_tread_trns.blocks.0')
        nodes = {'in_patch.proj', 'ctx_proj', 'time_emb', 'shared_adaln_attn', 'out_patch.proj'}
        for prefix in prefixes:
            nodes.update(prefix + suffix for suffix in ('', '.attn_pre_norm', '.attn.qkv', '.attn', '.mlp.w12', '.mlp', '.mlp_pre_norm'))
        observed = {}
        def capture(name):
            def save(_m, _args, output):
                v = output[0] if isinstance(output, tuple) else output
                v = v.flatten(2).transpose(1, 2) if name == 'in_patch.proj' else v
                observed.setdefault(name, []).append(v.detach().reshape(-1, v.shape[-1]).contiguous())
            return save
        handlers = [module.register_forward_hook(capture(name)) for name,module in dit.model.named_modules() if name in nodes]
        try:
            with torch.no_grad():
                forward(context)
        finally:
            for handler in handlers:
                handler.remove()
        return {name: torch.cat(tensors) for name,tensors in observed.items()}

    # A direct check that using precomputed lengths keeps the original selection identities.
    small = Layout.create((7, 8), (1, 1), device)
    left = small.route_indices(5, 2, device)
    torch.cuda.set_rng_state(rng, device)
    right = torch.cat(tuple(torch.cat(((torch.randperm(5, device=device)<2).nonzero().flatten(),
                                     torch.arange(5, n, device=device)))+start for n,start in ((7,0),(8,7))))
    assert torch.equal(left,right)
    stock_route = capture_route()
    ref, rng_ref = grads()
    repeat, _ = grads()
    fixed_ref, _ = grads(fixed=True)
    baseline_trace = trace() if args.trace else None
    linear_policy = 'attn-out' if args.grouped_attn_out else args.grouped_linears
    install(model, attention=args.backend, grouped_linears=linear_policy)
    assert all(p is dict(dit.named_parameters())[k] for k,p in params.items())
    packed_route = capture_route()
    route_match = all(torch.equal(stock_route[k], packed_route[k]) for k in stock_route)
    cand, rng_cand = grads()
    fixed_cand, _ = grads(fixed=True)
    e = report(ref, cand)
    results, repeats = summary(e), summary(report(ref, repeat))
    fixed = summary(report(fixed_ref, fixed_cand))
    # Perturb one of the two length-213 samples in the half-precision fixture.
    row = 1
    others = torch.arange(b, device=device) != row
    with torch.no_grad():
        a = forward(context)
        again = forward(context)
        perturbed = tuple((idx, t + (idx==row)[:,None,None] * t.sin() * .5) for idx,t in context)
        image = latent + (~others)[:,None,None,None] * latent.cos() * .25
        changed = forward(perturbed, image)
    noise = (a-again).abs().flatten(1).amax(1)
    change = (a-changed).abs().flatten(1).amax(1)
    isolated = bool((change[others] <= noise[others].clamp_min(1.e-6)).all() and change[row] > noise[row].clamp_min(1.e-6))
    leakage = float(change[others].max())
    output = forward(context)
    inputs = (latent, *(x for _,x in context))
    g_image, *g_text = torch.autograd.grad(output[row].float().square().mean() * loss_scale, inputs)
    off_row = [g_image[others], *(g[idx != row] for (idx, _), g in zip(context, g_text, strict=True))]
    if not all(bool(torch.isfinite(g).all()) for g in (g_image, *g_text)):
        raise ValueError('Nonfinite gradient in sample-isolation backward')
    backward_leakage = max(float(g.abs().max()) for g in off_row if g.numel()) / loss_scale
    rng_match = torch.equal(rng_ref, rng_cand)
    def within(s):
        return s['output'] < .01 and max(s.values()) < .035
    accepted = within(results) and within(repeats) and within(fixed) and isolated and backward_leakage == 0 and rng_match and route_match
    print('PACK_GATE', json.dumps(dict(mode=args.mode, backend=args.backend, device=args.device, tread=args.tread,
        compile=args.compile, precision_casts=args.precision_casts, seed=args.seed,
        grouped_linears=linear_policy, accepted=accepted, isolated=isolated,
        rng_match=rng_match, route_match=route_match, backward_leakage=backward_leakage,
        other_rows_change=leakage, other_rows_repeat=float(noise[others].max()), repeat=repeats, result=results, fixed_vjp=fixed,
        largest=[(k, v, errors(ref[k],cand[k])[1], float(ref[k].float().norm()))
                 for k,v in sorted(e.items(), key=lambda item: -item[1])[:6]])), flush=True)
    if baseline_trace is not None:
        packed_trace = trace()
        assert baseline_trace.keys() == packed_trace.keys()
        print('PACK_TRACE', json.dumps({k: dict(zip(('rel_l2','max_abs'),errors(v,packed_trace[k])))
                                      for k,v in baseline_trace.items()}), flush=True)
    return 0 if accepted else 3


if __name__ == '__main__':
    raise SystemExit(main())

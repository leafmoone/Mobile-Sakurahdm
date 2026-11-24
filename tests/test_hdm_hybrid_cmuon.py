import importlib.util
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class HybridCMuonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        try:
            import hdm_hybrid_cmuon as hybrid
        finally:
            sys.path.pop(0)
        cls.torch = torch
        cls.make = staticmethod(hybrid.HybridCMuon)
        cls.ns = staticmethod(hybrid.ns4)
        cls.chunks = staticmethod(hybrid.cmuon_chunks)

    def test_semantic_allowlist(self):
        block = 'dit.model.backbone.dec_blocks.2.0.'
        for name, count in [('attn.qkv.weight', 3), ('xattn.kv.weight', 2), ('mlp.w12.weight', 2),
                            ('mlp.w3.weight', 1), ('xattn_pre_norm.norm.weight', 0), ('mlp.w3.bias', 0)]:
            self.assertEqual(self.chunks(block + name), count)
        self.assertEqual(self.chunks('dit.model.shared_adaln_ffw.3.weight'), 3)
        self.assertEqual(self.chunks('dit.model.prev_tread_trns.blocks.0.attn.out.weight'), 1)
        self.assertEqual(self.chunks('dit._orig_mod.model.backbone._orig_mod.enc_blocks.0.0.attn.qkv.weight'), 3)
        for name in ('dit.model.ctx_proj.weight','dit.model.in_patch.proj.weight','connector.mlp.w3.weight','dit.model.out_patch.proj.weight'):
            self.assertEqual(self.chunks(name),0)

    def _optimizer(self, p, q):
        return self.make([
            dict(params=[p],param_names=['a'],algorithm='cmuon',name='xut_cmuon',chunk_counts=[2],lr=0.03,momentum=0.95),
            dict(params=[q],param_names=['b'],algorithm='adamw',name='xut_adamw',lr=0.002),
        ],betas=(.9,.95),eps=1e-8,weight_decay=.005)

    def test_update_matches_chunk_reference_and_stock_adamw(self):
        t = self.torch
        p = t.nn.Parameter(t.zeros(6,4)); q = t.nn.Parameter(t.tensor([1.,2.,3.]))
        qp = t.nn.Parameter(q.detach().clone())
        p.grad = t.arange(24.).reshape(6,4)*.002
        q.grad = qp.grad = t.tensor([.4,.7,-.1])
        opt = self._optimizer(p,q)
        adam = t.optim.AdamW([qp],lr=.002,betas=(.9,.95),eps=1e-8,weight_decay=.005)
        opt.step(); adam.step()
        x = p.grad.bfloat16(); m = t.zeros_like(x).lerp(x,.05); x = x.lerp(m,.95)
        expected = t.cat([(-.03*.2*2)*self.ns(chunk,t.bfloat16) for chunk in x.chunk(2)]).float()
        t.testing.assert_close(p,expected)
        t.testing.assert_close(q,qp)
        self.assertEqual(opt.rescued_chunks,0)

    def test_checkpoint_restores_bf16_momentum_and_rejects_stock_adamw(self):
        import copy
        t = self.torch
        p = t.nn.Parameter(t.ones(6,4)); q = t.nn.Parameter(t.ones(3))
        p.grad = t.full_like(p,.02); q.grad = t.full_like(q,.01)
        opt = self._optimizer(p,q); opt.step()
        p2 = t.nn.Parameter(p.detach().clone()); q2 = t.nn.Parameter(q.detach().clone())
        opt2 = self._optimizer(p2,q2)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(opt2.state[p2]['momentum_buffer'].dtype,t.bfloat16)
        p2.grad = p.grad.clone(); q2.grad = q.grad.clone()
        opt.step(); opt2.step()
        t.testing.assert_close(p,p2); t.testing.assert_close(q,q2)
        with self.assertRaisesRegex(ValueError,'Not a compatible'):
            opt2.load_state_dict(t.optim.AdamW([p2,q2]).state_dict())
        wrong = self._optimizer(t.nn.Parameter(t.ones(8,4)), q2)
        with self.assertRaisesRegex(ValueError,'state shape'):
            wrong.load_state_dict(copy.deepcopy(opt.state_dict()))

    def test_grad_scaler_skips_both_groups_on_overflow(self):
        t = self.torch
        p, q = t.nn.Parameter(t.ones(6,4)), t.nn.Parameter(t.ones(3))
        opt = self._optimizer(p,q)
        scaler = t.amp.GradScaler('cpu',init_scale=128.)
        scaler.scale(p.sum()*float('inf')+q.sum()).backward()
        scaler.step(opt); scaler.update()
        self.assertEqual(scaler.get_scale(),64.)
        self.assertFalse(opt.state)
        t.testing.assert_close(p,t.ones_like(p)); t.testing.assert_close(q,t.ones_like(q))

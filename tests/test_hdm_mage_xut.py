import importlib.util
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class MageXUTTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        try:
            import hdm_mage_xut
        finally:
            sys.path.pop(0)
        cls.torch = torch
        cls.transfer = staticmethod(hdm_mage_xut.transfer_weights)

    def test_chunk_order_and_input_isolation(self):
        t = self.torch
        old, new = 4, 6
        for name, chunks, width, wider, old_in, new_in in (
            ('model.a.attn.qkv.weight', 3, old, new, old, new),
            ('model.a.xattn.kv.weight', 2, old, new, old, new),
            ('model.a.mlp.w12.weight', 2, 8, 10, old, new),
            ('model.a.mlp.w3.weight', 1, old, new, 8, 10),
            ('model.shared_adaln_attn.3.weight', 3, old, new, 16, 24),
        ):
            with self.subTest(name=name):
                src = t.arange(chunks * width * old_in).reshape(chunks * width, old_in).float()
                dst = t.ones(chunks * wider, new_in)
                self.transfer({name: src}, {name: dst}, old, new, 8, 10)
                for original, widened in zip(src.chunk(chunks), dst.chunk(chunks), strict=True):
                    t.testing.assert_close(widened[:width, :old_in], original)
                    self.assertEqual(widened[:width, old_in:].count_nonzero().item(), 0)
                    self.assertGreater(widened[width:].count_nonzero().item(), 0)

    def test_timestep_sin_cos_axes_and_frequencies(self):
        t = self.torch
        old, new = 4, 6
        k = 'model.time_emb.proj.0.weight'
        f = 'model.time_emb.freqs'
        src = t.arange(old * old).reshape(old, old).float()
        dst = t.ones(new, new)
        fr = t.tensor([[1., .02]])
        fr_dst = t.tensor([[1., .18, .03]])
        self.transfer({k: src, f: fr}, {k: dst, f: fr_dst}, old, new, 8, 10)
        t.testing.assert_close(dst[:old, [0, 1, 3, 4]], src)
        self.assertEqual(dst[:old, [2, 5]].count_nonzero().item(), 0)
        t.testing.assert_close(fr_dst, t.tensor([[1., .02, .03]]))

    def test_rope_heads_and_fresh_latent_interfaces(self):
        t = self.torch
        r = 'model.a.attn.rope.freqs'
        w = 'model.in_patch.proj.weight'
        src = t.full((2, 4, 2), 2.)
        new = t.ones(3, 4, 2)
        in_patch = t.ones(6, 128, 1, 1)
        self.transfer({r: src, w: t.zeros(4, 4, 2, 2)}, {r: new, w: in_patch}, 4, 6, 8, 10)
        t.testing.assert_close(new[:2], src)
        t.testing.assert_close(in_patch, t.ones_like(in_patch))

    def test_norm_scaling_and_unknown_parameter_guard(self):
        t = self.torch
        r = 'model.a.attn_pre_norm.norm.weight'
        l = 'model.shared_adaln_attn.0.weight'
        a, b = t.ones(6), t.ones(6)
        self.transfer({r: t.ones(4), l: t.arange(4.)}, {r: a, l: b}, 4, 6, 8, 10)
        t.testing.assert_close(a[:4], t.full((4,), (4 / 6) ** .5))
        t.testing.assert_close(b[:4], t.arange(4.))
        with self.assertRaisesRegex(ValueError, 'Unhandled widened tensor'):
            self.transfer({'x': t.ones(4)}, {'x': t.ones(6)}, 4, 6, 8, 10)

    def test_malformed_or_already_widened_source_is_rejected(self):
        t = self.torch
        for name, src, dst in (
            ('model.time_emb.freqs', t.ones(1, 1), t.ones(1, 3)),
            ('model.a.attn.rope.freqs', t.ones(1, 4, 2), t.ones(3, 4, 2)),
            ('model.a.attn.qkv.weight', t.ones(18, 6), t.ones(18, 6)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.transfer({name: src}, {name: dst}, 4, 6, 8, 10)

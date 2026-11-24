import importlib.util
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class PackedLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        try:
            import torch
            from xut_packed_pilot import Layout
        finally:
            sys.path.pop(0)
        cls.torch = torch
        cls.layout = Layout.create((7, 7, 9), (2, 1), torch.device('cpu'))

    def test_boundaries_and_image_positions(self):
        layout = self.layout
        self.assertEqual(layout.cu.tolist(), [0, 7, 14, 23])
        self.assertEqual(layout.group_sizes, (14, 9))
        self.assertEqual(layout.rows.tolist(), [0] * 7 + [1] * 7 + [2] * 9)
        self.assertEqual(layout.image_indices(4, 'cpu').tolist(), [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17])

    def test_routes_match_stock_and_preserve_rng(self):
        torch, layout = self.torch, self.layout
        with torch.random.fork_rng(devices=[]):
            for keep in (2, 4):
                torch.manual_seed(42)
                actual = layout.route_indices(4, keep, 'cpu')
                after = torch.get_rng_state()
                torch.manual_seed(42)
                expected = []
                for start, count in zip((0, 7, 14), layout.lengths):
                    mask = torch.cat((torch.randperm(4) < keep, torch.ones(count - 4, dtype=torch.bool)))
                    expected.extend(torch.arange(start, start + count)[mask].tolist())
                self.assertEqual(actual.tolist(), expected)
                self.assertTrue(torch.equal(torch.get_rng_state(), after))

    def test_bypassed_tokens_survive_reinsertion(self):
        torch, layout = self.torch, self.layout
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            chosen = layout.route_indices(4, 2, 'cpu')
        tokens = torch.arange(layout.total).reshape(1, layout.total, 1).float().requires_grad_()
        updated = tokens.index_copy(1, chosen, 2 * tokens.index_select(1, chosen))
        updated.sum().backward()
        expected_grad = torch.ones_like(tokens)
        expected_grad[:, chosen, :] = 2
        self.assertTrue(torch.equal(tokens.grad, expected_grad))

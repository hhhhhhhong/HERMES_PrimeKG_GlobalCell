"""Nonvacuous residual limits, finite zero-init gradients and symmetric decode."""
import unittest
import torch
from torch import nn
from dual_branch.stable_prediction import StableTransformerModel, bounded_delta
from dual_branch.transformer_prediction import PredictionTokenInteraction


class TinyHead(nn.Module):
    def forward_once(self, a, b, cell):
        return (a * b + a * cell).sum(-1).sigmoid()


class StablePredictionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_huge_raw_increments_remain_bounded_and_combined_cap_holds(self):
        own = torch.randn(7, 3, 16)
        a = bounded_delta(1e4 * torch.randn_like(own), own, .25)
        b = bounded_delta(1e4 * torch.randn_like(own), own, .10)
        norm = lambda x: torch.linalg.vector_norm(x, dim=-1)
        self.assertTrue(bool((norm(a) <= .250001 * norm(own)).all()))
        self.assertTrue(bool((norm(b) <= .100001 * norm(own)).all()))
        self.assertTrue(bool((norm(a + b) <= .350002 * norm(own)).all()))

    def test_zero_initial_delta_retains_finite_nonzero_gradient_and_detaches_scale(self):
        own = torch.randn(3, 3, 16, requires_grad=True)
        raw = torch.zeros_like(own, requires_grad=True)
        out = bounded_delta(raw, own, .25)
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        torch.testing.assert_close(raw.grad, torch.ones_like(raw))
        self.assertIsNone(own.grad)
        torch.testing.assert_close(out, raw, atol=0, rtol=0)
        empty = bounded_delta(torch.randn_like(own), torch.zeros_like(own), .25)
        self.assertEqual(float(empty.abs().max()), 0.)

    def test_active_transformer_is_symmetric_and_has_real_Cell_gradient(self):
        model = StableTransformerModel.__new__(StableTransformerModel)
        nn.Module.__init__(model)
        model.prediction_transformer = PredictionTokenInteraction(dim=16, heads=4, feedforward_dim=32, dropout=0.)
        model.head = TinyHead()
        with torch.no_grad():
            model.prediction_transformer.output.weight.copy_(1e4 * torch.eye(16))
        model.eval()
        own = torch.randn(4, 3, 16, requires_grad=True)
        pred, delta, _ = model._decode_bounded(own, own)
        swapped = model.decode(own[:, [1, 0, 2]])
        torch.testing.assert_close(pred, swapped, atol=2e-6, rtol=0)
        self.assertGreater(float(delta.abs().max()), 0.)
        delta[:, 0, 0].sum().backward()
        self.assertGreater(float(own.grad[:, 2].abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()

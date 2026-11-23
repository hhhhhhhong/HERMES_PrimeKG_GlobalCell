import unittest
import torch
from dual_branch.numerical_safety import (
    finite_masked_softmax, SafeQueryGatedFusion, SafeConditionalGamma, _same_state_replacement)
from dual_branch.gated_mechanism import QueryGatedFusion
from global_cell.attention import ConditionalGamma


class MaskedSoftmaxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_all_masked_single_source_and_extremes_have_finite_backward(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logits = torch.tensor([[2., -3., 9.], [-60000., 60000., -20.], [0., 0., 0.]], dtype=dtype, requires_grad=True)
            mask = torch.tensor([[False, False, False], [True, False, False], [True, False, True]])
            weights = finite_masked_softmax(logits, mask)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertEqual(float(weights[~mask].abs().sum()), 0.)
            torch.testing.assert_close(weights[1], torch.tensor([1., 0., 0.], dtype=dtype), rtol=0, atol=0)
            torch.testing.assert_close(weights[2], torch.tensor([.5, 0., .5], dtype=dtype), rtol=0, atol=0)
            (weights.float() * torch.tensor([1., 2., 3.])).sum().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(float(logits.grad[~mask].abs().sum()), 0.)

    def test_unavailable_nonfinite_is_ignored_available_nonfinite_is_rejected(self):
        value = torch.tensor([[1., float('nan'), float('inf')]], requires_grad=True)
        mask = torch.tensor([[True, False, False]])
        weights = finite_masked_softmax(value, mask)
        self.assertEqual(weights.tolist(), [[1., 0., 0.]])
        weights.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        with self.assertRaises(FloatingPointError):
            finite_masked_softmax(value, torch.ones_like(mask))

    def test_fusion_forward_and_backward_fp16_bf16_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            module = SafeQueryGatedFusion(dim=8, hidden=16, dropout=0.).to(dtype).eval()
            sources = [torch.randn(4, 3, 8, dtype=dtype, requires_grad=True) for _ in range(3)]
            mask = torch.ones(4, 3, 3, dtype=torch.bool)
            mask[0, :, 1:] = False
            mask[1, :, 2] = False
            out, weights = module(*sources, source_available=mask, return_weights=True)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(float(weights[~mask].abs().sum()), 0.)
            (out.float() * torch.randn_like(out.float())).sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in sources))
            with self.assertRaises(ValueError):
                module(*sources, source_available=torch.zeros_like(mask))

    def test_legacy_fusion_parity_and_constructor_rng_preservation(self):
        old = QueryGatedFusion(dim=8, hidden=16, dropout=0.).eval()
        rng = torch.random.get_rng_state().clone()
        new = _same_state_replacement(old, SafeQueryGatedFusion, dim=8, hidden=16, dropout=0.)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        sources = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(3)]
        mask = torch.ones(4, 3, 3, dtype=torch.bool)
        mask[0, :, 1:] = False
        expected = old(*sources, source_available=mask)
        actual = new(*sources, source_available=mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
        grad_old = torch.autograd.grad((expected * torch.arange(8)).sum(), sources, retain_graph=True)
        grad_new = torch.autograd.grad((actual * torch.arange(8)).sum(), sources)
        for a, b in zip(grad_old, grad_new):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)

    def test_gamma_empty_evidence_and_mixed_precision_backward(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            gamma = SafeConditionalGamma(dim=8).to(dtype)
            own = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
            table = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
            mask = torch.ones(2, 3, 4, dtype=torch.bool)
            mask[0] = False
            out, weights = gamma(own, table, mask, capture=True)
            self.assertEqual(float(out[0].abs().sum()), 0.)
            self.assertEqual(float(weights[0].abs().sum()), 0.)
            self.assertTrue(torch.isfinite(out).all())
            (out.float()*torch.randn_like(out.float())).sum().backward()
            self.assertTrue(torch.isfinite(own.grad).all() and torch.isfinite(table.grad).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in gamma.parameters()))

    def test_gamma_dense_and_indexed_interfaces_match_legacy(self):
        old = ConditionalGamma(dim=8).eval()
        new = _same_state_replacement(old, SafeConditionalGamma, dim=8)
        own = torch.randn(2, 3, 8)
        values = torch.randn(2, 3, 4, 8)
        mask = torch.rand(2, 3, 4) > .3
        torch.testing.assert_close(new(own, values, mask)[0], old(own, values, mask)[0], rtol=0, atol=2e-6)
        indexed_values = torch.randn(5, 4, 8)
        indexed_mask = torch.rand(5, 4) > .3
        indices = torch.tensor([[0, 1, 2], [1, 3, 4]])
        torch.testing.assert_close(new(own, indexed_values, indexed_mask, indices=indices)[0],
            old(own, indexed_values, indexed_mask, indices=indices)[0], rtol=0, atol=2e-6)

    def test_CPU_autocast_bfloat16(self):
        fusion = SafeQueryGatedFusion(dim=8, hidden=16, dropout=0.)
        gamma = SafeConditionalGamma(dim=8)
        own = torch.randn(2, 3, 8, requires_grad=True)
        table = torch.randn(2, 3, 4, 8, requires_grad=True)
        available = torch.ones(2, 3, 4, dtype=torch.bool)
        available[0] = False
        source_mask = torch.ones(2, 3, 3, dtype=torch.bool)
        source_mask[0, :, 2] = False
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            context, _ = gamma(own, table, available)
            fused = fusion(own, own*.5, context, source_available=source_mask)
            loss = (fused.float()*torch.randn_like(fused.float())).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.isfinite(own.grad).all() and torch.isfinite(table.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for module in (fusion, gamma) for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()

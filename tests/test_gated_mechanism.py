import unittest
import torch
from dual_branch.gated_mechanism import QueryGatedFusion


class QueryGatedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.fusion = QueryGatedFusion(dim=8, hidden=16, dropout=0.).eval()
        self.sources = [torch.randn(5, 3, 8, requires_grad=True) for _ in range(3)]

    def test_first_backward_reaches_all_sources_alignments_and_query_gates(self):
        output, weights = self.fusion(*self.sources, return_weights=True)
        (output * torch.randn_like(output)).sum().backward()
        for value in self.sources:
            self.assertGreater(float(value.grad.norm()), 1e-8)
        for module in [self.fusion.query, self.fusion.drug_gate, self.fusion.cell_gate,
                       self.fusion.drug_alignment, self.fusion.cell_alignment]:
            self.assertTrue(all(p.grad is not None and float(p.grad.norm()) > 1e-8 for p in module.parameters()))
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=0, atol=2e-6)

    def test_empty_sources_have_zero_weight_and_perturbations_are_isolated(self):
        mask = torch.ones(5, 3, 3, dtype=torch.bool)
        mask[:, :, 2] = False
        reference, weights = self.fusion(*self.sources, source_available=mask, return_weights=True)
        self.assertEqual(float(weights[:, :, 2].abs().max()), 0.)
        changed = self.sources[:2] + [torch.randn_like(self.sources[2]) * 100]
        torch.testing.assert_close(self.fusion(*changed, source_available=mask), reference, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.fusion(*self.sources, source_available=torch.zeros_like(mask))

    def test_drug_exchange_batch_isolation_and_Cell_conditioning(self):
        reference, weights = self.fusion(*self.sources, return_weights=True)
        p = [1, 0, 2]
        swap, swap_weights = self.fusion(*(x[:, p] for x in self.sources), return_weights=True)
        torch.testing.assert_close(swap, reference[:, p], rtol=0, atol=2e-6)
        torch.testing.assert_close(swap_weights, weights[:, p], rtol=0, atol=2e-6)
        separate = torch.cat([self.fusion(*(x[i:i+1] for x in self.sources)) for i in range(5)])
        torch.testing.assert_close(separate, reference, rtol=0, atol=2e-6)
        changed = [x.detach().clone() for x in self.sources]
        changed[0][:, 2] += torch.linspace(-1, 1, 8)
        _, actual = self.fusion(*changed, return_weights=True)
        self.assertGreater(float((actual[:, :2] - weights[:, :2]).abs().max()), 1e-8)

    def test_strict_state_reload(self):
        other = QueryGatedFusion(dim=8, hidden=16, dropout=0.).eval()
        other.load_state_dict(self.fusion.state_dict(), strict=True)
        torch.testing.assert_close(other(*self.sources), self.fusion(*self.sources), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

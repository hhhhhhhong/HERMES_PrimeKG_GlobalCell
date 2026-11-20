"""Small invariants that complement the real full-graph correctness checks."""
import unittest
import torch
from dual_branch.model import ConditionalResidual, DualBranchModel
from global_cell.attention import ConditionalGamma


class FusionTests(unittest.TestCase):
    def test_main_model_rejects_unimplemented_modal_dimension(self):
        with self.assertRaisesRegex(ValueError, 'requires dim=256'):
            DualBranchModel({}, dim=128)

    def test_empty_knowledge_gradient_is_finite_with_open_output(self):
        fusion = ConditionalResidual(8)
        with torch.no_grad():
            fusion.drug_out.weight.copy_(torch.eye(8))
            fusion.cell_out.weight.copy_(torch.eye(8))
        own = torch.randn(3, 3, 8, requires_grad=True)
        knowledge = torch.zeros_like(own, requires_grad=True)
        final, delta, _ = fusion(own, knowledge)
        final.square().sum().backward()
        self.assertTrue(torch.isfinite(knowledge.grad).all())
        self.assertEqual(delta.abs().max().item(), 0.)

    def test_coder_uncased_vocabulary_preserves_drug_and_cell_identity(self):
        from dual_branch.prepare import build_tokenizer
        tokenizer = build_tokenizer()
        self.assertEqual(tokenizer.tokenize('Methotrexate'), tokenizer.tokenize('methotrexate'))
        self.assertNotIn(tokenizer.unk_token, tokenizer.tokenize('Methotrexate'))
        self.assertNotEqual(tokenizer.encode('A549'), tokenizer.encode('A498'))

    def test_zero_projection_identity_and_gradient_order(self):
        torch.manual_seed(0)
        fusion = ConditionalResidual(8)
        own = torch.randn(4, 3, 8, requires_grad=True)
        knowledge = torch.randn(4, 3, 8, requires_grad=True)
        final, delta, _ = fusion(own, knowledge)
        torch.testing.assert_close(final, own, rtol=0, atol=0)
        self.assertEqual(delta.abs().max().item(), 0.)
        final.square().sum().backward()
        self.assertGreater(fusion.drug_out.weight.grad.norm().item(), 0)
        self.assertGreater(fusion.cell_out.weight.grad.norm().item(), 0)
        self.assertEqual(knowledge.grad.abs().max().item(), 0.)
        self.assertEqual(fusion.drug_router.weight.grad.abs().max().item(), 0.)

    def test_open_fusion_is_drug_exchange_equivariant_and_empty_safe(self):
        torch.manual_seed(1)
        fusion = ConditionalResidual(8)
        with torch.no_grad():
            fusion.drug_out.weight.copy_(torch.eye(8))
            fusion.cell_out.weight.copy_(torch.eye(8))
        own, knowledge = torch.randn(5, 3, 8), torch.randn(5, 3, 8)
        permutation = [1, 0, 2]
        reference = fusion(own, knowledge)[0]
        swapped = fusion(own[:, permutation], knowledge[:, permutation])[0]
        torch.testing.assert_close(swapped, reference[:, permutation], rtol=0, atol=0)
        empty, delta, _ = fusion(own, torch.zeros_like(knowledge))
        torch.testing.assert_close(empty, own, rtol=0, atol=0)
        self.assertEqual(delta.abs().max().item(), 0.)

    def test_modal_cell_conditions_reading_with_fixed_semantic_evidence(self):
        torch.manual_seed(3)
        gamma = ConditionalGamma(dim=8).eval()
        own = torch.randn(4, 3, 8)
        table = torch.randn(6, 5, 8)
        ids = torch.tensor([[0, 1, 4], [1, 2, 4], [0, 3, 5], [2, 3, 5]])
        mask = torch.ones(6, 5, dtype=torch.bool)
        _, reference = gamma(own, table, mask, capture=True, indices=ids)
        changed = own.clone()
        changed[:, 2] += torch.linspace(-1, 1, 8)
        _, actual = gamma(changed, table, mask, capture=True, indices=ids)
        self.assertGreater((actual - reference).abs().max().item(), 1e-7)


if __name__ == '__main__':
    unittest.main()

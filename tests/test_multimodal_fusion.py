"""Joint fusion must use both sources immediately and preserve query invariants."""
import unittest
import torch
from dual_branch.multimodal_fusion import ConcatenationFusion, JointTokenInteraction


class MultimodalFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.fusion = ConcatenationFusion(dim=16, dropout=0.).eval()
        self.interaction = JointTokenInteraction(dim=16, dropout=0.).eval()

    def test_both_sources_and_transformer_learn_in_the_first_backward(self):
        modal = torch.randn(4, 3, 16, requires_grad=True)
        knowledge = torch.randn_like(modal, requires_grad=True)
        output = self.interaction(self.fusion(modal, knowledge))
        (output * torch.randn_like(output)).sum().backward()
        for grad in (modal.grad, knowledge.grad,
                     self.fusion.drug[0].weight.grad, self.fusion.cell[0].weight.grad,
                     self.interaction.encoder.self_attn.in_proj_weight.grad):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()), 1e-8)
        self.assertGreater(float(self.fusion.drug[0].weight.grad[:, 16:].norm()), 1e-8)
        self.assertGreater(float(self.fusion.cell[0].weight.grad[:, 16:].norm()), 1e-8)

    def test_exchange_equivariance_and_batch_independence(self):
        modal, knowledge = torch.randn(4, 3, 16), torch.randn(4, 3, 16)
        perm = [1, 0, 2]
        reference = self.interaction(self.fusion(modal, knowledge))
        swapped = self.interaction(self.fusion(modal[:, perm], knowledge[:, perm]))
        torch.testing.assert_close(swapped, reference[:, perm], rtol=0, atol=2e-6)
        separate = torch.cat([self.interaction(self.fusion(m[None], k[None]))
                              for m, k in zip(modal, knowledge)])
        torch.testing.assert_close(separate, reference, rtol=0, atol=2e-6)
        changed = knowledge.clone()
        changed[1:] = torch.randn_like(changed[1:])
        torch.testing.assert_close(self.interaction(self.fusion(modal, changed))[0],
                                   reference[0], rtol=0, atol=0)

    def test_empty_source_and_true_modality_interventions(self):
        modal, knowledge = torch.randn(4, 3, 16), torch.randn(4, 3, 16)
        empty = self.fusion(modal, torch.zeros_like(knowledge))
        self.assertTrue(bool(torch.isfinite(empty).all()))
        full = self.fusion(modal, knowledge)
        self.assertGreater(float((full - empty).abs().max()), 1e-5)
        self.assertGreater(float((full - self.fusion(torch.zeros_like(modal), knowledge)).abs().max()), 1e-5)

    def test_cell_has_an_attention_path_to_drug_and_checkpoint_reloads(self):
        modal = torch.randn(4, 3, 16, requires_grad=True)
        knowledge = torch.randn_like(modal)
        output = self.interaction(self.fusion(modal, knowledge))
        (output[:, 0] * torch.randn_like(output[:, 0])).sum().backward()
        self.assertGreater(float(modal.grad[:, 2].norm()), 1e-8)
        other_fusion = ConcatenationFusion(16, dropout=0.).eval()
        other_interaction = JointTokenInteraction(16, dropout=0.).eval()
        other_fusion.load_state_dict(self.fusion.state_dict(), strict=True)
        other_interaction.load_state_dict(self.interaction.state_dict(), strict=True)
        torch.testing.assert_close(other_interaction(other_fusion(modal, knowledge)), output, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

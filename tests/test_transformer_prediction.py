"""Check active interactions, symmetry, gradients and prediction preservation."""
import unittest

import torch

from dual_branch.transformer_prediction import PredictionTokenInteraction


class PredictionTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def active_block(self, dim=16):
        block = PredictionTokenInteraction(
            dim=dim, heads=4, feedforward_dim=32, dropout=0.2).eval()
        with torch.no_grad():
            block.output.weight.copy_(0.1 * torch.eye(dim))
        return block

    def test_initial_identity_and_staged_attention_gradients(self):
        block = PredictionTokenInteraction(dim=16, heads=4, feedforward_dim=32, dropout=0.)
        x = torch.randn(5, 3, 16)
        target = torch.randn_like(x)
        optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
        out, delta = block(x, return_delta=True)
        torch.testing.assert_close(out, x, rtol=0, atol=0)
        self.assertEqual(delta.abs().max().item(), 0.)
        (out - target).square().mean().backward()
        self.assertGreater(block.output.weight.grad.norm().item(), 0.)
        self.assertEqual(block.encoder.self_attn.in_proj_weight.grad.norm().item(), 0.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (block(x) - target).square().mean().backward()
        self.assertGreater(block.encoder.self_attn.in_proj_weight.grad.norm().item(), 0.)
        self.assertGreater(block.role.weight.grad.norm().item(), 0.)

    def test_active_drug_exchange_and_sample_independence(self):
        block = self.active_block()
        x = torch.randn(5, 3, 16)
        swap = [1, 0, 2]
        reference = block(x)
        self.assertGreater((reference - x).abs().max().item(), 1e-5)
        torch.testing.assert_close(block(x[:, swap]), reference[:, swap], rtol=0, atol=2e-6)
        single = torch.cat([block(row.unsqueeze(0)) for row in x])
        torch.testing.assert_close(single, reference, rtol=0, atol=2e-6)
        changed = x.clone()
        changed[1:] += 10 * torch.randn_like(changed[1:])
        torch.testing.assert_close(block(changed)[0], reference[0], rtol=0, atol=0)

    def test_cell_token_has_a_real_attention_path_to_drug_tokens(self):
        block = self.active_block()
        x = torch.randn(4, 3, 16, requires_grad=True)
        block(x)[:, 0].square().sum().backward()
        self.assertGreater(x.grad[:, 2].abs().max().item(), 1e-8)

    def test_original_author_head_identity_and_active_symmetry(self):
        from global_cell.common import author_imports
        author_imports()
        from HERMES.model import Consolidator
        head = Consolidator(256).eval()
        block = PredictionTokenInteraction().eval()
        x = torch.randn(4, 3, 256)

        def decode(v):
            a, b, cell = v.unbind(1)
            return 0.5 * (head.forward_once(a, b, cell)
                          + head.forward_once(b, a, cell))

        torch.testing.assert_close(decode(block(x)), decode(x), rtol=0, atol=0)
        with torch.no_grad():
            block.output.weight.copy_(torch.eye(256) * .1)
        torch.testing.assert_close(decode(block(x[:, [1, 0, 2]])),
                                   decode(block(x)), rtol=0, atol=2e-6)
        self.assertGreater((decode(block(x)) - decode(x)).abs().max().item(), 1e-8)

    def test_reloaded_active_block_matches(self):
        block = self.active_block()
        restored = PredictionTokenInteraction(
            dim=16, heads=4, feedforward_dim=32, dropout=.2).eval()
        restored.load_state_dict(block.state_dict(), strict=True)
        x = torch.randn(5, 3, 16)
        torch.testing.assert_close(restored(x), block(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

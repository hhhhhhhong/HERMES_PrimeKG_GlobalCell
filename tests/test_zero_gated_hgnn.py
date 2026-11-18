"""Identity at initialization, real state feedback, and gate-opening gradients."""
import unittest

import torch

from global_cell.common import author_imports
from global_cell.network import GlobalCellModel, evaluate, train_epoch
from models.zero_gated_hgnn import ZeroGatedHypergraphModel
from test_global_cell import fixture


class ZeroGatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        author_imports()

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.graph = fixture()
        self.cache = dict(num_drug=2, num_cell=2, cell_input_dim=3,
            inputs=dict(drug_x=torch.randn(6, 75),
                drug_adj=torch.tensor([[0,1,1,2,3,4,4,5], [1,0,2,1,4,3,5,4]]),
                ibatch=torch.tensor([0,0,0,1,1,1]), cells=torch.randn(2, 3)))
        self.coder = torch.randn(3, 768)
        torch.manual_seed(0)
        self.model = ZeroGatedHypergraphModel(self.cache, self.graph, self.coder)
        self.rows = torch.tensor([[0.,1.,2.,1.], [1.,0.,3.,0.], [0.,1.,3.,1.]])

    def test_every_round_initially_identity_and_existing_initialization_preserved(self):
        torch.manual_seed(0)
        original = GlobalCellModel(self.cache, self.graph, self.coder)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(self.model.state_dict()[name], value, rtol=0, atol=0)
        self.model.eval()
        x = torch.randn(7, 256)
        slots, h = self.model.propagate(x, capture=True)
        torch.testing.assert_close(h, x, rtol=0, atol=0)
        torch.testing.assert_close(slots, x[self.model.slot_nodes], rtol=0, atol=0)
        for layer in self.model.propagation_audit['layers']:
            self.assertEqual(layer['delta_rms'], 0.)
            self.assertEqual(layer['identity_max_error'], 0.)

    def test_updated_external_and_root_nodes_feed_the_next_round(self):
        model = self.model.eval()
        for g in model.propagation_gates:
            g.data.fill_(0.1)
        inputs = []
        hooks = [m.register_forward_pre_hook(lambda m, args: inputs.append(args[0].detach().clone()))
                 for m in [model.encoder]+list(model.extra_encoders)]
        # One additional isolated node must remain exactly unchanged.
        x = torch.randn(8, 256)
        slots, updated = model.propagate(x, capture=True)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(inputs), 3)
        self.assertGreater(float((inputs[1][:4]-inputs[0][:4]).abs().max()), 0.)
        self.assertGreater(float((inputs[1][4:7]-inputs[0][4:7]).abs().max()), 0.)
        self.assertGreater(float((inputs[2]-inputs[1]).abs().max()), 0.)
        torch.testing.assert_close(updated[7], x[7], rtol=0, atol=0)
        # Changed intermediate protein input can reach a Drug after real updates.
        altered = x.clone(); altered[4] += torch.linspace(-1, 1, 256)
        altered_slots, _ = model.propagate(altered)
        drug_slots = self.model.slot_nodes < 2
        self.assertGreater(float((slots[drug_slots]-altered_slots[drug_slots]).abs().max()), 0.)

    def test_gate_first_gradient_then_branch_gradient_and_prediction_invariants(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.01)
        for epoch in [1, 2, 3]:
            _, gradients = train_epoch(model, optimizer, self.rows, self.rows[:, 3], chunk=2)
            model.verify_gradients(gradients, epoch)
            for gate in model.propagation_gates:
                self.assertGreater(float(gate.grad.norm()), 0.)
        prediction = evaluate(model, self.rows, chunk=3)
        torch.testing.assert_close(evaluate(model, self.rows, chunk=1), prediction)
        torch.testing.assert_close(evaluate(model, self.rows[:, [1,0,2,3]]), prediction)
        restored = ZeroGatedHypergraphModel(self.cache, self.graph, self.coder)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(evaluate(restored, self.rows), prediction)


if __name__ == '__main__':
    unittest.main()

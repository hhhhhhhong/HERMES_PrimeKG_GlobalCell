"""Verify optimizer isolation, matching initialization, and genuine updates."""
from pathlib import Path
import tempfile
import unittest

import torch

from global_cell.common import author_imports, read, write
from global_cell.network import evaluate, train_epoch
from models.zero_gated_checkpoint import CheckpointedZeroGatedModel
from scripts.knowledge_decay import parameter_groups, observed_adam_class, final_ablations
from test_global_cell import fixture


class KnowledgeDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        author_imports()

    def make_model(self):
        torch.manual_seed(0)
        graph = fixture()
        cache = dict(num_drug=2, num_cell=2, cell_input_dim=3,
            inputs=dict(drug_x=torch.randn(6, 75),
                drug_adj=torch.tensor([[0,1,1,2,3,4,4,5], [1,0,2,1,4,3,5,4]]),
                ibatch=torch.tensor([0,0,0,1,1,1]), cells=torch.randn(2, 3)))
        return CheckpointedZeroGatedModel(cache, graph, torch.randn(3, 768))

    def test_groups_cover_parameters_and_decay_only_backbone(self):
        model = self.make_model()
        groups, metadata = parameter_groups(model, 'knowledge_no_decay')
        self.assertEqual(len(groups), 2)
        for group in metadata:
            for name in group['names']:
                self.assertEqual(group['weight_decay'], .01 if name.startswith(('initializer.', 'head.')) else 0.)
        # A zero supervised gradient decays only the intended parameter group.
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.Adam(groups, lr=1e-4, weight_decay=.01)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
        changed_backbone = False
        for name, p in model.named_parameters():
            if name.startswith(('initializer.', 'head.')):
                changed_backbone |= not torch.equal(p, before[name])
            else:
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        self.assertTrue(changed_backbone)

    def test_matched_diagnostics_do_not_change_loss_updates_or_random_stream(self):
        a = self.make_model()
        # Reconstruct instead of deepcopy: checkpoint wrappers close over their
        # owning modules, so copying the instance would retain those closures.
        b = self.make_model()
        rows = torch.tensor([[0.,1.,2.,1.], [1.,0.,3.,0.], [0.,1.,3.,1.]])
        ordinary = torch.optim.Adam(a.parameters(), lr=1e-4, weight_decay=.01)
        with tempfile.TemporaryDirectory(dir='/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/tmp') as path:
            observed = observed_adam_class(torch.optim.Adam, b, Path(path), 'matched_decay')(
                b.parameters(), lr=1e-4, weight_decay=.01)
            for _ in range(3):
                rng = torch.get_rng_state()
                loss_a, _ = train_epoch(a, ordinary, rows, rows[:, 3], chunk=2)
                end = torch.get_rng_state()
                torch.set_rng_state(rng)
                loss_b, _ = train_epoch(b, observed, rows, rows[:, 3], chunk=2)
                self.assertAlmostEqual(loss_a, loss_b, places=7)
                self.assertTrue(torch.equal(end, torch.get_rng_state()))
                for pa, pb in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            torch.testing.assert_close(evaluate(a, rows), evaluate(b, rows), rtol=0, atol=0)

    def test_no_decay_still_learns_gates_and_branch_after_exact_identity(self):
        model = self.make_model()
        x = torch.randn(8, 256)
        slots, h = model.eval().propagate(x, capture=True)
        torch.testing.assert_close(h, x, rtol=0, atol=0)
        groups, _ = parameter_groups(model, 'knowledge_no_decay')
        optimizer = torch.optim.Adam(groups, lr=1e-4, weight_decay=.01)
        rows = torch.tensor([[0.,1.,2.,1.], [1.,0.,3.,0.], [0.,1.,3.,1.]])
        for epoch in (1, 2, 3):
            _, gradients = train_epoch(model, optimizer, rows, rows[:, 3], chunk=2)
            model.verify_gradients(gradients, epoch)

    def test_integer_validation_labels_and_fixed_checkpoint_interventions(self):
        from utils import get_metrics
        model = self.make_model()
        rows = torch.tensor([[0, 1, 2, 1], [1, 0, 3, 0], [0, 1, 3, 1]], dtype=torch.long)
        pred = evaluate(model, rows)
        auc, ap = get_metrics(rows[:, 3].numpy(), pred.numpy())
        with tempfile.TemporaryDirectory(dir='/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/tmp') as path:
            d = Path(path)
            torch.save(dict(epoch=1, model_state=model.state_dict()), d / 'selected.pt')
            write(d / 'selection.json', dict(epoch=1, valid_auc=float(auc), valid_aupr=float(ap)))
            write(d / 'EXPERIMENT.json', dict(stage='check'))
            final_ablations(model, d, dict(valid=dict(rows=rows)), evaluate, get_metrics)
            self.assertEqual(read(d / 'SELECTED_DIAGNOSTICS.json')['status'], 'passed')

    def test_uncaptured_propagation_is_not_reported_as_current_epoch(self):
        model = self.make_model()
        model.capture_propagation = False
        model.propagation_audit = dict(stale=True)
        with tempfile.TemporaryDirectory(dir='/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/tmp') as path:
            d = Path(path)
            optimizer = observed_adam_class(torch.optim.Adam, model, d, 'knowledge_no_decay')(
                model.parameters(), lr=1e-4, weight_decay=.01)
            optimizer.diagnostic_epoch = 19
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            optimizer.step()
            self.assertIsNone(read(d / 'OPTIMIZER_epoch20.json')['propagation_forward'])


if __name__ == '__main__':
    unittest.main()

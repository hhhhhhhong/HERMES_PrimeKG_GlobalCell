"""Current symmetric synergy roles and train-only inverted message dropout."""
import unittest
from unittest.mock import patch

import torch

from global_cell.attention import AlphaBeta
from global_cell.structure import compile_graph
from global_cell.synergy import SYNERGY, add_training_synergy, rows_sha256
from test_global_cell import fixture


class CurrentSynergyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        base = fixture()
        # Additional fixture families exercise the 15-family graph machinery.
        for i in range(10):
            base['groups']['fixture_relation_'+str(i)] = base['groups']['drug_target']
        base['n_cell'] = 2
        base['summary'] = dict(relation_counts={k: len(v) for k, v in base['groups'].items()})
        self.fit = torch.tensor([[0, 1, 2, 1], [1, 0, 3, 1], [0, 1, 3, 0]])
        self.base = base
        self.graph = add_training_synergy(base, self.fit, fold=0,
            expected_train_sha256=rows_sha256(self.fit))
        self.groups, _, self.trace = compile_graph(self.graph)
        self.encoder = AlphaBeta(self.graph, self.groups, dim=16, heads=4)
        self.x = torch.randn(7, 16)
        self.syn_slots = torch.tensor([i for i, m in enumerate(self.trace['slots']) if m['family'] == SYNERGY])
        self.other_slots = torch.tensor([i for i, m in enumerate(self.trace['slots']) if m['family'] != SYNERGY])

    def test_positive_training_edges_share_drug_role_and_keep_base(self):
        self.assertNotIn(SYNERGY, self.base['groups'])
        self.assertEqual(len(self.graph['groups']), 15)
        self.assertEqual(len(self.graph['groups'][SYNERGY]), 2)
        for e in self.graph['groups'][SYNERGY]:
            self.assertEqual(e['positions'], [0, 0, 2])
        self.assertEqual([e['members'] for e in self.graph['groups'][SYNERGY]], [[0, 1, 2], [0, 1, 3]])
        drug_slots = [m for m in self.trace['slots'] if m['family'] == SYNERGY and m['node'] < 2]
        self.assertTrue(drug_slots)
        self.assertTrue(all(m['position'] == 0 for m in drug_slots))

    def test_eval_is_deterministic_without_dropping_edges(self):
        self.encoder.eval()
        with torch.no_grad(), patch('torch.rand', side_effect=AssertionError('eval sampled dropout')):
            first, _ = self.encoder(self.x, self.groups)
            second, _ = self.encoder(self.x, self.groups)
        self.assertTrue(torch.equal(first, second))

    def test_train_dropout_masks_whole_edges_after_beta_with_inverted_scale(self):
        self.encoder.eval()
        eval_slots, _ = self.encoder(self.x, self.groups)
        self.encoder.train()
        # Force every edge to be retained, then every edge to be dropped.
        with patch('torch.rand', return_value=torch.ones(2, 1, 1)):
            retained, _ = self.encoder(self.x, self.groups)
        with patch('torch.rand', return_value=torch.zeros(2, 1, 1)):
            dropped, _ = self.encoder(self.x, self.groups)
        self.assertTrue(torch.allclose(retained[self.syn_slots], eval_slots[self.syn_slots]/0.8, atol=1e-6))
        self.assertEqual(float(dropped[self.syn_slots].abs().max()), 0.)
        self.assertTrue(torch.equal(dropped[self.other_slots], eval_slots[self.other_slots]))
        self.assertTrue(torch.equal(retained[self.other_slots], eval_slots[self.other_slots]))

    def test_synergy_keeps_partner_drug_and_cell_gradients(self):
        self.encoder.eval()
        x = self.x.clone().requires_grad_(True)
        slots, _ = self.encoder(x, self.groups)
        idx = next(i for i, m in enumerate(self.trace['slots'])
                   if m == dict(node=0, family=SYNERGY, position=0))
        slots[idx].square().sum().backward()
        self.assertGreater(float(x.grad[1].norm()), 0.)
        self.assertGreater(float(x.grad[2:4].norm()), 0.)
        self.assertGreater(float(self.encoder.alpha[SYNERGY].grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()

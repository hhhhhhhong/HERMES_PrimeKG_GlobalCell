"""Check split isolation, retained binary facts and actual ternary propagation."""
import copy
import unittest

import torch

from global_cell.attention import AlphaBeta, RelationInteraction
from global_cell.structure import compile_graph
from global_cell.synergy import SYNERGY, add_training_synergy, query_overlap_audit, rows_sha256
from test_global_cell import fixture


def base_fixture():
    graph = fixture()
    families = ['bioprocess_protein', 'cellcomp_protein', 'molfunc_protein',
                'drug_carrier', 'drug_enzyme', 'drug_transporter',
                'prime_disease_protein', 'prime_drug_disease_indication',
                'prime_drug_disease_contraindication', 'prime_drug_disease_off_label_use']
    # Filler families exercise compilation; their metadata is not biological data.
    for family in families:
        graph['groups'][family] = copy.deepcopy(graph['groups']['drug_target'])
    graph['n_cell'] = 2
    graph['summary'] = dict(relation_counts={k: len(v) for k, v in graph['groups'].items()})
    return graph


class SynergyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.fit = torch.tensor([[0, 1, 2, 1], [1, 0, 2, 1], [0, 1, 3, 0]])

    def build(self, graph=None, rows=None, fold=0):
        rows = self.fit if rows is None else rows
        return add_training_synergy(graph or base_fixture(), rows, fold=fold,
                                   expected_train_sha256=rows_sha256(rows))

    def test_positive_fit_only_roles_and_base_unchanged(self):
        graph = base_fixture()
        old = copy.deepcopy(graph['groups'])
        augmented = self.build(graph)
        self.assertEqual(graph['groups'], old)
        self.assertEqual(len(graph['groups']), 14)
        self.assertEqual(len(augmented['groups']), 15)
        edges = augmented['groups'][SYNERGY]
        self.assertEqual(len(edges), 2)
        self.assertEqual(augmented['synergy_audit']['duplicate_positive_observations'], 1)
        for index, edge in enumerate(edges):
            self.assertEqual(edge['members'], [0, 1, 2])
            self.assertEqual(edge['positions'], [0, 1, 2])
            self.assertEqual(edge['proof']['train_row_index'], index)
            self.assertEqual(edge['proof']['fold'], 0)
        self.assertEqual(augmented['summary']['relation_counts'][SYNERGY], 2)

    def test_each_fold_rebuilds_and_other_labels_cannot_change_edges(self):
        cache = dict(fit=self.fit.clone(), valid=torch.tensor([[0, 1, 3, 0]]),
                     test=torch.tensor([[0, 1, 3, 1]]))
        first = self.build(rows=cache['fit'])['groups'][SYNERGY]
        cache['valid'][:, 3] = 1
        cache['test'][:, 3] = 0
        self.assertEqual(first, self.build(rows=cache['fit'])['groups'][SYNERGY])
        changed = self.fit.clone()
        changed[:, 3] = torch.tensor([0, 0, 1])
        second = self.build(rows=changed, fold=1)
        self.assertEqual([e['members'] for e in second['groups'][SYNERGY]], [[0, 1, 3]])
        self.assertEqual(second['synergy_audit']['fold'], 1)
        self.assertEqual([e['members'] for e in first], [[0, 1, 2], [0, 1, 2]])

    def test_bad_split_and_cell_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            add_training_synergy(base_fixture(), self.fit, fold=0, expected_train_sha256='bad')
        bad = self.fit.clone()
        bad[0, 2] = 0
        with self.assertRaises(ValueError):
            self.build(rows=bad)
        with self.assertRaises(ValueError):
            self.build(graph=self.build())

    def test_cell_and_partner_affect_drug_via_synergy_channel(self):
        graph = self.build()
        groups, packed, trace = compile_graph(graph)
        encoder = AlphaBeta(graph, groups, dim=16, heads=4)
        x = torch.randn(7, 16, requires_grad=True)
        slots, audit = encoder(x, groups, capture=True)
        index = next(i for i, item in enumerate(trace['slots'])
                     if item == dict(node=0, family=SYNERGY, position=0))
        slots[index].square().sum().backward()
        self.assertGreater(float(x.grad[1].norm()), 0.)
        self.assertGreater(float(x.grad[2].norm()), 0.)
        self.assertGreater(float(encoder.alpha[SYNERGY].grad.norm()), 0.)
        self.assertLess(audit['normalization'][SYNERGY]['alpha_sum_error'], 1e-6)
        self.assertLess(audit['normalization'][SYNERGY]['beta_sum_error'], 1e-6)
        with torch.no_grad():
            changed = x.detach().clone()
            changed[2] += torch.linspace(-1, 1, 16)
            changed_slots, _ = encoder(changed, groups)
            self.assertGreater(float((slots[index]-changed_slots[index]).abs().max()), 1e-7)
        self.assertTrue(any(s['family'] == SYNERGY and s['position'] == 2 and s['node'] == 2
                            for s in trace['slots']))
        self.assertTrue(any(SYNERGY in p['descriptor'] and 'drug_target' in p['descriptor']
                            for p in trace['pairs']))
        self.assertTrue(any(SYNERGY in p['descriptor'] and 'cell_expression' in p['descriptor']
                            for p in trace['pairs']))
        table, _ = RelationInteraction(packed, 4, len(trace['token_keys']), dim=16)(slots)
        self.assertTrue(torch.isfinite(table).all())

    def test_overlap_audit_canonicalizes_without_accepting_labels(self):
        valid = torch.tensor([[1, 0, 2], [0, 1, 3]])
        audit = query_overlap_audit(self.fit[:, :3], valid)
        self.assertEqual(audit['canonical_train_valid_shared_queries'], 2)
        self.assertFalse(audit['labels_read'])
        with self.assertRaises(ValueError):
            query_overlap_audit(self.fit, valid)


if __name__ == '__main__':
    unittest.main()

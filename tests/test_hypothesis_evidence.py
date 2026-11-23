import copy
import unittest
import torch
from global_cell.structure import compile_graph
from global_cell.attention import AlphaBeta, RelationInteraction
from dual_branch.query_exclusion import QueryEdgeIndex, query_keys, compile_excluded
from dual_branch.hypothesis_evidence import HypothesisConditionalGamma
from test_global_cell import fixture


def graph_fixture():
    graph = fixture()
    graph['n_cell'] = 2
    graph['groups']['Synergy_Hyperedge'] = [
        dict(members=nodes, positions=[0, 0, 2], weight=1.)
        for nodes in ([0, 1, 2], [1, 0, 2], [0, 1, 3])]
    return graph


class QueryExclusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_labels_and_drug_order_do_not_change_exclusion_duplicates_all_removed(self):
        graph = graph_fixture()
        index = QueryEdgeIndex(graph)
        retained, keys, blocked = index.exclude(torch.tensor([[1, 0, 2, 0]]))
        self.assertEqual(blocked.tolist(), [True, True, False])
        self.assertEqual([e['members'] for e in retained], [[0, 1, 3]])
        changed, changed_keys, _ = index.exclude(torch.tensor([[0, 1, 2, 1]]))
        self.assertEqual(changed, retained)
        self.assertEqual(changed_keys.tolist(), keys.tolist())
        self.assertEqual(len(graph['groups']['Synergy_Hyperedge']), 3)

    def test_actual_graph_recompiled_and_routes_map_to_fixed_parameter_schema(self):
        graph = graph_fixture()
        _, _, full_trace = compile_graph(graph)
        retained, _, _ = QueryEdgeIndex(graph).exclude(torch.tensor([[0, 1, 2]]))
        groups, packed, trace = compile_excluded(graph, retained, full_trace['token_keys'])
        self.assertEqual(groups['Synergy_Hyperedge'].num_edges, 1)
        self.assertEqual(groups['Synergy_Hyperedge'].vertex.tolist(), [0, 1, 3])
        self.assertFalse(any(s['node'] == 2 and s['family'] == 'Synergy_Hyperedge' for s in trace['slots']))
        single = full_trace['token_keys'].index(('single', 'Synergy_Hyperedge', 2))
        self.assertFalse(bool(packed['mask'][2, single]))
        self.assertTrue(bool(packed['mask'][3, single]))
        for dynamic, target in zip(trace['route_proofs'], packed['token'].tolist()):
            self.assertEqual(tuple(full_trace['token_keys'][target]), tuple(dynamic['key']))

    def test_no_edges_left_has_no_fake_incidence_and_biomedical_gradient_survives(self):
        graph = graph_fixture()
        full_groups, _, full_trace = compile_graph(graph)
        retained, _, _ = QueryEdgeIndex(graph).exclude(torch.tensor([[0, 1, 2], [0, 1, 3]]))
        groups, packed, trace = compile_excluded(graph, retained, full_trace['token_keys'])
        self.assertNotIn('Synergy_Hyperedge', groups)
        self.assertFalse(any(s['family'] == 'Synergy_Hyperedge' for s in trace['slots']))
        encoder = AlphaBeta(graph, full_groups, dim=16, heads=4).eval()
        interaction = RelationInteraction(packed, 4, len(full_trace['token_keys']), dim=16)
        x = torch.randn(7, 16, requires_grad=True)
        table, _ = interaction(encoder(x, groups)[0])
        self.assertTrue(torch.isfinite(table).all())
        table.square().sum().backward()
        self.assertGreater(float(x.grad.norm()), 0.)
        self.assertGreater(float(encoder.n2e['drug_target']['0'].weight.grad.norm()), 0.)

    def test_invalid_entity_keys_rejected_and_missing_query_removes_nothing(self):
        index = QueryEdgeIndex(graph_fixture())
        retained, _, blocked = index.exclude(torch.tensor([[0, 0, 2, 1]]))
        self.assertEqual(len(retained), 3)
        self.assertEqual(int(blocked.sum()), 0)
        for row in ([[2, 1, 2]], [[0, 1, 1]], [[0.5, 1., 2.]]):
            with self.assertRaises(ValueError):
                query_keys(torch.tensor(row), 2, 2)


class HypothesisGammaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.gamma = HypothesisConditionalGamma(dim=8).eval()
        self.own = torch.randn(3, 3, 8)
        self.cf = torch.randn(3, 3, 8, requires_grad=True)
        self.table = torch.randn(3, 3, 4, 8)
        self.mask = torch.ones(3, 3, 4, dtype=torch.bool)

    def test_collaborative_features_change_attention_and_receive_reading_gradient(self):
        context, weights = self.gamma(self.own, self.table, self.mask, True, collaborative=self.cf)
        changed = self.cf.detach().clone()
        changed[:, 0] += torch.linspace(-2, 2, 8)
        _, alternative = self.gamma(self.own, self.table, self.mask, True, collaborative=changed)
        self.assertGreater(float((weights-alternative).abs().max()), 1e-8)
        (context*torch.randn_like(context)).sum().backward()
        self.assertGreater(float(self.cf.grad.norm()), 0.)
        self.assertGreater(float(self.gamma.query[0].weight.grad[:, 24:].norm()), 0.)
        off = self.gamma(self.own, self.table, self.mask, collaborative=self.cf, use_collaborative_query=False)[0]
        off_changed = self.gamma(self.own, self.table, self.mask, collaborative=changed, use_collaborative_query=False)[0]
        torch.testing.assert_close(off, off_changed, rtol=0, atol=0)

    def test_exchange_equivariance_empty_evidence_and_finite_reduced_precision(self):
        expected = self.gamma(self.own, self.table, self.mask, collaborative=self.cf)[0]
        p = [1, 0, 2]
        swapped = self.gamma(self.own[:, p], self.table[:, p], self.mask[:, p], collaborative=self.cf[:, p])[0]
        torch.testing.assert_close(swapped, expected[:, p], rtol=0, atol=2e-6)
        empty = self.gamma(self.own, self.table, torch.zeros_like(self.mask), collaborative=self.cf)[0]
        self.assertEqual(float(empty.abs().sum()), 0.)
        for dtype in (torch.float16, torch.bfloat16):
            gamma = copy.deepcopy(self.gamma).to(dtype)
            own = self.own.to(dtype).detach().requires_grad_(True)
            cf = self.cf.to(dtype).detach().requires_grad_(True)
            value = gamma(own, self.table.to(dtype), self.mask, collaborative=cf)[0]
            value.float().square().sum().backward()
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(cf.grad).all() and torch.isfinite(own.grad).all())


if __name__ == '__main__':
    unittest.main()

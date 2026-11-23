"""Selection must use real shared targets, avoid hubs, and preserve Cell data."""
import copy
import unittest
from dual_branch.knowledge_selection import select_graph


def fixture():
    def edge(members):
        return dict(members=members, positions=[0, 1], weight=.7,
            proof=dict(source='audited test fact'))
    ppi = [edge([a, b]) for a, b in [(3, 4), (3, 5), (4, 5), (3, 6),
        (3, 7), (4, 7)] + [(7, v) for v in range(8, 16)]]
    small = dict(members=[16, 3, 4, 5], positions=[0, 1, 1, 1], weight=1., proof={'source': 'members'})
    broad = dict(members=[16] + list(range(3, 16)), positions=[0] + [1] * 13, weight=1.)
    groups = dict(drug_target=[edge([0, 3]), edge([1, 4])], drug_enzyme=[edge([0, 6])],
        protein_protein=ppi, cell_expression=[dict(edge([2, v]), expression_bounded=.123) for v in range(3, 16)],
        bioprocess_protein=[small, broad], prime_disease_protein=[edge([17, 3]), edge([17, 6])],
        prime_drug_disease_indication=[edge([0, 17])],
        Synergy_Hyperedge=[dict(members=[0, 1, 2], positions=[0, 0, 2], weight=1.)])
    return dict(n_drug=2, n_cell=1, n_base=3, n_roots=3, groups=groups,
        node_types=[0, 0, 1] + [2] * 13 + [3, 4],
        type_names=['hermes_drug', 'hermes_cell', 'gene_protein', 'biological_process', 'disease'],
        routes={2: [dict(protein=6, relation='cell_expression', proof={'value': .123})]})


class KnowledgeSelectionTests(unittest.TestCase):
    def test_real_bridge_and_hub_ranking(self):
        graph = fixture()
        before = copy.deepcopy(graph)
        result, audit = select_graph(graph, bridges_per_target=1, max_function_members=3)
        self.assertEqual(graph, before)
        self.assertEqual(audit['bridge_nodes'], [5])
        self.assertEqual([e['members'] for e in result['groups']['protein_protein']], [[3, 4], [3, 5], [4, 5]])
        self.assertEqual(result['routes'], graph['routes'])

    def test_preserve_cell_DTI_synergy_weights_and_raw_proofs(self):
        graph = fixture()
        result, audit = select_graph(graph, bridges_per_target=1, max_function_members=3)
        for family in ['cell_expression', 'drug_target', 'drug_enzyme', 'Synergy_Hyperedge', 'prime_drug_disease_indication']:
            self.assertEqual(result['groups'][family], graph['groups'][family])
        self.assertEqual(len(result['groups']['bioprocess_protein']), 1)
        for e in result['groups']['bioprocess_protein']:
            self.assertTrue(set(e['members']) <= set(graph['groups']['bioprocess_protein'][0]['members']))
            self.assertEqual(e['proof'], graph['groups']['bioprocess_protein'][0]['proof'])
        self.assertFalse(audit['selection_uses_synergy_labels'])

    def test_fit_synergy_changes_cannot_change_knowledge_selection(self):
        graph = fixture()
        changed = copy.deepcopy(graph)
        changed['groups']['Synergy_Hyperedge'] = []
        a, _ = select_graph(graph, bridges_per_target=1, max_function_members=3)
        b, _ = select_graph(changed, bridges_per_target=1, max_function_members=3)
        for k in set(a['groups']) | set(b['groups']):
            if k != 'Synergy_Hyperedge':
                self.assertEqual(a['groups'].get(k), b['groups'].get(k))


if __name__ == '__main__':
    unittest.main()

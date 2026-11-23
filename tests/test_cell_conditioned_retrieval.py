import unittest
from dual_branch.cell_conditioned_retrieval import CellExpressionSupport, CellMechanismIndex


class CellConditionedRetrievalTests(unittest.TestCase):
    def fixture(self):
        facts = [dict(family='drug_target', head=d, tail=p, row=i)
                 for i, (d, p) in enumerate([(0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)])]
        types = {0: 'drug', 1: 'drug', 2: 'gene/protein', 3: 'gene/protein', 4: 'gene/protein'}
        index = CellMechanismIndex(facts, types)
        support = CellExpressionSupport({2: [-1., 1.], 3: [1., -1.]}, types)
        pool = index.candidates(0, 1, [], allowed_nodes={2, 3, 4})['paths']
        return index, support, pool

    def test_cell_changes_membership_before_attention(self):
        _, support, pool = self.fixture()
        self.assertEqual(support.select(pool, 0, k=1)[0]['entities'], [0, 3, 1])
        self.assertEqual(support.select(pool, 1, k=1)[0]['entities'], [0, 2, 1])
        self.assertEqual(support.select(pool, 0, k=1, use_cell=False), support.select(pool, 1, k=1, use_cell=False))

    def test_missing_is_unknown_and_ties_are_zero(self):
        index, support, pool = self.fixture()
        unknown = next(p for p in pool if p['entities'][1] == 4)
        self.assertEqual(support.support(unknown, 0)['relative_expression_support'], 0.)
        self.assertEqual(support.support(unknown, 0)['mapping_coverage'], 0.)
        tied = CellExpressionSupport({2: [-3., -3.]}, index.types)
        self.assertEqual(tied.ranks[2], [0., 0.])
        self.assertEqual(CellExpressionSupport({2: [1.]}, index.types).ranks[2], [0.])

    def test_expanded_pool_source_proof_and_drug_exchange(self):
        index, _, pool = self.fixture()
        self.assertEqual(len(pool), 3)  # Pool admits >2 entities for this template.
        reverse = index.candidates(1, 0, [], {2, 3, 4})['paths']
        self.assertEqual(pool, reverse)
        for path in pool:
            self.assertEqual(path['source_rows'], [index.facts[i]['row'] for i in path['fact_ids']])
            self.assertEqual(path['relations'][-1], 'drug_target^-1')
        self.assertEqual(index.candidates(0, None, [], set())['paths'], [])
        self.assertFalse(index.candidates(0, None, [], set())['pool_budget_truncated'])
        self.assertEqual(index.candidates(0, 0, [], set())['paths'], [])

    def test_unencoded_nodes_rejected_and_budget_disclosed(self):
        index, _, _ = self.fixture()
        result = index.candidates(0, 1, [], {2}, enumeration_budget=1)
        self.assertTrue(result['truncated']['shared_protein'])
        self.assertEqual(len(result['paths']), 1)
        result = index.candidates(0, 1, [], {2})
        self.assertEqual(result['excluded_unencoded'], 2)

    def test_finite_values_and_cell_order_validated(self):
        with self.assertRaises(ValueError):
            CellExpressionSupport({2: [float('nan')]}, {})
        with self.assertRaises(ValueError):
            CellExpressionSupport({2: [1.], 3: [1., 2.]}, {})
        _, support, pool = self.fixture()
        with self.assertRaises(ValueError):
            support.support(pool[0], -1)


if __name__ == '__main__':
    unittest.main()

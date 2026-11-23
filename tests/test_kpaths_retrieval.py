"""K-Paths adaptations must preserve direction, instances and raw witnesses."""
import unittest
from dual_branch.kpaths_retrieval import KPathsIndex


def fact(family, h, t, row):
    return dict(family=family, head=h, tail=t, row=row, raw_x=h, raw_y=t)


class KPathsTests(unittest.TestCase):
    def test_parallel_target_enzyme_facts_are_not_overwritten(self):
        facts = [fact('drug_target', 0, 2, 10), fact('drug_enzyme', 0, 2, 11), fact('drug_target', 1, 2, 12)]
        idx = KPathsIndex(facts, {0:'drug', 1:'drug', 2:'gene/protein'})
        paths = idx.reference(0, 1)['paths']
        self.assertEqual(len(paths), 2)
        self.assertEqual({p['relations'][0] for p in paths}, {'drug_target', 'drug_enzyme'})
        self.assertTrue(all(p['roles'][-1] == [1, 0] for p in paths))

    def test_four_hop_pathway_is_a_declared_extension(self):
        facts = [fact('drug_target', 0, 2, 0), fact('drug_target', 1, 3, 1),
            fact('pathway_protein', 4, 2, 2), fact('pathway_protein', 4, 3, 3)]
        idx = KPathsIndex(facts, {0:'drug', 1:'drug', 2:'gene/protein', 3:'gene/protein', 4:'pathway'})
        self.assertEqual(idx.reference(0, 1)['paths'], [])
        result = idx.adapted(0, 1)
        self.assertFalse(result['exact_author_reproduction'])
        self.assertEqual(result['paths'][0]['entities'], [0, 2, 4, 3, 1])
        self.assertEqual(result['paths'][0]['hops'], 4)
        self.assertEqual(result['paths'][0]['source_rows'], [0, 2, 3, 1])

    def test_two_protein_variants_survive_and_drug_exchange_is_canonical(self):
        facts = []
        for a,b in [(2,3), (4,5)]:
            facts += [fact('drug_target', 0, a, len(facts)), fact('protein_protein', a, b, len(facts)+1),
                fact('drug_target', 1, b, len(facts)+2)]
        idx = KPathsIndex(facts, {0:'drug', 1:'drug', **{i:'gene/protein' for i in range(2,6)}})
        self.assertEqual(len(idx.reference(0, 1)['paths']), 1)
        a,b = idx.adapted(0,1)['paths'], idx.adapted(1,0)['paths']
        self.assertEqual(len(a), 2)
        self.assertEqual(a, b)

    def test_third_drug_shortcuts_and_synergy_facts_are_forbidden(self):
        types = {0:'drug', 1:'drug', 2:'drug', 3:'gene/protein', 4:'gene/protein'}
        facts = [fact('drug_target',0,3,0), fact('drug_target',2,3,1),
            fact('drug_target',2,4,2), fact('drug_target',1,4,3)]
        idx = KPathsIndex(facts, types)
        self.assertEqual(idx.reference(0,1,max_hops=4)['paths'], [])
        with self.assertRaises(ValueError):
            KPathsIndex([fact('Synergy_Hyperedge',0,1,0)],types)
        with self.assertRaises(ValueError):
            KPathsIndex([fact('drug_target',0,1,0)],types)

    def test_missing_endpoint_has_only_explicit_one_sided_context(self):
        idx = KPathsIndex([fact('drug_target',0,2,0)], {0:'drug',2:'gene/protein'})
        r = idx.adapted(0,None)
        self.assertEqual(r['paths'], [])
        self.assertEqual(r['fallback'][0]['entities'], [0,2])

    def test_search_censoring_is_reported(self):
        facts = [fact('drug_target',0,p,p) for p in [2,3,4]] + [fact('drug_target',1,p,p+4) for p in [2,3,4]]
        idx = KPathsIndex(facts, {0:'drug',1:'drug',2:'gene/protein',3:'gene/protein',4:'gene/protein'})
        r=idx.reference(0,1,candidate_budget=1)
        self.assertTrue(r['censored'])
        self.assertEqual(r['entity_paths_examined'],1)


if __name__ == '__main__':
    unittest.main()

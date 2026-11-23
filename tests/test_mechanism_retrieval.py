import unittest
from dual_branch.mechanism_retrieval import MechanismIndex


class MechanismRetrievalTests(unittest.TestCase):
    def index(self):
        fs = [('drug_target', 0, 2), ('drug_target', 1, 3),
              ('protein_protein', 2, 4), ('protein_protein', 4, 3)]
        return MechanismIndex([dict(family=f, head=a, tail=b, row=i)
                               for i, (f, a, b) in enumerate(fs)],
                              {0: 'drug', 1: 'drug', 2: 'gene/protein',
                               3: 'gene/protein', 4: 'gene/protein'})

    def test_bridge_is_not_required_to_be_a_drug_target(self):
        index = self.index()
        self.assertEqual(index.adapted(0, 1)['paths'], [])
        result = index.extend(0, 1, [])
        self.assertEqual(result['paths'][0]['entities'], [0, 2, 4, 3, 1])
        self.assertEqual(result['paths'][0]['source_rows'], [0, 2, 3, 1])
        self.assertEqual(result['paths'][0]['relations'][-1], 'drug_target^-1')
        self.assertNotIn(4, index.targets[0])

    def test_canonical_drug_order_and_explicit_missing_evidence(self):
        index = self.index()
        self.assertEqual(index.extend(0, 1, [])['paths'], index.extend(1, 0, [])['paths'])
        self.assertEqual(index.extend(0, None, [])['paths'], [])
        self.assertEqual(index.extend(0, 0, [])['paths'], [])


if __name__ == '__main__':
    unittest.main()

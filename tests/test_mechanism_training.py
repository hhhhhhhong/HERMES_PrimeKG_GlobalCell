import unittest
import torch
from dual_branch.mechanism_training import CheckpointSelection
from dual_branch.mechanism_model import ThreeSourceFusion, kept_synergy_incidence


class MechanismTrainingTests(unittest.TestCase):
    def test_early_AP_peak_is_kept_independent_of_auc_and_legacy_window(self):
        selector = CheckpointSelection()
        self.assertEqual(selector.consider(1, .7, .4), ['selected', 'selected_auc'])
        self.assertEqual(selector.consider(300, .71, .6), ['selected', 'selected_auc'])
        self.assertEqual(selector.consider(1000, .8, .5), ['selected_auc', 'legacy_selected_auc'])
        self.assertEqual(selector.best_ap, .6)
        self.assertEqual(selector.consider(1500, .79, .49), [])
        with self.assertRaises(ValueError):
            selector.consider(10, float('nan'), .5)

    def test_three_sources_learn_immediately_with_shared_drug_projection(self):
        torch.manual_seed(0)
        fusion = ThreeSourceFusion(dim=8, dropout=0.).eval()
        sources = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(3)]
        result = fusion(*sources)
        (result * torch.randn_like(result)).sum().backward()
        self.assertTrue(all(float(x.grad.norm()) > 1e-6 for x in sources))
        p = [1, 0, 2]
        torch.testing.assert_close(fusion(*(x[:, p] for x in sources)), result[:, p], rtol=0, atol=0)
        for start in [0, 8, 16]:
            self.assertGreater(float(fusion.drug[0].weight.grad[:, start:start+8].norm()), 1e-6)

    def test_whole_edge_dropout_removes_incidence_and_compresses_ids(self):
        incidence = torch.tensor([[0, 1, 4, 1, 2, 5, 0, 2, 4], [0, 0, 0, 1, 1, 1, 2, 2, 2]])
        unchanged, weights = kept_synergy_incidence(incidence, 3, False, 'cpu')
        torch.testing.assert_close(unchanged, incidence, rtol=0, atol=0)
        self.assertEqual(weights.tolist(), [1., 1., 1.])
        torch.manual_seed(0)
        selected, weights = kept_synergy_incidence(incidence, 3, True, 'cpu')
        self.assertEqual(len(weights), 2)
        self.assertEqual(selected.shape[1], 6)
        self.assertEqual(sorted(set(selected[1].tolist())), [0, 1])
        self.assertEqual(torch.bincount(selected[1]).tolist(), [3, 3])


if __name__ == '__main__':
    unittest.main()

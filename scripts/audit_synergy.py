"""CPU audit of fold-specific ternary edges. Never launches a training job."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from global_cell.common import read, write, sources, sha
from global_cell.synergy import SYNERGY, add_training_synergy, query_overlap_audit


def main():
    torch.set_num_threads(2)
    all_folds = {}
    for dataset in ['ALMANAC', 'ONEIL']:
        base_path = ROOT/'data'/dataset/'graph.pt'
        before = sha(base_path)
        base = torch.load(base_path, weights_only=True, map_location='cpu')
        audits = []
        for fold in range(5):
            raw = torch.load(ROOT/'data'/dataset/f'fold{fold}/inputs.pt', weights_only=True, map_location='cpu')
            fit = raw['fit']['rows']
            augmented = add_training_synergy(base, fit, fold=fold,
                expected_train_sha256=raw['metadata']['split']['train_array_sha256'])
            for family, edges in base['groups'].items():
                assert augmented['groups'][family] is edges
            assert SYNERGY not in base['groups']
            assert len(augmented['groups']) == 15
            for edge in augmented['groups'][SYNERGY]:
                source = fit[edge['proof']['train_row_index']]
                assert source[3] == 1
                assert sorted(map(int, source[:2])) + [int(source[2])] == edge['members']
                assert edge['positions'] == [0, 1, 2] and edge['weight'] == 1.
            audit = dict(augmented['synergy_audit'], dataset=dataset,
                existing_split_overlap=query_overlap_audit(fit[:, :3], raw['valid']['rows'][:, :3]))
            audits.append(audit)
            print(dataset, fold, 'positive fit hyperedges', len(augmented['groups'][SYNERGY]), flush=True)
        assert sha(base_path) == before
        all_folds[dataset] = audits
    write(ROOT/'artifacts/synergy15/SYNERGY_FIVE_FOLD_AUDIT.json', dict(
        status='passed', datasets=all_folds, source_sha256=sources(),
        base_fact_audit=read(ROOT/'artifacts/FACT_AUDIT.json'),
        unchanged_base_graphs=True, formal_training_started=False,
        note='Training-positive labels intentionally used; validation/test labels never used for graph construction.'))


if __name__ == '__main__':
    main()

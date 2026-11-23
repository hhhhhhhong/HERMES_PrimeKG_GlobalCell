"""Read-only canonical query overlap audit; no validation labels used."""
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import write, sha


def main():
    import torch
    torch.set_num_threads(2)
    from dual_branch.query_exclusion import query_keys
    results = {}
    for dataset in ('ALMANAC', 'ONEIL'):
        folds = []
        for fold in range(5):
            path = ROOT / 'data' / dataset / ('fold%d' % fold) / 'inputs.pt'
            data = torch.load(path, map_location='cpu', weights_only=True)
            fit, valid = data['fit']['rows'], data['valid']['rows']
            n, nc = data['num_drug'], data['num_cell']
            fk, vk = query_keys(fit, n, nc), query_keys(valid, n, nc)
            pk = fk[fit[:, 3] == 1]
            duplicate_positive = len(pk) - len(Counter(pk.tolist()))
            folds.append(dict(fold=fold, fit_rows=len(fit), valid_rows=len(valid),
                positive_edge_instances=len(pk), unique_positive_query_keys=len(set(pk.tolist())),
                duplicate_positive_instances=duplicate_positive,
                valid_rows_with_positive_training_query=int(torch.isin(vk, pk).sum()),
                valid_rows_with_any_training_query=int(torch.isin(vk, fk).sum()),
                input_sha256=sha(path), validation_labels_used=False))
        results[dataset] = folds
    write(ROOT / 'artifacts/hypothesis_evidence_20261122/TARGET_OVERLAP_AUDIT.json',
        dict(datasets=results, graph_target_exclusion_does_not_remove_supervised_duplicate_overlap=True,
            existing_splits_unchanged=True, no_retraining=True, no_test_data_access=True,
            recommendation='A separately declared grouped split is needed to assess generalization without repeated Drug/Drug/Cell supervision.'))
    print(results, flush=True)


if __name__ == '__main__':
    main()

"""CPU-only observer for the user's latest in-place Synergy15 CV5 run."""
import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write
from formal_cv5 import report as observer


def report(run, frozen, checks):
    observer.RUN = run
    observer.FROZEN = frozen
    package = read(frozen/'source_manifest.json')['package']
    for ds in ['ALMANAC', 'ONEIL']:
        d = run/ds/'fold0'
        progress = observer.get(d/'progress.json')
        if progress.get('optimizer_updates', 0) < 2 or (run/f'ENTRY_{ds}.json').exists():
            continue
        audit = read(d/'SYNERGY_AUDIT.json')
        assert audit['fold'] == 0 and audit['relation_types_after'] == 15
        assert audit['positive_training_rows'] == audit['synergy_hyperedges'] > 0
        assert not audit['validation_labels_used'] and not audit['test_labels_used']
        for dataset in ['ALMANAC', 'ONEIL']:
            check = read(checks/dataset/'CHECKS.json')
            assert check['status'] == 'passed' and check['source_sha256'] == package
            changes = {k: v for k, v in check['actual_parameter_changes'].items()
                       if 'Synergy_Hyperedge' in k}
            assert changes and all(math.isfinite(v) and v > 0 for v in changes.values())
        assert read(d/'EXPERIMENT.json')['recipe']['relation_type_count'] == 15
    done = observer.report()
    state = read(run/'STATUS.json')
    state.update(variant='Synergy15 + user edits 2026-11-17 19:38',
                 source_manifest=str(frozen/'source_manifest.json'),
                 gpu_checks={ds: observer.get(checks/ds/'progress.json') for ds in ['ALMANAC', 'ONEIL']})
    write(run/'STATUS.json', state)
    p = run/'RESULTS.md'
    p.write_text(p.read_text().replace('# GlobalCell：seed0 五折状态', '# GlobalCell Synergy15：用户最新修改版 seed0 五折状态', 1))
    return done


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--checks', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while True:
        done = report(args.run, args.frozen, args.checks)
        if done or not args.watch:
            break
        time.sleep(15)

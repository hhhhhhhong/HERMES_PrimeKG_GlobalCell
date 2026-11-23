"""Observe two ALMANAC fold0 repair pilots; never launch GPU processes."""
import argparse
import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha
DISPATCH = ROOT / 'artifacts/transformer_repair_20261120'
RUN = ROOT / 'runs/transformer_repair_fold0_seed0_20261120'


def get(path):
    return read(path) if path.exists() else {}


def report():
    modes = ['full', 'selected']
    cpu_marker = DISPATCH / 'CPU_READY.json'
    if not cpu_marker.exists():
        checks = {m: get(DISPATCH / ('cpu_' + m + '_retry1') / 'CHECKS.json') for m in modes}
        if all(c.get('status') == 'passed' for c in checks.values()):
            exp = {m: read(DISPATCH / ('cpu_' + m + '_retry1') / 'EXPERIMENT.json') for m in modes}
            assert exp['full']['initialization_sha256'] == exp['selected']['initialization_sha256']
            assert exp['full']['parameters'] == exp['selected']['parameters']
            assert exp['full']['inputs'] == exp['selected']['inputs']
            manifest = read(DISPATCH / 'frozen/manifest.json')
            assert all(sha(p) == h for p, h in manifest['source'].items())
            write(cpu_marker, dict(status='complete', paired_CPU_checks=checks,
                parameter_count=exp['full']['parameters'], same_parameter_initialization=True,
                same_inputs_and_Cell=True, training_seed=0, fold=0, no_performance_claim=True))
    gpu = {m: get(DISPATCH / ('gpu_' + m) / 'CHECKS.json') for m in modes}
    if not (DISPATCH / 'GPU_READY.json').exists() and all(c.get('status') == 'passed' for c in gpu.values()):
        assert cpu_marker.exists()
        reference = read(DISPATCH / 'cpu_full_retry1/EXPERIMENT.json')
        for mode, check in gpu.items():
            assert check['training_rows'] == 1024 and check['optimizer_updates'] == 3
            assert check['residual_caps_verified'] and check['strict_fresh_instance_reload_error'] < 2e-6
            exp = read(DISPATCH / ('gpu_' + mode) / 'EXPERIMENT.json')
            assert exp['initialization_sha256'] == reference['initialization_sha256']
        write(DISPATCH / 'GPU_READY.json', dict(status='complete', checks=gpu,
            same_CPU_GPU_parameter_initialization=True, no_performance_claim=True))
    jobs = []
    for mode in modes:
        p = RUN / 'ALMANAC' / mode / 'fold0'
        progress = get(p / 'progress.json') or dict(status='queued', epoch=0)
        selection = get(p / 'selection.json')
        row = dict(knowledge_mode=mode, progress=progress, selection=selection,
            diagnostic_best_aupr=get(p / 'best_aupr.json'))
        if progress['status'] == 'complete':
            assert progress['epoch'] == progress['optimizer_updates'] == 3000
            assert sha(p / 'selected.pt') == selection['checkpoint_sha256']
            write(p / 'FINAL_VERIFICATION.json', dict(status='passed', updates=3000,
                selection=selection, internal_validation_only=True, test_evaluated=False))
        jobs.append(row)
    write(RUN / 'STATUS.json', dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='ALMANAC seed0/fold0, paired stability/full and stability/selected knowledge pilots',
        CPU_ready=cpu_marker.exists(), GPU_checks=gpu, jobs=jobs, test_evaluated=False))
    return all(j['progress']['status'] in ('complete', 'failed') for j in jobs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while True:
        if report() or not args.watch:
            break
        time.sleep(5)

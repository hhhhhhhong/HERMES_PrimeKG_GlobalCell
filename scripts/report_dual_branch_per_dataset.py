"""Reuse the CV5 observer, admitting each dataset after its own GPU check.

This changes resource dependencies only: the completed 1024-query check is
the evidence for its dataset's memory budget. It never launches GPU workers.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment, RUNTIME
from global_cell.common import read, write, sha

environment()
DISPATCH = ROOT / 'artifacts/dual_branch_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_cv5_seed0_20261119'
spec = importlib.util.spec_from_file_location('dual_branch_original_observer', ROOT / 'scripts/report_dual_branch_cv5.py')
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
original_verify_entry = BASE.verify_entry


def calibrate_per_dataset(checks):
    for dataset, progress in checks.items():
        marker = DISPATCH / ('MEMORY_READY_' + dataset + '.json')
        if marker.exists() or progress.get('status') != 'complete':
            continue
        result = read(DISPATCH / ('gpu_check_' + dataset) / 'CHECKS.json')
        assert result['status'] == 'passed'
        assert result['training_rows'] == result['requested_check_rows'] == 1024
        # The balanced validation subset may contain fewer positives (ONEIL
        # has323 positive validation rows). The 1024-row backward determines
        # query memory; do not duplicate validation rows to reach a count.
        assert 0 < result['validation_rows'] <= 1024
        assert result['optimizer_updates'] == 3
        assert result['all_components_supervised_gradient_nonzero_after_step3']
        allocated = result['peak_cuda_memory_mib']
        observed = BASE.get(DISPATCH / ('PEAK_PROCESS_' + dataset + '.json')).get('peak_process_mib', 0)
        # The correctness run includes three optimizer updates plus a second
        # live model for strict reload. Keep headroom over the observed peak.
        reserve = max(7168 if dataset == 'ALMANAC' else 5120,
            math.ceil(max(1.15 * allocated + 1024, observed + 64) / 256) * 256)
        registry_path = RUNTIME / 'auxiliary_jobs.json'
        registry = read(registry_path)
        changed = []
        for job in registry['jobs']:
            if job['module'] != 'train_hermes_dual_branch' or job['dataset'] != dataset or \
                    job['extra_args'][job['extra_args'].index('--stage') + 1] != 'train':
                continue
            fold = int(job['extra_args'][job['extra_args'].index('--fold') + 1])
            assert job['directory'].startswith(str(RUN) + '/')
            job['memory_mib'] = reserve
            job['requires'] = [str(DISPATCH / ('gpu_check_' + dataset) / 'progress.json'), str(marker)]
            if fold:
                job['requires'].append(str(RUN / ('ENTRY_' + dataset + '.json')))
            changed.append(job['key'])
        assert len(changed) == 5
        write(registry_path, registry)
        record = dict(status='complete', dataset=dataset, allocated_peak_mib=allocated,
            reserved_peak_mib=result['peak_cuda_reserved_mib'], sampled_process_peak_mib=observed,
            reserve_mib=reserve, changed_job_keys=changed,
            formula='ceil256(max(1.15*allocated+1024, sampled_process_peak+64)); AL min7168, ON min5120',
            training_check_rows=result['training_rows'], validation_check_rows=result['validation_rows'],
            check_includes_two_live_models_for_reload=True,
            own_dataset_GPU_check_required=True, other_dataset_not_a_dependency=True,
            model_recipe_seed_split_and_process_limit_unchanged=True,
            observer_source=str(Path(__file__).resolve()), observer_source_sha256=sha(__file__))
        write(DISPATCH / ('MEMORY_BUDGET_' + dataset + '.json'), record)
        write(marker, record)


def verify_own_dataset(dataset, directory, checks, audit, manifest):
    return original_verify_entry(dataset, directory, {dataset: checks[dataset]}, audit, manifest)


BASE.calibrate = calibrate_per_dataset
BASE.verify_entry = verify_own_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if BASE.report() or not args.watch:
            break
        time.sleep(2)

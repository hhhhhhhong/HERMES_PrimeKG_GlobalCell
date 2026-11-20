"""Gemini rerun CPU observer: per-dataset admission, no GPU subprocesses."""
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
DISPATCH = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119'
spec = importlib.util.spec_from_file_location('gemini_shared_report', ROOT / 'scripts/report_dual_branch_cv5.py')
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
BASE.DISPATCH, BASE.RUN = DISPATCH, RUN


def calibrate(checks):
    for dataset, progress in checks.items():
        marker = DISPATCH / ('MEMORY_READY_' + dataset + '.json')
        if marker.exists() or progress.get('status') != 'complete':
            continue
        result = read(DISPATCH / ('gpu_check_' + dataset) / 'CHECKS.json')
        assert result['status'] == 'passed' and result['training_rows'] == result['requested_check_rows'] == 1024
        assert 0 < result['validation_rows'] <= 1024
        assert result['all_components_supervised_gradient_nonzero_after_step3']
        peak = result['peak_cuda_memory_mib']
        observed = BASE.get(DISPATCH / ('PEAK_PROCESS_' + dataset + '.json')).get('peak_process_mib', 0)
        reserve = max(7168 if dataset == 'ALMANAC' else 5120,
            math.ceil(max(1.15 * peak + 1024, observed + 64) / 256) * 256)
        path = RUNTIME / 'auxiliary_jobs.json'
        registry = read(path)
        changed = []
        for job in registry['jobs']:
            if not job['directory'].startswith(str(RUN) + '/'):
                continue
            assert job['module'] == 'train_hermes_dual_branch'
            if job['dataset'] != dataset:
                continue
            job['memory_mib'] = reserve
            changed.append(job['key'])
        assert len(changed) == 5
        write(path, registry)
        record = dict(status='complete', dataset=dataset, reserve_mib=reserve,
            peak_allocated_mib=peak, peak_reserved_mib=result['peak_cuda_reserved_mib'],
            sampled_process_peak_mib=observed, training_rows=result['training_rows'],
            validation_rows=result['validation_rows'], changed_job_keys=changed,
            formula='ceil256(max(1.15*allocated+1024, sampled_peak+64)); AL min7168/ON min5120',
            check_includes_second_live_reload_model=True, model_and_recipe_unchanged=True)
        write(DISPATCH / ('MEMORY_BUDGET_' + dataset + '.json'), record)
        write(marker, record)


def verify_entry(dataset, directory, checks, audit, manifest):
    marker = RUN / ('ENTRY_' + dataset + '.json')
    progress = BASE.get(directory / 'progress.json')
    if marker.exists() or progress.get('optimizer_updates', 0) < 3:
        return
    assert checks[dataset].get('status') == 'complete'
    experiment = read(directory / 'EXPERIMENT.json')
    recipe = experiment['recipe']
    assert recipe['version'] == 'dual_branch_gemini_v2_20261119'
    assert recipe['graph_propagation_gates_initial'] == .1
    assert recipe['all_downstream_trainable'] and not recipe['old_checkpoint_loaded']
    assert experiment['inputs']['original_split'] == audit['folds'][dataset + '/fold0']['split']
    assert experiment['source_manifest'] == str(manifest)
    g1, g2, g3 = [read(directory / ('GRADIENT_epoch%d.json' % i)) for i in [1, 2, 3]]
    blocked = ['graph_type_projection', 'alpha_beta', 'propagation_gates', 'relation_interaction', 'gamma', 'fusion_router']
    assert all(g1[key] == 0 for key in blocked)
    assert all(value > 0 for key, value in g1.items() if key not in blocked)
    assert all(math.isfinite(value) and value > 0 for value in list(g2.values()) + list(g3.values()))
    groups = read(directory / 'OPTIMIZER_GROUPS.json')['groups']
    assert [(g['lr'], g['weight_decay']) for g in groups] == [(1e-3, 1e-4), (2e-3, 1e-4), (2e-3, 0.)]
    write(marker, dict(status='complete', dataset=dataset, full_fit_updates=3,
        first_step_outer_projection_then_step2_all_components=True,
        actual_grouped_optimizer_verified=True, no_old_checkpoint_loaded=True))


BASE.calibrate, BASE.verify_entry = calibrate, verify_entry


def report():
    finished = BASE.report()
    status = read(RUN / 'STATUS.json')
    status['variant'] = 'corrected Gemini dual branch; inner gates0.1 / RMS scaling / grouped Adam'
    write(RUN / 'STATUS.json', status)
    version = read(ROOT / 'DUAL_BRANCH_VERSION.json')
    cpu = {dataset: BASE.get(DISPATCH / ('cpu_check_' + dataset) / 'progress.json') for dataset in ['ALMANAC', 'ONEIL']}
    gpu = status['gpu_checks']
    version['CPU_checks'] = cpu
    version['CPU_check_results'] = {dataset: BASE.get(DISPATCH / ('cpu_check_' + dataset) / 'CHECKS.json') for dataset in cpu}
    version['status'] = 'training' if version['formal_training']['running_fits'] else \
        'GPU_verified_partial' if any(value.get('status') == 'complete' for value in gpu.values()) else \
        'CPU_verified' if all(value.get('status') == 'complete' for value in cpu.values()) else 'checking'
    version.update(version='dual_branch_gemini_v2_20261119', code_directory=str(ROOT / 'dual_branch'),
        documentation=str(ROOT / 'DUAL_BRANCH_GEMINI_V2.md'), old_variant_replaced=True)
    write(ROOT / 'DUAL_BRANCH_VERSION.json', version)
    return finished


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if report() or not args.watch:
            break
        time.sleep(2)

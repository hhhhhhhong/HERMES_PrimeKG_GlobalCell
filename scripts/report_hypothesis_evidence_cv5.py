"""CPU-only status, memory admission and checkpoint observer; never starts GPUs."""
import argparse
import datetime
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import RUNTIME
from dual_branch.hypothesis_common import DISPATCH, RUN, VERSION
from global_cell.common import read, write, sha


def get(path):
    return read(path) if path.exists() else {}


def report():
    # Sample only this suite's check workers. Max allocated memory and an
    # explicit headroom floor remain available if sampling misses a peak.
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if query.returncode:
        raise RuntimeError('NVML query failed; no memory admission changes made')
    processes = {}
    for line in query.stdout.splitlines():
        pieces = [p.strip() for p in line.split(',')]
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1].isdigit():
            processes[int(pieces[0])] = int(pieces[1])
    samples = get(DISPATCH / 'PROCESS_MEMORY_PEAKS.json')
    for ds in ['ALMANAC', 'ONEIL']:
        progress = get(DISPATCH / ('gpu_check_' + ds) / 'progress.json')
        if progress.get('pid') in processes:
            samples[ds] = max(samples.get(ds, 0), processes[progress['pid']])
    write(DISPATCH / 'PROCESS_MEMORY_PEAKS.json', samples)
    summaries, rows = {}, []
    for ds in ['ALMANAC', 'ONEIL']:
        check = get(DISPATCH / ('gpu_check_' + ds) / 'CHECKS.json')
        marker = DISPATCH / ('GPU_READY_' + ds + '.json')
        if check.get('status') == 'passed' and not marker.exists():
            cpu = read(DISPATCH / ('cpu_' + ds) / 'EXPERIMENT.json')
            gpu = read(DISPATCH / ('gpu_check_' + ds) / 'EXPERIMENT.json')
            assert cpu['initialization_sha256'] == gpu['initialization_sha256']
            assert check['supervised_rows'] == 2048 and check['graph_tables_per_check_epoch'] == 2 and check['optimizer_updates'] == 3
            assert check['fusion_gate']['status'] == 'passed'
            peak = max(check['peak_cuda_allocated_mib'] + 1536, samples.get(ds, 0))
            budget = max(4096, math.ceil((1.15 * peak + 512) / 512) * 512)
            registry_file = RUNTIME / 'auxiliary_jobs.json'
            import fcntl
            edit_lock = (RUNTIME / 'auxiliary_registry_edit.lock').open('a')
            fcntl.flock(edit_lock, fcntl.LOCK_EX)
            registry = read(registry_file)
            changed = []
            for job in registry['jobs']:
                if job.get('module') == 'train_hermes_hypothesis_evidence_cv5' and job.get('dataset') == ds and '--stage' in job.get('extra_args', []):
                    args = job['extra_args']
                    if args[args.index('--stage') + 1] == 'train' and not (Path(job['directory']) / 'progress.json').exists():
                        changed.append(dict(key=job['key'], previous=job['memory_mib'], new=budget))
                        job['memory_mib'] = budget
            write(DISPATCH / ('MEMORY_ADMISSION_' + ds + '.json'), dict(changes=changed,
                allocated_peak_mib=check['peak_cuda_allocated_mib'], sampled_process_peak_mib=samples.get(ds),
                formula='ceil512(1.15*max(allocated+1536,sampled process)+512), minimum4096',
                per_GPU_all_user_process_cap=2, model_and_chunk_unchanged=True))
            write(registry_file, registry)
            fcntl.flock(edit_lock, fcntl.LOCK_UN)
            edit_lock.close()
            write(marker, dict(status='complete', GPU_check=check, CPU_GPU_initialization_identical=True,
                measured_fit_memory_budget_mib=budget, no_performance_claim=True))
        dataset_rows = []
        for fold in range(5):
            directory = RUN / ds / ('fold%d' % fold)
            progress = get(directory / 'progress.json') or dict(status='queued', epoch=0, optimizer_updates=0)
            selected = get(directory / 'selected.json')
            if fold == 0 and progress.get('optimizer_updates', 0) >= 3 and not (RUN / ('ENTRY_' + ds + '.json')).exists():
                experiment = read(directory / 'EXPERIMENT.json')
                assert experiment['initialization_sha256'] == read(DISPATCH / ('cpu_' + ds) / 'EXPERIMENT.json')['initialization_sha256']
                for epoch in [1, 2, 3]:
                    gradients = read(directory / ('GRADIENT_epoch%d.json' % epoch))['component_gradient_L2']
                    assert all(value > 0 for value in gradients.values())
                    assert gradients['fusion_gate'] > 0
                    assert read(directory / ('GRADIENT_epoch%d.json' % epoch))['collaborative_query_weight_gradient_L2'] > 0
                write(RUN / ('ENTRY_' + ds + '.json'), dict(status='complete', full_fit_updates=3,
                    all_component_and_gate_gradients_nonzero=True, initialized_from_scratch=True))
            if progress.get('status') == 'complete' and not (directory / 'FINAL_VERIFICATION.json').exists():
                assert progress['optimizer_updates'] == 3000
                history = [read_row for read_row in (directory / 'history.jsonl').read_text().splitlines()]
                assert len(history) == 3000
                selections = {name: read(directory / (name + '.json'))
                              for name in ['selected', 'selected_auc', 'legacy_selected_auc']}
                assert all(sha(directory / (name + '.pt')) == value['checkpoint_sha256'] for name, value in selections.items())
                write(directory / 'FINAL_VERIFICATION.json', dict(status='passed', updates=3000,
                    selections=selections, internal_validation_only=True, test_evaluated=False))
            dataset_rows.append(dict(dataset=ds, fold=fold, progress=progress, selection=selected,
                selected_auc=get(directory / 'selected_auc.json'), legacy_selected_auc=get(directory / 'legacy_selected_auc.json')))
        if all(r['progress']['status'] == 'complete' for r in dataset_rows):
            summaries[ds] = {metric: dict(mean=statistics.mean(r['selection'][metric] for r in dataset_rows),
                standard_deviation=statistics.pstdev(r['selection'][metric] for r in dataset_rows))
                for metric in ['valid_auc', 'valid_aupr']}
        rows += dataset_rows
    status = dict(version=VERSION, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        datasets=['ALMANAC', 'ONEIL'], seed=0, folds_per_dataset=5, epochs=3000, jobs=rows,
        completed=sum(r['progress']['status'] == 'complete' for r in rows),
        running=sum(r['progress']['status'] == 'running' for r in rows),
        queued=sum(r['progress']['status'] == 'queued' for r in rows),
        failed=sum(r['progress']['status'] == 'failed' for r in rows),
        complete_fivefold_internal_validation=summaries, primary_selection='Val AUPRC from first evaluation',
        original_HERMES_comparison_requires_same_selection_policy=True, test_evaluated=False, original_splits_have_duplicate_supervision=True,
        overlap_audit=str(ROOT / 'artifacts/hypothesis_evidence_20261122/TARGET_OVERLAP_AUDIT.json'))
    write(RUN / 'STATUS.json', status)
    lines = ['# 目标边屏蔽与协同引导证据模型：seed0 双数据集五折', '',
        '主选点：首次评估起内部验证 AUPRC。独立保留全程/1000轮以后 AUROC；不是测试成绩。', '',
        '| Dataset | Fold | Status | Epoch | Selected AUROC | Selected AUPRC |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        s = r['selection']
        lines.append('| %s | %d | %s | %d | %s | %s |' % (r['dataset'], r['fold'], r['progress']['status'],
            r['progress'].get('epoch', 0), '%.6f' % s['valid_auc'] if s else '', '%.6f' % s['valid_aupr'] if s else ''))
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return all(r['progress']['status'] in ['complete', 'failed'] for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if report() or not args.watch:
            break
        time.sleep(5)

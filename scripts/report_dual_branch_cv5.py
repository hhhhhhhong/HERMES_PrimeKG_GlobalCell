"""CPU-only observer: measured placement, first-fold admission and honest status."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

DISPATCH = ROOT / 'artifacts/dual_branch_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_cv5_seed0_20261119'
RUNTIME = ROOT.parent / 'runtime/paper_routes_20261009'


def get(path):
    return read(path) if path.exists() else {}


def sample_memory():
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
        memory = {int(pid): int(value) for pid, value in csv.reader(rows.splitlines())}
    except (ValueError, subprocess.CalledProcessError):
        return
    for dataset in ['ALMANAC', 'ONEIL']:
        progress = get(DISPATCH / ('gpu_check_' + dataset) / 'progress.json')
        pid = progress.get('pid')
        if pid in memory:
            path = DISPATCH / ('PEAK_PROCESS_' + dataset + '.json')
            previous = get(path).get('peak_process_mib', 0)
            if memory[pid] > previous:
                write(path, dict(pid=pid, peak_process_mib=memory[pid]))


def calibrate(checks):
    marker = DISPATCH / 'MEMORY_READY.json'
    if marker.exists() or not all(value.get('status') == 'complete' for value in checks.values()):
        return
    budgets = {}
    for dataset in checks:
        result = read(DISPATCH / ('gpu_check_' + dataset) / 'CHECKS.json')
        assert result['status'] == 'passed' and result['training_rows'] == result['validation_rows'] == 1024
        assert result['all_components_supervised_gradient_nonzero_after_step3']
        allocated = result['peak_cuda_memory_mib']
        observed = get(DISPATCH / ('PEAK_PROCESS_' + dataset + '.json')).get('peak_process_mib', 0)
        reserve = max(7680 if dataset == 'ALMANAC' else 6144,
            math.ceil(max(1.15 * allocated + 1536, observed + 1024) / 512) * 512)
        budgets[dataset] = dict(allocated_peak_mib=allocated,
            sampled_process_peak_mib=observed, reserve_mib=reserve)
    path = RUNTIME / 'auxiliary_jobs.json'
    registry = read(path)
    changed = []
    for job in registry['jobs']:
        if job['module'] == 'train_hermes_dual_branch' and '--stage' in job['extra_args'] and \
                job['extra_args'][job['extra_args'].index('--stage') + 1] == 'train':
            assert job['directory'].startswith(str(RUN) + '/')
            job['memory_mib'] = budgets[job['dataset']]['reserve_mib']
            changed.append(job['key'])
    assert len(changed) == 10
    write(path, registry)
    write(DISPATCH / 'MEMORY_BUDGET.json', dict(status='complete', datasets=budgets,
        changed_job_keys=changed, model_and_training_recipe_unchanged=True,
        formula='ceil512(max(1.15*allocated+1536, sampled_process_peak+1024)); AL min7680, ON min6144'))
    write(marker, dict(status='complete', datasets=budgets))


def verify_entry(dataset, directory, checks, audit, manifest):
    marker = RUN / ('ENTRY_' + dataset + '.json')
    progress = get(directory / 'progress.json')
    if marker.exists() or progress.get('optimizer_updates', 0) < 3:
        return
    assert all(value.get('status') == 'complete' for value in checks.values())
    experiment = read(directory / 'EXPERIMENT.json')
    assert experiment['recipe']['all_downstream_trainable']
    assert not experiment['recipe']['old_checkpoint_loaded']
    assert experiment['inputs']['original_split'] == audit['folds'][dataset + '/fold0']['split']
    assert experiment['source_manifest'] == str(manifest)
    g1, g2, g3 = [read(directory / ('GRADIENT_epoch%d.json' % i)) for i in [1, 2, 3]]
    blocked = ['graph_type_projection', 'alpha_beta', 'propagation_gates', 'relation_interaction', 'gamma', 'fusion_router']
    assert all(g1[key] == 0 for key in blocked)
    assert all(g1[key] > 0 for key in set(g1) - set(blocked))
    assert g2['alpha_beta'] == 0 and all(value > 0 for key, value in g2.items() if key != 'alpha_beta')
    assert all(math.isfinite(value) and value > 0 for value in g3.values())
    write(marker, dict(status='complete', dataset=dataset, full_fit_updates=3,
        staged_supervised_gradients_verified=True, all_downstream_trainable=True,
        no_old_checkpoint_loaded=True, original_split_verified=True))


def report():
    sample_memory()
    manifest = DISPATCH / 'frozen/manifest.json'
    audit = read(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json')
    checks = {dataset: get(DISPATCH / ('gpu_check_' + dataset) / 'progress.json')
        for dataset in ['ALMANAC', 'ONEIL']}
    calibrate(checks)
    records, summaries = [], {}
    for dataset in checks:
        for fold in range(5):
            directory = RUN / dataset / ('fold%d' % fold)
            progress = get(directory / 'progress.json') or dict(status='queued', epoch=0)
            if fold == 0:
                verify_entry(dataset, directory, checks, audit, manifest)
            selection = get(directory / 'selection.json')
            row = dict(dataset=dataset, fold=fold, progress=progress, selection=selection)
            if progress['status'] == 'complete':
                lock = directory / 'FINAL_VERIFICATION.json'
                if not lock.exists():
                    assert progress['epoch'] == progress['optimizer_updates'] == 3000
                    history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
                    assert [value['epoch'] for value in history] == list(range(1, 3001))
                    assert sha(directory / 'selected.pt') == selection['checkpoint_sha256']
                    experiment = read(directory / 'EXPERIMENT.json')
                    assert experiment['source_manifest'] == str(manifest)
                    assert experiment['inputs']['original_split'] == audit['folds']['%s/fold%d' % (dataset, fold)]['split']
                    assert experiment['recipe']['all_downstream_trainable'] and not experiment['recipe']['old_checkpoint_loaded']
                    write(lock, dict(status='passed', updates=3000, selection=selection,
                        source_manifest=str(manifest), test_evaluated=False))
                row['verified_complete'] = True
            records.append(row)
        completed = [row for row in records if row['dataset'] == dataset and row.get('verified_complete')]
        if len(completed) == 5:
            summaries[dataset] = dict(folds=5,
                internal_AUROC_mean=statistics.mean(row['selection']['valid_auc'] for row in completed),
                internal_AUROC_std=statistics.pstdev(row['selection']['valid_auc'] for row in completed),
                internal_AUPRC_mean=statistics.mean(row['selection']['valid_aupr'] for row in completed),
                internal_AUPRC_std=statistics.pstdev(row['selection']['valid_aupr'] for row in completed),
                internal_validation_only=True, test_evaluated=False)
    write(RUN / 'STATUS.json', dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variant='all-node CODER / molecular-expression dual branch', seed=0, folds_per_dataset=5,
        jobs=records, gpu_checks=checks, full_fivefold_summaries=summaries, test_evaluated=False))
    lines = ['# 全节点CODER＋2D/3D与Cell表达：seed0五折', '',
        '| 数据集 | fold | 状态 | 轮次 | 选定内部AUROC | 对应AUPRC |', '|---|---:|---|---:|---:|---:|']
    for row in records:
        p, s = row['progress'], row['selection']
        lines.append('| %s | %d | %s | %d/3000 | %s | %s |' % (row['dataset'], row['fold'],
            p['status'], p.get('epoch', 0), s.get('valid_auc', '—'), s.get('valid_aupr', '—')))
    lines += ['', 'GPU完整图1024查询检查通过后，按实测显存放行fold0；每数据集fold0完成3次整折更新并通过梯度核验，再放行其余折。',
        '全部下游从头可训练，不载入旧协同checkpoint；固定CODER/Uni-Mol输入。',
        '3000完整轮，epoch1000–3000按内部验证AUROC选轮；不读取封存测试，不加seed或网格。',
        '每卡所有用户目录任务合计最多2GPU进程；旧模型/运行保留。']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    version = read(ROOT / 'DUAL_BRANCH_VERSION.json')
    version['formal_training'] = dict(run_directory=str(RUN), status_file=str(RUN / 'STATUS.json'),
        GPU_checks=checks, fits_submitted=10, running_fits=sum(row['progress']['status'] == 'running' for row in records),
        completed_fits=sum(row['progress']['status'] == 'complete' for row in records))
    version['formal_training_fits_submitted'] = 10
    version['GPU_check_performed'] = any(value.get('epoch', 0) > 0 for value in checks.values())
    write(ROOT / 'DUAL_BRANCH_VERSION.json', version)
    return all(row['progress']['status'] in ['complete', 'failed', 'cancelled'] for row in records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if report() or not args.watch:
            break
        time.sleep(5)

"""CPU-only progress and fixed-recipe verification for ONEIL no-decay CV5."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

RUN = ROOT / 'runs/knowledge_decay_ONEIL_cv5_seed0_20261119'
DISPATCH = ROOT / 'artifacts/knowledge_decay_ONEIL_cv5_seed0_20261119'
REFERENCE = ROOT / 'runs/zero_gated_hgnn3_cv5_seed0_20261118/ONEIL'


def get(path):
    return read(path) if path.exists() else {}


def report():
    manifest = read(DISPATCH / 'frozen/source_manifest.json')
    audit = read(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json')
    gpu_check = get(DISPATCH / 'gpu_check/progress.json')
    records = []
    for fold in range(5):
        directory = RUN / 'ONEIL' / ('fold%d' % fold)
        progress = get(directory / 'progress.json') or dict(status='queued', epoch=0)
        selection = get(directory / 'selection.json')
        diagnostics = get(directory / 'SELECTED_DIAGNOSTICS.json')
        reference = REFERENCE / ('fold%d' % fold)
        reference_progress = get(reference / 'progress.json')
        row = dict(fold=fold, progress=progress, selection=selection,
                   diagnostics=diagnostics, reference_progress=reference_progress,
                   reference_selection=get(reference / 'selection.json'),
                   original_HERMES=audit['folds']['ONEIL/fold%d' % fold]['baseline'])
        records.append(row)

        # This CPU observer opens the remaining folds only after two actual
        # full-fit optimizer updates with the expected zero-gate gradient order.
        marker = RUN / 'ENTRY_ONEIL.json'
        if fold == 0 and progress.get('optimizer_updates', 0) >= 2 and not marker.exists():
            assert gpu_check.get('status') == 'complete'
            experiment = read(directory / 'EXPERIMENT.json')
            assert experiment['recipe']['knowledge_weight_decay'] == 0.
            assert experiment['initialization_sha256'] == read(reference / 'EXPERIMENT.json')['initialization_sha256']
            assert experiment['baseline_metadata']['split'] == audit['folds']['ONEIL/fold0']['split']
            first, second = [read(directory / ('GRADIENT_epoch%d.json' % e)) for e in (1, 2)]
            blocked = ['alpha', 'beta', 'node_to_edge', 'edge_to_node', 'expression_edge', 'extra_alpha_beta']
            assert all(first[k] == 0. for k in blocked) and first['propagation_gates'] > 0
            assert all(math.isfinite(v) and v > 0 for v in second.values())
            write(marker, dict(status='complete', full_fit_updates=2,
                first_step_gate_then_branch_gradients_verified=True,
                observer_does_not_dispatch_GPU_workers=True))

        if progress.get('status') == 'complete':
            lock = directory / 'FINAL_VERIFICATION.json'
            if not lock.exists():
                assert progress['epoch'] == progress['optimizer_updates'] == 3000
                epochs = [json.loads(line)['epoch'] for line in (directory / 'history.jsonl').read_text().splitlines()]
                assert epochs == list(range(1, 3001))
                assert sha(directory / 'selected.pt') == selection['checkpoint_sha256']
                assert diagnostics['status'] == 'passed' and diagnostics['epoch'] == selection['epoch']
                experiment = read(directory / 'EXPERIMENT.json')
                assert experiment['source_sha256'] == manifest['package']
                assert all(manifest['extra'][p] == h for p, h in experiment['variant_sources'].items())
                assert experiment['recipe']['all_downstream_trainable']
                assert not experiment['recipe']['original_checkpoint_loaded']
                assert experiment['recipe']['knowledge_weight_decay'] == 0.
                assert experiment['recipe']['weight_decay'] == .01
                assert experiment['initialization_sha256'] == read(reference / 'EXPERIMENT.json')['initialization_sha256']
                assert experiment['baseline_metadata']['split'] == audit['folds']['ONEIL/fold%d' % fold]['split']
                write(lock, dict(status='passed', seed=0, fold=fold, epochs=3000,
                    selection=selection, initialization_and_split_match_reference=True,
                    original_checkpoint_loaded=False, test_evaluated=False))
            row['verified_complete'] = True

    summary = {}
    if all(row.get('verified_complete') for row in records):
        aucs = [row['selection']['valid_auc'] for row in records]
        aps = [row['selection']['valid_aupr'] for row in records]
        summary = dict(folds=5, valid_auc_mean=statistics.mean(aucs), valid_auc_std=statistics.pstdev(aucs),
            valid_aupr_mean=statistics.mean(aps), valid_aupr_std=statistics.pstdev(aps),
            internal_validation_only=True, test_evaluated=False)
        # Compare complete matched references only; never mix partial fold4
        # values with four finished folds or select a recipe on these scores.
        if all(row['reference_progress'].get('status') == 'complete' for row in records):
            summary.update(
                all_parameter_decay_auc_mean=statistics.mean(row['reference_selection']['valid_auc'] for row in records),
                all_parameter_decay_aupr_mean=statistics.mean(row['reference_selection']['valid_aupr'] for row in records))
        summary.update(
            original_HERMES_auc_mean=statistics.mean(row['original_HERMES']['reference_valid_auc'] for row in records),
            original_HERMES_aupr_mean=statistics.mean(row['original_HERMES']['reference_valid_aupr'] for row in records))

    write(RUN / 'STATUS.json', dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset='ONEIL', training_seed=0, formal_fits=5, jobs=records, gpu_check=gpu_check,
        cpu_check=get(DISPATCH / 'cpu_check/progress.json'), full_fivefold_summary=summary,
        architecture_changed=False, test_evaluated=False))
    lines = ['# ONEIL：知识分支不衰减，seed0 五折', '',
        '| fold | 状态 | 轮次 | 选定AUROC | 对应AUPRC |', '|---:|---|---:|---:|---:|']
    for row in records:
        p, s = row['progress'], row['selection']
        lines.append('| %d | %s | %s/3000 | %s | %s |' % (
            row['fold'], p['status'], p.get('epoch', 0), s.get('valid_auc', '—'), s.get('valid_aupr', '—')))
    lines += ['', '仅知识分支Adam weight_decay=0；Initializer及预测头仍为0.01。',
        '原架构、全部参数从头训练、原划分、3000轮、epoch1000–3000内部AUROC选轮不变。',
        'GPU检查通过后先运行fold0；其前两次整折更新和梯度顺序通过才放行其余折。',
        '优先卡0/卡1，统一队列等待显存；所有用户任务合计每卡最多2进程。',
        '五折全部核验完成后汇总；不读封存测试，不自动增加ALMANAC其他折、seed或模块。']
    if summary:
        lines.append('\n内部AUROC %.6f ± %.6f；AUPRC %.6f ± %.6f。' % (
            summary['valid_auc_mean'], summary['valid_auc_std'], summary['valid_aupr_mean'], summary['valid_aupr_std']))
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return all(row['progress']['status'] in ('complete', 'failed', 'cancelled') for row in records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    options = parser.parse_args()
    while True:
        if report() or not options.watch:
            break
        time.sleep(20)

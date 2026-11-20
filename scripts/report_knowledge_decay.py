"""Read-only CPU monitor for the ALMANAC fold0 optimizer diagnostic."""
import argparse
import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

RUN = ROOT / 'runs/knowledge_decay_ALMANAC_fold0_seed0_20261119'
DISPATCH = ROOT / 'artifacts/knowledge_decay_ALMANAC_fold0_seed0_20261119'
BASELINE = ROOT / 'runs/zero_gated_hgnn3_cv5_seed0_20261118/ALMANAC/fold0'


def get(path):
    return read(path) if path.exists() else {}


def report():
    reference = read(BASELINE / 'selection.json')
    d = RUN / 'ALMANAC/fold0'
    progress = get(d / 'progress.json') or dict(status='queued', epoch=0)
    selection = get(d / 'selection.json')
    diagnostics = get(d / 'SELECTED_DIAGNOSTICS.json')
    check = get(DISPATCH / 'gpu_check/progress.json')
    cpu = get(DISPATCH / 'cpu_check/progress.json')
    if progress.get('status') == 'complete':
        import json
        assert progress['optimizer_updates'] == progress['epoch'] == 3000
        assert [json.loads(x)['epoch'] for x in (d / 'history.jsonl').read_text().splitlines()] == list(range(1, 3001))
        assert sha(d / 'selected.pt') == selection['checkpoint_sha256']
        assert diagnostics['status'] == 'passed' and diagnostics['epoch'] == selection['epoch']
        exp = read(d / 'EXPERIMENT.json')
        assert exp['recipe']['decay_mode'] == 'knowledge_no_decay'
        assert exp['recipe']['knowledge_weight_decay'] == 0.
        assert exp['initialization_sha256'] == read(BASELINE / 'EXPERIMENT.json')['initialization_sha256']
        assert exp['baseline_metadata']['split'] == read(BASELINE / 'EXPERIMENT.json')['baseline_metadata']['split']
        write(d / 'FINAL_VERIFICATION.json', dict(status='passed', seed=0, fold=0, epochs=3000,
            selection=selection, initialization_and_split_match_reference=True,
            original_checkpoint_loaded=False, test_evaluated=False))
    record = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset='ALMANAC', fold=0, seed=0, reference=reference, progress=progress,
        selection=selection, diagnostics=diagnostics, gpu_check=check, cpu_check=cpu,
        architecture_changed=False, formal_fits=1, full_fivefold_result=False, test_evaluated=False)
    write(RUN / 'STATUS.json', record)
    lines = ['# ALMANAC fold0：知识分支权重衰减对照', '',
        '仅优化参数分组变化；结构、初始化、划分、3000轮和内部AUROC选轮规则保持一致。', '',
        '| 配方 | 状态 | 轮次 | AUROC | 对应AUPRC |', '|---|---|---:|---:|---:|',
        '| 原全部参数 decay=0.01 | complete | 3000 | %.6f | %.6f |' % (reference['valid_auc'], reference['valid_aupr']),
        '| 知识分支 decay=0 | %s | %s/3000 | %s | %s |' % (progress.get('status'), progress.get('epoch', 0),
            selection.get('valid_auc', '—'), selection.get('valid_aupr', '—')), '',
        '原参考为已完成的同折训练，不重新训练匹配对照；不是新五折结果。',
        '门控打开不等于性能提升；需结合SELECTED_DIAGNOSTICS中的实际传播增量、知识关闭干预和同折指标。',
        '诊断不使用封存测试，不自动追加五折、ONEIL或其他结构。']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return progress.get('status') in ('complete', 'failed', 'cancelled')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    options = parser.parse_args()
    while True:
        done = report()
        if done or not options.watch:
            break
        time.sleep(20)

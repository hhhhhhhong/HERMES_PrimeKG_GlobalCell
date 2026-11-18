"""Observe one ALMANAC fit only; never launch or promote another experiment."""
import argparse
import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write


def get(p):
    return read(p) if p.exists() else {}


def report(run, check):
    progress = get(run/'progress.json') or dict(status='queued', epoch=0)
    selection = get(run/'selection.json')
    experiment = get(run/'EXPERIMENT.json')
    baseline = experiment.get('baseline_metadata', {})
    state = dict(status=progress['status'], updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 dataset='ALMANAC', fold=0, training_seed=0, synergy_positions=[0, 0, 2],
                 training_synergy_message_dropout=0.2, progress=progress, selection=selection,
                 gpu_check=get(check/'progress.json'), baseline=baseline,
                 internal_validation_only=True, test_evaluated=False, other_folds_authorized=False)
    write(run/'STATUS.json', state)
    auc = selection.get('valid_auc')
    ap = selection.get('valid_aupr')
    score = '尚未到第1000轮选模型' if auc is None else f'AUROC={auc:.6f}, AUPR={ap:.6f}'
    text = ['# ALMANAC seed0/fold0：[0,0,2] + 20%协同边消息Dropout', '',
            f"状态：{progress['status']}；轮次：{progress.get('epoch',0)}/3000；GPU：{progress.get('physical_device','等待')}。", '',
            f'内部验证当前选定：{score}。', '',
            '仅单折，从头初始化、全部下游参数可训练，不加载旧checkpoint；封存测试不评估。',
            '原14关系和当前折正训练协同边在同一全局图；药物角色[0,0]、Cell角色2。',
            '20% Dropout仅作用于训练期协同边消息，在β权重计算之后施加，保留消息按1/0.8缩放；验证时关闭。',
            '昨日同折参照：AUROC=0.828298，AUPR=0.497304；原HERMES：AUROC=0.853738，AUPR=0.543928。',
            '历史五折剩余任务已搁置，不启动其他折或新seed。']
    (run/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return progress['status'] in ['complete', 'failed', 'cancelled']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--check', required=True, type=Path)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while True:
        done = report(args.run, args.check)
        if done or not args.watch:
            break
        time.sleep(15)

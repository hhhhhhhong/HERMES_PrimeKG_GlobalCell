"""Read-only CPU reporting; no GPU dispatch or automatic extra experiments."""
import argparse
import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write

RUN = ROOT/'runs/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118'
CHECK = ROOT/'artifacts/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118/gpu_check'
PREDECESSOR = ROOT/'runs/almanac_fold0_synergy002_drop20_seed0_20261118'


def get(path):
    return read(path) if path.exists() else {}


def report():
    progress = get(RUN/'progress.json') or dict(status='queued', epoch=0)
    check = get(CHECK/'progress.json') or dict(status='queued', epoch=0)
    selection = get(RUN/'selection.json')
    write(RUN/'STATUS.json', dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status=progress['status'], progress=progress, selection=selection, gpu_check=check,
        predecessor_final_selection=get(PREDECESSOR/'selection.json'),
        layers=3, propagation_gate_initialization=0., original_readout_gate_initialization=.05,
        full_cv5_authorized=False, test_evaluated=False))
    text = ['# 三层零门控真实超图传播：ALMANAC seed0/fold0','',
        f"训练：{progress['status']}，{progress.get('epoch',0)}/3000轮；GPU检查：{check['status']}。",'',
        f"当前按内部AUROC选定：{selection or '尚无正式结果'}",'',
        '原单层002+20% Dropout已完成：AUROC0.832360，AUPRC0.502258。',
        '原HERMES同折：AUROC0.853738，AUPRC0.543928。',
        '三层均基于真实超边更新全节点，逐层零门控；无相似图平滑、无新增损失。',
        '原末端读出gate保持0.05，不能称整模型初始预测恢复为单层版。',
        '当前仅一折，全下游从头可训练，不加载旧checkpoint；GPU检查通过后才放行fit。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return progress['status'] in ['complete','failed','cancelled'] or check['status']=='failed'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    while True:
        if report() or not a.watch:
            break
        time.sleep(20)

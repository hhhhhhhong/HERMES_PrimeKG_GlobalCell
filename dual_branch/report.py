"""Read-only CPU status writer for this implementation and its correctness checks."""
from .common import ROOT, DISPATCH
from global_cell.common import read, write


def get(path):
    return read(path) if path.exists() else {}


def main():
    preparation = get(DISPATCH / 'PREPARATION.json')
    checks = {ds: get(DISPATCH / ('cpu_check_' + ds) / 'progress.json') for ds in ['ALMANAC', 'ONEIL']}
    ready = all(p.get('status') == 'complete' for p in checks.values())
    status = 'CPU_verified' if ready else 'check_failed' if any(p.get('status') == 'failed' for p in checks.values()) else 'checking' if preparation else 'encoding'
    details = {ds: get(DISPATCH / ('cpu_check_' + ds) / 'CHECKS.json') for ds in checks}
    record = dict(status=status, version='dual_branch_v1_20261119',
        code_directory=str(ROOT / 'dual_branch'), documentation=str(ROOT / 'DUAL_BRANCH_V1.md'),
        preparation=preparation, encoding_progress=get(DISPATCH / 'ENCODING_PROGRESS.json'),
        input_audit=get(DISPATCH / 'TOKEN_AND_VECTOR_AUDIT.json'),
        CPU_checks=checks, CPU_check_results=details, unit_tests=get(DISPATCH / 'UNIT_CHECKS.json'),
        GPU_check_performed=False, formal_training_fits_submitted=0, test_evaluated=False,
        previous_training_run=str(ROOT / 'runs/zero_gated_hgnn3_cv5_seed0_20261118'))
    write(ROOT / 'DUAL_BRANCH_VERSION.json', record)
    lines = ['# 双分支第一版：实现与正确性检查', '', '当前状态：' + status, '',
        '全节点CODER语义图，与BRICS 2D/Uni-Mol 3D及Cell表达模态分支独立建模；以DrugPair×Cell读取并零投影残差融合。', '',
        '| 数据集 | 图节点 | Drug / Cell | CPU检查 | 更新次数 |', '|---|---:|---|---|---:|']
    for ds, progress in checks.items():
        nodes = preparation.get('datasets', {}).get(ds, {})
        lines.append('| %s | %s | %s / %s | %s | %s |' % (ds, nodes.get('nodes', '—'),
            nodes.get('drug_nodes', '—'), nodes.get('cell_nodes', '—'), progress.get('status', 'pending'), progress.get('optimizer_updates', 0)))
    lines += ['', '新版本修复了本地do_lower_case=False造成的UNK输入问题，全部实体重新编码；旧缓存与历史实验保留。',
        'Cell注释只用唯一匹配的Cellosaurus官方记录；歧义项保留名称，不猜组织或疾病。',
        '正确性检查不是预测效果，不是完整训练；本轮未提交新正式fit/五折，旧版训练继续。',
        'GPU检查尚未进行。所有新数据/检查/缓存均IFS2。']
    (DISPATCH / 'VALIDATION_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(dict(status=status, CPU_checks={ds: p.get('status', 'pending') for ds, p in checks.items()}), flush=True)


if __name__ == '__main__':
    main()

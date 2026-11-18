"""CPU observer: entry markers and measured memory placement, never dispatch."""
import argparse
import datetime
from pathlib import Path
import math
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

RUN = ROOT/'runs/zero_gated_hgnn3_cv5_seed0_20261118'
DISPATCH = ROOT/'artifacts/zero_gated_hgnn3_cv5_seed0_20261118'


def get(path):
    return read(path) if path.exists() else {}


def calibrate_memory(checks):
    """Change only this suite's placement budget using real 1024-query peaks."""
    record=DISPATCH/'MEMORY_BUDGET.json'
    if record.exists() or not all(c.get('status')=='complete' for c in checks.values()):return
    budgets={}
    for ds in ['ALMANAC','ONEIL']:
        check=read(DISPATCH/f'gpu_check_{ds}/CHECKS.json')
        assert check['status']=='passed' and check['training_rows']==1024
        peak=check['peak_cuda_memory_mib']
        assert peak and math.isfinite(peak)
        budgets[ds]=dict(measured_peak_mib=peak,reserve_mib=max(6144,math.ceil((1.15*peak+1024)/512)*512))
    registry_path=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009/auxiliary_jobs.json')
    registry=read(registry_path)
    keys=set()
    for job in registry['jobs']:
        if job['module']=='train_hermes_zero_gated_hgnn_cv5' and '--stage' in job['extra_args'] and job['extra_args'][job['extra_args'].index('--stage')+1]=='train':
            assert job['directory'].startswith(str(RUN)+'/')
            assert all(v==2 for v in job['gpu_process_limits'].values())
            job['memory_mib']=budgets[job['dataset']]['reserve_mib'];keys.add(job['key'])
    assert len(keys)==10
    write(registry_path,registry)
    write(record,dict(status='complete',datasets=budgets,changed_job_keys=sorted(keys),
        formula='ceil512(1.15*actual_chunk1024_check_peak+1024), minimum6144',
        model_unchanged=True,training_budget_unchanged=True,global_process_limit_unchanged=True))


def report():
    manifest = read(DISPATCH/'frozen/source_manifest.json')
    audit = read(DISPATCH/'FIVEFOLD_INPUT_AUDIT.json')
    checks = {ds: get(DISPATCH/f'gpu_check_{ds}/progress.json') for ds in ['ALMANAC','ONEIL']}
    calibrate_memory(checks)
    records, summaries = [], {}
    for ds in ['ALMANAC','ONEIL']:
        for fold in range(5):
            d = RUN/ds/f'fold{fold}'
            progress = get(d/'progress.json') or dict(status='queued', epoch=0)
            selection = get(d/'selection.json')
            row = dict(dataset=ds,fold=fold,progress=progress,selection=selection)
            records.append(row)
            # Each dataset's extra folds wait until fold0 has two real full-fit
            # optimizer updates with the expected gate/branch gradient order.
            marker = RUN/f'ENTRY_{ds}.json'
            if fold == 0 and progress.get('optimizer_updates',0)>=2 and not marker.exists():
                assert all(c.get('status')=='complete' for c in checks.values())
                exp = read(d/'EXPERIMENT.json')
                assert exp['source_sha256']==manifest['package']
                assert exp['recipe']['propagation_layers']==3 and exp['recipe']['propagation_gate_initialization']==0.
                assert exp['baseline_metadata']['split']==audit['folds'][f'{ds}/fold0']['split']
                g1,g2 = read(d/'GRADIENT_epoch1.json'),read(d/'GRADIENT_epoch2.json')
                blocked = ['alpha','beta','node_to_edge','edge_to_node','expression_edge','extra_alpha_beta']
                assert all(g1[k]==0. for k in blocked) and g1['propagation_gates']>0
                assert all(math.isfinite(v) and v>0 for v in g2.values())
                write(marker,dict(status='complete',dataset=ds,fold=0,full_fit_updates=2,
                    first_step_gate_then_branch_gradients_verified=True,source_manifest=str(DISPATCH/'frozen/source_manifest.json')))
            if progress['status']=='complete':
                lock = d/'FINAL_VERIFICATION.json'
                if not lock.exists():
                    assert progress['epoch']==progress['optimizer_updates']==3000
                    history = [read_record for read_record in (d/'history.jsonl').read_text().splitlines()]
                    import json
                    assert [json.loads(x)['epoch'] for x in history]==list(range(1,3001))
                    assert sha(d/'selected.pt')==selection['checkpoint_sha256']
                    exp = read(d/'EXPERIMENT.json')
                    assert exp['source_sha256']==manifest['package']
                    assert all(manifest['extra'][p]==h for p,h in exp['variant_sources'].items())
                    assert exp['recipe']['all_downstream_trainable'] and not exp['recipe']['original_checkpoint_loaded']
                    assert exp['baseline_metadata']['split']==audit['folds'][f'{ds}/fold{fold}']['split']
                    write(lock,dict(status='passed',epochs=3000,selection=selection,
                        source_manifest=str(DISPATCH/'frozen/source_manifest.json'),test_evaluated=False))
                row['verified_complete']=True
        completed = [r for r in records if r['dataset']==ds and r.get('verified_complete')]
        if len(completed)==5:
            aucs=[r['selection']['valid_auc'] for r in completed]
            aps=[r['selection']['valid_aupr'] for r in completed]
            baselines=[audit['folds'][f'{ds}/fold{i}']['baseline'] for i in range(5)]
            summaries[ds]=dict(folds=5,valid_auc_mean=statistics.mean(aucs),valid_auc_std=statistics.pstdev(aucs),
                valid_aupr_mean=statistics.mean(aps),valid_aupr_std=statistics.pstdev(aps),
                original_same_fold_auc_mean=statistics.mean(x['reference_valid_auc'] for x in baselines),
                original_same_fold_aupr_mean=statistics.mean(x['reference_valid_aupr'] for x in baselines),
                internal_validation_only=True,test_evaluated=False)
    write(RUN/'STATUS.json',dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variant='three-layer zero-gated real alpha/beta',training_seed=0,folds_per_dataset=5,
        jobs=records,gpu_checks=checks,full_fivefold_summaries=summaries,test_evaluated=False))
    text=['# 三层零门控超图：ALMANAC / ONEIL 各 seed0 五折','',
        '| 数据集 | fold | 状态 | 轮次 | 选定AUROC | 对应AUPRC |','|---|---:|---|---:|---:|---:|']
    for r in records:
        p,s=r['progress'],r['selection']
        text.append(f"| {r['dataset']} | {r['fold']} | {p['status']} | {p.get('epoch',0)}/3000 | {s.get('valid_auc','—')} | {s.get('valid_aupr','—')} |")
    text += ['', '全部从头训练、不冻结、不加载旧checkpoint；每层传播gate初始为0，末端原读出gate仍0.05。',
        '每套GPU检查通过后，fold0先完成2次整折更新并核验梯度，才放行其余4折。',
        '仅完整五折核验后汇总内部验证均值/总体标准差；封存测试不读取。',
        'SimilarityRefiner已撤销；旧单折未训练任务已归档，不重复启动。']
    for ds,s in summaries.items():
        text += ['',f"{ds}：内部AUROC {s['valid_auc_mean']:.6f} ± {s['valid_auc_std']:.6f}；AUPRC {s['valid_aupr_mean']:.6f} ± {s['valid_aupr_std']:.6f}。"]
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return all(r['progress']['status'] in ['complete','failed','cancelled'] for r in records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        if report() or not a.watch:break
        time.sleep(20)

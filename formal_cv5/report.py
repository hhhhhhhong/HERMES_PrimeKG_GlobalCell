"""Observe progress, verify the first full-fit updates and lock final folds.

This observer never changes a model, selects a recipe, or launches a fit.
All performance summaries refer to the fixed internal validation split.
"""
import argparse,datetime,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha
RUN=ROOT/'runs/global_cell_cv5_seed0_20261117'
FROZEN=ROOT/'artifacts/source_freeze_20261117'

def get(path):return read(path) if path.exists() else {}

def formal_entry(dataset):
    marker=RUN/f'ENTRY_{dataset}.json'
    if marker.exists():return
    d=RUN/dataset/'fold0';p=get(d/'progress.json')
    if p.get('optimizer_updates',0)<2:return
    exp=get(d/'EXPERIMENT.json');r=exp['recipe'];m=read(FROZEN/'source_manifest.json')
    assert exp['stage']=='train' and exp['fold']==0 and exp['seed']==0
    assert exp['source_sha256']==m['package']
    assert r['epochs']==3000 and r['query_chunk']==1024 and r['one_full_fit_update_per_epoch']
    assert not r['original_checkpoint_loaded'] and r['all_downstream_trainable'] and r['global_Cell_message_passing']
    for epoch in [1,2]:
        grad=read(d/f'GRADIENT_epoch{epoch}.json')
        assert grad and all(math.isfinite(v) and v>0 for v in grad.values())
    assert get(d/'STRUCTURE.json')['all_cell_roots_have_graph_tokens']
    write(marker,dict(status='complete',dataset=dataset,seed=0,fold=0,optimizer_updates_verified=2,
        original_full_fit=True,query_chunk=1024,all_component_gradients_nonzero=True,
        initialization_sha256=exp['initialization_sha256'],source_sha256=exp['source_sha256'],
        no_check_checkpoint_warm_start=True,no_additional_fit=True))

def report():
    import numpy as np
    results={};done=True
    lines=['# GlobalCell：seed0 五折状态','','每数据集一张Drug/Cell/PrimeKG全局异构超图。以下为固定内部验证结果，封存测试不参与。','']
    lines += ['| 数据集 | Fold | 状态 | 轮次 / 3000 | 物理GPU | 已选AUROC | 已选AUPR |','|---|---:|---|---:|---|---:|---:|']
    for ds in ['ALMANAC','ONEIL']:
        formal_entry(ds);folds=[]
        for fold in range(5):
            d=RUN/ds/f'fold{fold}';p=get(d/'progress.json') or dict(status='queued')
            sel=get(d/'selection.json');exp=get(d/'EXPERIMENT.json')
            folds.append(dict(fold=fold,progress=p,selection=sel,baseline=exp.get('baseline_metadata',{})))
            done &= p.get('status')=='complete'
            auc=f"{sel['valid_auc']:.6f}" if sel else '—';ap=f"{sel['valid_aupr']:.6f}" if sel else '—'
            lines.append(f"| {ds} | {fold} | {p.get('status')} | {p.get('epoch',0)} | {p.get('physical_device','等待')} | {auc} | {ap} |")
        result=dict(folds=folds)
        if all(f['progress']['status']=='complete' for f in folds):
            lock=RUN/f'CHECKPOINT_LOCK_{ds}.json'
            if not lock.exists():
                hashes={str(RUN/ds/f"fold{f['fold']}"/'selected.pt'):sha(RUN/ds/f"fold{f['fold']}"/'selected.pt') for f in folds}
                assert all(hashes[str(RUN/ds/f"fold{f['fold']}"/'selected.pt')]==f['selection']['checkpoint_sha256'] for f in folds)
                write(lock,dict(status='locked',dataset=ds,seed=0,checkpoint_sha256=hashes,selection='internal validation AUROC epochs1000..3000',test_evaluated=False))
            for key in ['valid_auc','valid_aupr']:
                vals=[f['selection'][key] for f in folds];refs=[f['baseline']['reference_'+key] for f in folds]
                result[key]=dict(values=vals,mean=float(np.mean(vals)),std_population=float(np.std(vals)),std_sample=float(np.std(vals,ddof=1)),
                    original_HERMES_same_split_mean=float(np.mean(refs)),paired_difference_mean=float(np.mean(np.array(vals)-np.array(refs))))
            lines.append('\n'+ds+' 内部验证五折汇总：'+str({k:v for k,v in result.items() if k!='folds'}))
        results[ds]=result
    state=dict(status='complete' if done else 'running_or_queued',updated_utc=datetime.datetime.utcnow().isoformat(),
        training_seed=0,folds_per_dataset=5,original_checkpoints_loaded=False,all_downstream_trainable=True,
        internal_validation_only=True,test_evaluated=False,results=results)
    write(RUN/'STATUS.json',state);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=report()
        if done or not a.watch:break
        time.sleep(15)

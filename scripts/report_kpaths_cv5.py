"""CPU observer for ten K-Paths fits; the shared queue alone starts GPUs."""
import argparse
import datetime
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dual_branch.kpaths_data import DISPATCH,RUN
from global_cell.common import read,write,sha


def get(path):return read(path) if path.exists() else {}


def report():
    rows=[];summaries={}
    for ds in ['ALMANAC','ONEIL']:
        check=get(DISPATCH/('gpu_check_'+ds)/'CHECKS.json')
        marker=DISPATCH/('GPU_READY_'+ds+'.json')
        if check.get('status')=='passed' and not marker.exists():
            cpu=read(DISPATCH/('cpu_'+ds)/'EXPERIMENT.json')
            gpu=read(DISPATCH/('gpu_check_'+ds)/'EXPERIMENT.json')
            assert cpu['initialization_sha256']==gpu['initialization_sha256']
            assert check['training_rows']==1024 and check['optimizer_updates']==3
            assert check['path_reading']['status']=='passed'
            write(marker,dict(status='complete',GPU_check=check,
                CPU_GPU_initialization_identical=True,no_performance_claim=True))
        dataset_rows=[]
        for fold in range(5):
            directory=RUN/ds/('fold%d'%fold)
            progress=get(directory/'progress.json') or dict(status='queued',epoch=0,optimizer_updates=0)
            selection=get(directory/'selection.json')
            row=dict(dataset=ds,fold=fold,progress=progress,selection=selection,
                diagnostic_best_aupr=get(directory/'best_aupr.json'))
            if fold==0 and progress.get('optimizer_updates',0)>=3 and not (RUN/('ENTRY_'+ds+'.json')).exists():
                experiment=read(directory/'EXPERIMENT.json')
                assert experiment['initialization_sha256']==read(DISPATCH/('cpu_'+ds)/'EXPERIMENT.json')['initialization_sha256']
                for step in [2,3]:
                    gradient=read(directory/('GRADIENT_epoch%d.json'%step))['component_gradient_L2']
                    assert all(v>0 for v in gradient.values()) and gradient['path_encoder']>0
                write(RUN/('ENTRY_'+ds+'.json'),dict(status='complete',full_fit_updates=3,
                    initialized_from_scratch=True,all_component_gradients_verified=True,
                    path_reader_nonzero_gradient=True,no_performance_claim=True))
            if progress['status']=='complete':
                assert progress['epoch']==progress['optimizer_updates']==3000
                verified=directory/'FINAL_VERIFICATION.json'
                if not verified.exists():
                    assert sha(directory/'selected.pt')==selection['checkpoint_sha256']
                    write(verified,dict(status='passed',updates=3000,selection=selection,
                        internal_validation_only=True,sealed_test_evaluated=False))
            dataset_rows.append(row)
        if all(r['progress']['status']=='complete' for r in dataset_rows):
            summaries[ds]={metric:dict(mean=statistics.mean(r['selection'][metric] for r in dataset_rows),
                standard_deviation=statistics.pstdev(r['selection'][metric] for r in dataset_rows))
                for metric in ['valid_auc','valid_aupr']}
        rows+=dataset_rows
    status=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        version='target_kpaths_global_Cell_cv5',datasets=['ALMANAC','ONEIL'],seed=0,
        folds_per_dataset=5,epochs=3000, jobs=rows,complete_fivefold_internal_validation=summaries,
        completed=sum(r['progress']['status']=='complete' for r in rows),
        running=sum(r['progress']['status']=='running' for r in rows),
        queued=sum(r['progress']['status']=='queued' for r in rows),
        failed=sum(r['progress']['status']=='failed' for r in rows),sealed_test_evaluated=False)
    write(RUN/'STATUS.json',status)
    lines=['# K-Paths + 全局 Cell / Synergy：seed0 五折','',
        '内部验证指标；主选点为 epoch1000–3000 内部 AUROC。不是外层或封存测试结果。','',
        '| Dataset | Fold | Status | Epoch | Selected AUROC | Selected AUPRC |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        s=r['selection'];p=r['progress']
        lines.append('| %s | %d | %s | %d | %s | %s |'%(r['dataset'],r['fold'],p['status'],p.get('epoch',0),
            ('%.6f'%s['valid_auc']) if s else '',('%.6f'%s['valid_aupr']) if s else ''))
    if summaries:lines+=['',str(summaries)]
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return all(r['progress']['status'] in ('complete','failed') for r in rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        if report() or not args.watch:break
        time.sleep(5)

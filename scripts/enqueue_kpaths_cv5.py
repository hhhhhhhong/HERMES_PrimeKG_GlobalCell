"""Freeze and submit the user-authorized two-dataset seed0 fivefold suite."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dual_branch.common import environment,RUNTIME
from dual_branch.kpaths_data import PATH_DATA,DISPATCH,RUN
from global_cell.common import read,write,sha

PROJECT=ROOT.parent/'mcdsp_moe/MCDSP-main'
MODULE='train_hermes_kpaths_cv5'
BRIDGE=PROJECT/'tools/paper_routes'/(MODULE+'.py')


def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    environment()
    assert not (DISPATCH/'LAUNCH.json').exists(),'No duplicate folds'
    assert read(DISPATCH/'PREPARATION.json')['status']=='complete'
    assert read(DISPATCH/'UNIT_CHECKS.json')['status']=='passed'
    for ds in ['ALMANAC','ONEIL']:
        check=read(DISPATCH/('cpu_'+ds)/'CHECKS.json')
        assert check['status']=='passed' and check['path_reading']['status']=='passed'
        write(DISPATCH/('CPU_READY_'+ds+'.json'),dict(status='complete',check=check))
    before=read(ROOT/'artifacts/transformer_repair_20261120/frozen/manifest.json')
    assert all(sha(p)==v for p,v in before['source'].items())
    assert all(sha(p)==v for p,v in before['inputs'].items())
    source=dict(before['source'])
    names=['dual_branch/kpaths_retrieval.py','dual_branch/kpaths_data.py','dual_branch/kpaths_model.py',
        'dual_branch/kpaths_checks.py','dual_branch/kpaths_cli.py','scripts/prepare_kpaths_cv5.py',
        'scripts/enqueue_kpaths_cv5.py','scripts/report_kpaths_cv5.py',
        'tests/test_kpaths_retrieval.py','tests/test_kpaths_model.py']
    source.update({str(ROOT/n):sha(ROOT/n) for n in names})
    source[str(BRIDGE)]=sha(BRIDGE)
    inputs=dict(before['inputs'])
    for ds in ['ALMANAC','ONEIL']:
        for name in ['graph.pt','paths.pt','all_nodes_coder.pt','paths.jsonl','PROVENANCE.json']:
            p=PATH_DATA/ds/name;inputs[str(p)]=sha(p)
    inputs[str(DISPATCH/'PREPARATION.json')]=sha(DISPATCH/'PREPARATION.json')
    manifest=DISPATCH/'frozen/manifest.json'
    write(manifest,dict(source=source,inputs=inputs,version='target_kpaths_global_Cell_cv5_20261120',
        original_53_training_sources_unchanged=True,from_scratch=True))
    for name in source:
        p=Path(name);relative=str(p.relative_to(ROOT)) if ROOT in p.parents else 'external/'+str(p).lstrip('/')
        target=DISPATCH/'frozen/source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    previous_audit=ROOT/'artifacts/dual_branch_gemini_cv5_seed0_20261119/FIVEFOLD_INPUT_AUDIT.json'
    split_audit=read(previous_audit)
    assert split_audit['status']=='passed' and len(split_audit['folds'])==10
    write(DISPATCH/'FIVEFOLD_INPUT_AUDIT.json',dict(split_audit,
        legacy_audit=str(previous_audit),all_split_inputs_rehashed=True,
        new_path_cache_label_free=True,Cell_expression_unchanged=True,
        fold_synergy_added_by_unchanged_helper=True,from_scratch=True))
    common=dict(module=MODULE,result_storage_root=str(ROOT),gpus=[0,1,2,3],
        gpu_process_limits={str(i):2 for i in range(4)})
    extra=['--expected-manifest',str(manifest)]
    checks=[dict(common,key='aux_hermes_kpaths_%s_check_20261120'%ds,dataset=ds,
        directory=str(DISPATCH/('gpu_check_'+ds)),memory_mib=6144 if ds=='ALMANAC' else 4096,
        requires=[str(DISPATCH/('CPU_READY_'+ds+'.json'))],
        extra_args=['--stage','check','--fold','0','--check-rows','1024']+extra)
        for ds in ['ALMANAC','ONEIL']]
    fits=[dict(common,key='aux_hermes_kpaths_%s_fold%d_20261120'%(ds,fold),dataset=ds,
        directory=str(RUN/ds/('fold%d'%fold)),memory_mib=6144 if ds=='ALMANAC' else 4096,
        requires=[str(DISPATCH/('GPU_READY_'+ds+'.json'))]+([] if fold==0 else [str(RUN/('ENTRY_'+ds+'.json'))]),
        extra_args=['--stage','train','--fold',str(fold)]+extra)
        for fold in range(5) for ds in ['ALMANAC','ONEIL']]
    jobs=checks+fits
    assert len({j['directory'] for j in jobs})==12
    state=read(RUNTIME/'pipeline_progress.json');policy=read(RUNTIME/'resource_policy.json')
    assert policy['max_user_processes_per_gpu']==2 and not policy.get('gpu_process_limit_overrides')
    assert state['status']=='running' and alive(state['pid'])
    command=Path('/proc/%d/cmdline'%state['pid']).read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in command and '--auxiliary-keys' in command
    assert '--runtime-directory' in command and str(RUNTIME) in command and '--gpu-process-limit' not in command
    registry=read(RUNTIME/'auxiliary_jobs.json')
    assert not {j['key'] for j in jobs}&{j['key'] for j in registry['jobs']}
    write(DISPATCH/'REGISTRY_BEFORE.json',registry);write(DISPATCH/'STATE_BEFORE.json',state)
    old_workers={j['pid']:j for j in state['active'] if alive(j['pid'])}
    # Restart only this CPU controller; independent GPU sessions are adopted.
    os.kill(state['pid'],signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']):break
        time.sleep(.05)
    assert not alive(state['pid'])
    assert read(RUNTIME/'auxiliary_jobs.json')==registry
    registry['jobs']=jobs+registry['jobs']
    registry['kpaths_cv5_authorization_20261120']=dict(user_request='五折',
        datasets=['ALMANAC','ONEIL'],seed=0,folds=list(range(5)),formal_fits=10,
        no_freezing=True,no_old_checkpoint=True,original_global_Cell_synergy_preserved=True)
    write(RUNTIME/'auxiliary_jobs.json',registry)
    command += [j['key'] for j in jobs]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (DISPATCH/'controller.log').open('a') as stream:
        controller=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    with (DISPATCH/'report.log').open('a') as stream:
        reporter=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/report_kpaths_cv5.py'),'--watch'],
            cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(DISPATCH/'LAUNCH.json',dict(controller=controller.pid,reporter=reporter.pid,command=command,
        jobs=jobs,old_workers=old_workers,old_GPU_workers_signalled=False,
        formal_fits=10,source_manifest=str(manifest),all_user_GPU_process_cap=2))
    pointer=dict(variant='K-Paths evidence + global Cell/Synergy + prediction Transformer',
        status='submitted',run_directory=str(RUN),dispatch=str(DISPATCH),status_file=str(RUN/'STATUS.json'),
        datasets=['ALMANAC','ONEIL'],folds_per_dataset=5,seed=0,epochs=3000,
        all_downstream_trainable=True,old_checkpoint_loaded=False,all_outputs_on_IFS2=True)
    write(ROOT/'KPATHS_RUN.json',pointer)
    write(ROOT/'CURRENT_RUN.json',dict(pointer,previous_run=read(ROOT/'CURRENT_RUN.json')))
    for pid,job in old_workers.items():
        assert alive(pid) or read(Path(job['directory'])/'progress.json')['status']=='complete'
    print(dict(status='submitted',controller=controller.pid,reporter=reporter.pid,GPU_checks=2,formal_fits=10))


if __name__=='__main__':main()

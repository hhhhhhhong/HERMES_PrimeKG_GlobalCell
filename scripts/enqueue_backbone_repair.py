"""Freeze verified repair and enqueue exactly two checks + two fold0 pilots."""
import fcntl
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
from global_cell.common import read,write,sha
D=ROOT/'artifacts/backbone_repair_fold0_20261123'
RUN=ROOT/'runs/backbone_repair_fold0_20261123'
PROJECT=ROOT.parent/'mcdsp_moe/MCDSP-main'
MODULE='train_hermes_backbone_repair'


def main():
    environment()
    assert not (D/'LAUNCH.json').exists(), 'Already submitted'
    checks=[read(D/('cpu_'+mode)/'CHECKS.json') for mode in ['none','real']]
    assert all(c['status']=='passed' and c['optimizer_updates']==3 for c in checks)
    assert checks[0]['initialization_sha256']==checks[1]['initialization_sha256']
    prior=read(ROOT/'artifacts/entity_token_adapter_cv5_seed0_20261122/frozen/manifest.json')
    for kind in ['source','inputs']:
        assert all(sha(p)==s for p,s in prior[kind].items()), kind
    paths=[ROOT/p for p in ['dual_branch/backbone_repair.py','dual_branch/backbone_repair_training.py',
        'dual_branch/backbone_repair_cli.py','tests/test_backbone_repair.py','BACKBONE_REPAIR.md',
        'scripts/enqueue_backbone_repair.py']]
    paths.append(PROJECT/'tools/paper_routes'/ (MODULE+'.py'))
    source=dict(prior['source']);source.update({str(p):sha(p) for p in paths})
    frozen=D/'frozen';assert not frozen.exists()
    for name in source:
        p=Path(name)
        relative=p.relative_to(ROOT) if ROOT in p.parents else Path('external')/str(p).lstrip('/')
        dest=frozen/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest=frozen/'manifest.json'
    write(manifest,dict(source=source,inputs=prior['inputs'],reference_initialization_sha256=checks[0]['initialization_sha256'],
        version=D.name,previous125_sources_unchanged=True))
    write(D/'CPU_VERIFIED.json',dict(status='complete',checks=checks,unit_tests='2 passed',
        same_initialization=True,no_performance_claim=True))
    policy=read(RUNTIME/'resource_policy.json');state=read(RUNTIME/'pipeline_progress.json')
    assert policy['gpu_process_limit_overrides']=={str(i):3 for i in range(4)}
    common=dict(module=MODULE,dataset='ALMANAC',result_storage_root=str(ROOT),gpus=[3,1,0,2],
        gpu_process_limits={str(i):3 for i in range(4)},memory_mib=8192)
    jobs=[]
    for mode in ['none','real']:
        jobs.append(dict(common,key='aux_hermes_backbone_repair_'+mode+'_check_20261123',
            directory=str(D/('gpu_'+mode)),requires=[str(D/'CPU_VERIFIED.json')],
            extra_args=['--stage','check','--knowledge',mode,'--check-rows','2048','--expected-manifest',str(manifest)]))
    for mode in ['none','real']:
        jobs.append(dict(common,key='aux_hermes_backbone_repair_'+mode+'_fold0_20261123',directory=str(RUN/mode),
            requires=[str(D/('gpu_'+m)/'progress.json') for m in ['none','real']],
            extra_args=['--stage','train','--knowledge',mode,'--expected-manifest',str(manifest)]))
    oldkeys=policy['selected_auxiliary_keys'];pid=state['pid']
    write(D/'STATE_BEFORE.json',state);write(D/'POLICY_BEFORE.json',policy)
    queue=PROJECT/'tools/paper_routes/queue.py';text=queue.read_text()
    write(D/'QUEUE_BEFORE.json',dict(sha256=sha(queue)))
    shutil.copy2(queue,D/'queue_before.py')
    assert "'"+MODULE+"'" not in text
    # Only extend the existing module allowlist; resource accounting unchanged.
    queue.write_text(text.replace("assert job['module'] in [", "assert job['module'] in ['"+MODULE+"',",1))
    proc=Path('/proc')/str(pid)
    if proc.exists() and state.get('status')!='complete':
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert proc.stat().st_uid==os.getuid() and 'tools.paper_routes.queue ' in cmd and str(RUNTIME) in cmd
        os.kill(pid,signal.SIGTERM)  # CPU queue only, never its GPU workers.
    with (RUNTIME/'pipeline.lock').open('a') as lock:
        for i in range(120):
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError: time.sleep(.25)
        else: raise RuntimeError('Existing controller still holds pipeline lock')
        with (RUNTIME/'auxiliary_registry_edit.lock').open('a') as edit:
            fcntl.flock(edit,fcntl.LOCK_EX)
            registry=read(RUNTIME/'auxiliary_jobs.json');write(D/'REGISTRY_BEFORE.json',registry)
            assert not {j['key'] for j in jobs}&{j['key'] for j in registry['jobs']}
            registry['jobs']=jobs+registry['jobs'];write(RUNTIME/'auxiliary_jobs.json',registry)
        fcntl.flock(lock,fcntl.LOCK_UN)
    command=[sys.executable,'-B','-u','-m','tools.paper_routes.queue',
        '--output','/ifs/data/liushuhong/work2/results/paper_routes_20261009','--runtime-directory',str(RUNTIME),
        '--gpus','3','1','0','2','--qa-gpu','-1',
        '--gpu-process-limit','0=3','--gpu-process-limit','1=3',
        '--gpu-process-limit','2=3','--gpu-process-limit','3=3',
        '--auxiliary-only','--no-source-checks','--auxiliary-keys']+oldkeys+[j['key'] for j in jobs]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (D/'controller.log').open('a') as stream:
        controller=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(D/'LAUNCH.json',dict(controller=controller.pid,previous_controller=pid,command=command,jobs=jobs,
        previous_workers=state.get('active',[]),GPU_workers_signalled=False,global_user_process_cap=3,
        old_jobs_preserved=len(registry['jobs'])-4,formal_fits=2,checks=2))
    pointer=dict(status='submitted',run_directory=str(RUN),dispatch=str(D),dataset='ALMANAC',fold=0,seed=0,
        variants=['none','real'],model='dual_branch.backbone_repair.BackboneRepairModel',
        entry='dual_branch.backbone_repair_cli',frozen_manifest=str(manifest),selection='validation AUROC',
        source_and_inputs_preserved=True,no_new_fivefold=True)
    write(ROOT/'BACKBONE_REPAIR_RUN.json',pointer)
    print(pointer,flush=True)

if __name__=='__main__': main()

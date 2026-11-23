"""Register the authorized two-dataset seed0 fivefold in the shared controller."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import RUNTIME, environment
from dual_branch.hypothesis_common import DISPATCH, RUN, VERSION
from global_cell.common import read, write, sha
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'
MODULE = 'train_hermes_hypothesis_evidence_cv5'
SHARED_OUTPUT = '/ifs/data/liushuhong/work2/results/paper_routes_20261009'


def main():
    environment()
    assert not (DISPATCH / 'LAUNCH.json').exists(), 'Do not duplicate an existing submission'
    frozen_path = DISPATCH / 'frozen/manifest.json'
    frozen = read(frozen_path)
    for kind in ['source', 'inputs']:
        assert all(sha(p) == s for p,s in frozen[kind].items()), kind
    assert read(ROOT / 'artifacts/hypothesis_evidence_20261122/VERIFIED.json')['status'] == 'CPU_verified'
    for ds in ['ALMANAC', 'ONEIL']:
        checks = read(DISPATCH / ('cpu_'+ds) / 'CHECKS.json')
        assert checks['status'] == 'passed' and checks['graph_tables_per_check_epoch'] == 2
        write(DISPATCH / ('CPU_READY_'+ds+'.json'), dict(status='complete', checks=checks))
    common = dict(module=MODULE, result_storage_root=str(ROOT), gpus=[3,0,2,1],
        gpu_process_limits={str(i):2 for i in range(4)})
    extra = ['--expected-manifest', str(frozen_path)]
    jobs = [dict(common, key='aux_hermes_hypothesis_%s_check_20261122'%ds, dataset=ds,
        directory=str(DISPATCH / ('gpu_check_'+ds)), memory_mib=8192 if ds=='ALMANAC' else 6144,
        requires=[str(DISPATCH / ('CPU_READY_'+ds+'.json'))],
        extra_args=['--stage','check','--fold','0','--check-rows','2048']+extra)
        for ds in ['ALMANAC','ONEIL']]
    jobs += [dict(common, key='aux_hermes_hypothesis_%s_fold%d_20261122'%(ds,fold), dataset=ds,
        directory=str(RUN / ds / ('fold%d'%fold)), memory_mib=8192 if ds=='ALMANAC' else 6144,
        requires=[str(DISPATCH / ('GPU_READY_'+ds+'.json'))]+([] if fold==0 else [str(RUN / ('ENTRY_'+ds+'.json'))]),
        extra_args=['--stage','train','--fold',str(fold)]+extra)
        for fold in range(5) for ds in ['ALMANAC','ONEIL']]
    assert len({j['directory'] for j in jobs})==12
    state=read(RUNTIME/'pipeline_progress.json')
    policy=read(RUNTIME/'resource_policy.json')
    assert policy['max_user_processes_per_gpu']==2 and not policy.get('gpu_process_limit_overrides')
    old_keys=policy['selected_auxiliary_keys']
    write(DISPATCH/'STATE_BEFORE.json',state)
    write(DISPATCH/'POLICY_BEFORE.json',policy)
    old_active=state.get('active',[])
    pid=state.get('pid')
    process=Path('/proc')/str(pid)
    if process.exists() and state.get('status')!='complete':
        assert process.stat().st_uid==os.getuid()
        cmd=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'tools.paper_routes.queue ' in cmd and str(RUNTIME) in cmd
        os.kill(pid,signal.SIGTERM)  # CPU controller ONLY; worker sessions are independent.
    lock=(RUNTIME/'pipeline.lock').open('a')
    for attempt in range(60):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.25)
    else:
        raise RuntimeError('Shared pipeline lock still held; refusing a second scheduler')
    try:
        edit=(RUNTIME/'auxiliary_registry_edit.lock').open('a')
        fcntl.flock(edit,fcntl.LOCK_EX)
        registry=read(RUNTIME/'auxiliary_jobs.json')
        assert not {j['key'] for j in jobs}&{j['key'] for j in registry['jobs']}
        write(DISPATCH/'REGISTRY_BEFORE.json',registry)
        registry['jobs']=jobs+registry['jobs']
        registry['hypothesis_evidence_authorization_20261122']=dict(user_request='那你直接运行5折',
            datasets=['ALMANAC','ONEIL'],seed=0,folds=list(range(5)),formal_fits=10,
            exact_batch_target_edge_exclusion_in_both_streams=True,collaboration_conditions_gamma=True,
            old_GPU_workers_signalled=False,sealed_test_evaluated=False)
        write(RUNTIME/'auxiliary_jobs.json',registry)
        fcntl.flock(edit,fcntl.LOCK_UN)
        edit.close()
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    command=[sys.executable,'-B','-u','-m','tools.paper_routes.queue','--output',SHARED_OUTPUT,
        '--runtime-directory',str(RUNTIME),'--gpus','3','0','2','1','--qa-gpu','-1',
        '--auxiliary-only','--no-source-checks','--auxiliary-keys']+old_keys+[j['key'] for j in jobs]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (DISPATCH/'controller.log').open('a') as stream:
        controller=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    with (DISPATCH/'report.log').open('a') as stream:
        reporter=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/report_hypothesis_evidence_cv5.py'),'--watch'],
            cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(DISPATCH/'LAUNCH.json',dict(controller=controller.pid,reporter=reporter.pid,command=command,jobs=jobs,
        previous_controller=pid,previous_active_workers=old_active,GPU_checks=2,formal_fits=10,
        all_user_GPU_process_cap=2,old_GPU_workers_signalled=False,source_manifest=str(frozen_path),
        old_registry_jobs_preserved=len(registry['jobs'])-12))
    for name in ['CURRENT_RUN','CURRENT_MODEL']:
        write(DISPATCH/('PREVIOUS_'+name+'.json'),read(ROOT/(name+'.json')))
    pointer=dict(variant='Exact target exclusion + collaboration-conditioned evidence',status='submitted',
        run_directory=str(RUN),dispatch=str(DISPATCH),status_file=str(RUN/'STATUS.json'),datasets=['ALMANAC','ONEIL'],
        folds_per_dataset=5,seed=0,epochs=3000,all_downstream_trainable=True,old_checkpoint_loaded=False,
        all_outputs_on_IFS2=True,primary_selection='Val AUPRC from first evaluation',source_manifest=str(frozen_path),
        original_splits_have_duplicate_supervision=True)
    write(ROOT/'HYPOTHESIS_EVIDENCE_RUN.json',pointer)
    write(ROOT/'CURRENT_RUN.json',pointer)
    write(ROOT/'CURRENT_MODEL.json',dict(pointer,model_class='dual_branch.hypothesis_evidence.HypothesisEvidenceModel',
        code=str(ROOT/'dual_branch/hypothesis_evidence.py'),training_entry='dual_branch.hypothesis_cli'))
    print(dict(status='submitted',controller=controller.pid,reporter=reporter.pid,GPU_checks=2,formal_fits=10),flush=True)

if __name__=='__main__':
    main()

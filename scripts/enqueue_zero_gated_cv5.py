"""Replace the unstarted pilot with two datasets' authorized seed0 CV5."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha,sources

PROJECT=Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
RUNTIME=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
DISPATCH=ROOT/'artifacts/zero_gated_hgnn3_cv5_seed0_20261118'
RUN=ROOT/'runs/zero_gated_hgnn3_cv5_seed0_20261118'
PILOT=ROOT/'artifacts/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118'


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    assert not (DISPATCH/'LAUNCH.json').exists()
    assert read(DISPATCH/'ENTRY_EQUIVALENCE.json')['status']=='passed'
    assert read(DISPATCH/'CHECKPOINT_EQUIVALENCE.json')['status']=='passed'
    assert read(PILOT/'cpu_check/CHECKS.json')['status']=='passed'
    assert read(DISPATCH/'cpu_check_ONEIL/CHECKS.json')['status']=='passed'
    assert read(DISPATCH/'FIVEFOLD_INPUT_AUDIT.json')['status']=='passed'
    previous=read(PILOT/'LAUNCH.json')
    oldkeys={j['key'] for j in previous['checks']+previous['jobs']}
    state=read(RUNTIME/'pipeline_progress.json')
    assert state['status']=='running' and state['pid']==previous['controller'] and not state['active']
    registry=read(RUNTIME/'auxiliary_jobs.json')
    oldjobs=[j for j in registry['jobs'] if j['key'] in oldkeys]
    assert len(oldjobs)==2 and all(not (Path(j['directory'])/'progress.json').exists() for j in oldjobs)
    oldmanifest=read(PILOT/'frozen/source_manifest.json')
    assert sources()==oldmanifest['package']
    assert sha(ROOT/'models/zero_gated_hgnn.py')==oldmanifest['extra'][str(ROOT/'models/zero_gated_hgnn.py')]
    frozen=DISPATCH/'frozen';frozen.mkdir()
    extras=[ROOT/'models/zero_gated_hgnn.py',ROOT/'models/zero_gated_checkpoint.py',ROOT/'scripts/train_zero_gated_hgnn_cv5.py',
        ROOT/'scripts/enqueue_zero_gated_cv5.py',ROOT/'scripts/report_zero_gated_cv5.py',
        ROOT/'tests/test_zero_gated_hgnn.py',ROOT/'tests/test_zero_gated_checkpoint.py',PROJECT/'tools/paper_routes/train_hermes_zero_gated_hgnn_cv5.py']
    write(frozen/'source_manifest.json',dict(package=sources(),extra={str(p):sha(p) for p in extras}))
    inputs=read(ROOT/'artifacts/source_freeze_20261117/input_manifest.json')
    assert all(sha(p)==h for p,h in inputs.items())
    write(frozen/'input_manifest.json',inputs)
    for p in [ROOT/name for name in sources()]+extras:
        relative='project/'+str(p.relative_to(ROOT)) if ROOT in p.parents else 'bridge/'+p.name
        dest=frozen/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    common=dict(module='train_hermes_zero_gated_hgnn_cv5',result_storage_root=str(ROOT),gpus=[1,0,3,2],
        gpu_process_limits={str(i):2 for i in range(4)})
    extra=['--expected-source-manifest',str(frozen/'source_manifest.json'),'--expected-input-manifest',str(frozen/'input_manifest.json')]
    checks=[];jobs=[]
    for ds in ['ALMANAC','ONEIL']:
        checks.append(dict(common,key=f'aux_hermes_zero_gated_hgnn3_cv5_{ds}_check_20261118',dataset=ds,
            directory=str(DISPATCH/f'gpu_check_{ds}'),memory_mib=11000,requires=[],
            extra_args=['--stage','check','--fold','0']+extra))
    check_dependencies=[str(Path(j['directory'])/'progress.json') for j in checks]
    for fold in range(5):
        for ds in ['ALMANAC','ONEIL']:
            jobs.append(dict(common,key=f'aux_hermes_zero_gated_hgnn3_cv5_{ds}_fold{fold}_20261118',dataset=ds,
                directory=str(RUN/ds/f'fold{fold}'),memory_mib=32000,
                requires=check_dependencies+([] if fold==0 else [str(RUN/f'ENTRY_{ds}.json')]),
                extra_args=['--stage','train','--fold',str(fold)]+extra))
    assert len({j['directory'] for j in checks+jobs})==12
    assert not {j['key'] for j in checks+jobs}&{j['key'] for j in registry['jobs']}
    write(DISPATCH/'REGISTRY_BEFORE.json',registry);write(DISPATCH/'STATE_BEFORE.json',state)
    command=Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in command and '--runtime-directory' in command and str(RUNTIME) in command
    assert '--gpu-process-limit' not in command
    command=[arg for arg in command if arg not in oldkeys]+[j['key'] for j in checks+jobs]
    # Only CPU controller/reporting processes are restarted. No GPU is signaled.
    for pid,expected in [(previous['reporter'],'report_zero_gated_hgnn.py'),(state['pid'],'tools.paper_routes.queue')]:
        if alive(pid):
            assert expected in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
            os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']) and not alive(previous['reporter']):break
        time.sleep(.1)
    assert not alive(state['pid']) and not alive(previous['reporter'])
    assert read(RUNTIME/'auxiliary_jobs.json')==registry
    for j in oldjobs:
        record=read(RUNTIME/'processes'/f"{j['key']}.json") if (RUNTIME/'processes'/f"{j['key']}.json").exists() else {}
        assert not record or not alive(record['pid']), 'Do not replace an unexpectedly started worker'
    registry['jobs']=[j for j in registry['jobs'] if j['key'] not in oldkeys]+checks+jobs
    registry.setdefault('shelved_experiments',[]).append(dict(method='unstarted zero-gated ALMANAC pilot',jobs=oldjobs,
        reason='User expanded to ALMANAC and ONEIL seed0 five folds; fold0 is included once in the CV5 run',
        archive=str(DISPATCH),automatic_resume_allowed=False))
    write(RUNTIME/'auxiliary_jobs.json',registry)
    write(PILOT/'SUPERSEDED_BY_CV5.json',dict(status='superseded_before_training',new_run=str(RUN),
        old_job_keys=sorted(oldkeys),GPU_workers_ever_started=False,CPU_results_preserved=True))
    oldrun=ROOT/'runs/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118'
    write(oldrun/'STATUS.json',dict(status='superseded',new_run=str(RUN),reason='Fold0 included in authorized CV5; not duplicated'))
    (oldrun/'RESULTS.md').write_text('# 已并入双数据集五折\n\n单折GPU任务从未启动，已并入新五折队列，CPU检查记录保留。\n')
    write(DISPATCH/'PLAN.json',dict(datasets=['ALMANAC','ONEIL'],folds=list(range(5)),seed=0,
        formal_fits=10,epochs=3000,select_from_epoch=1000,propagation_layers=3,propagation_gate_initialization=0.,
        original_readout_gate_initialization=.05,all_downstream_trainable=True,old_checkpoint_loaded=False,
        model_and_recipe_unchanged=True,test_evaluated=False,checks=checks,jobs=jobs,max_user_processes_per_gpu=2,
        memory_reserve_mib=dict(check=11000,fit_initial=32000,fit_calibration='ceil512(1.15*actual_chunk1024_check_peak+1024)'),
        activation_checkpointing='alpha/beta and relation interaction; preserve RNG, no formula changes',no_automatic_extra_seeds=True))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    for key,suffix in dict(TMPDIR='tmp',HF_HOME='cache/huggingface',TORCH_HOME='cache/torch',XDG_CACHE_HOME='cache/xdg',
        WANDB_DIR='wandb',WANDB_CACHE_DIR='wandb_cache',WANDB_DATA_DIR='wandb_data',WANDB_CONFIG_DIR='wandb_config').items():
        path=RUNTIME/suffix;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    with (DISPATCH/'controller.log').open('a') as f:
        controller=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    scope=ROOT/'artifacts/formal_cv5_seed0_20261117/gpu1_two_more/POLICY_STATE.json'
    policy=read(scope);assert not policy['active_scopes'];policy.update(controller=controller.pid,command=command);write(scope,policy)
    with (DISPATCH/'report.log').open('a') as f:
        reporter=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/report_zero_gated_cv5.py'),'--watch'],
            cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(DISPATCH/'LAUNCH.json',dict(controller=controller.pid,command=command,reporter=reporter.pid,
        checks=checks,jobs=jobs,run=str(RUN),source_manifest=str(frozen/'source_manifest.json'),old_GPU_workers_signalled=False))
    write(ROOT/'CURRENT_RUN.json',dict(variant='3-layer zero-gated alpha/beta / both datasets CV5',
        datasets=['ALMANAC','ONEIL'],folds_per_dataset=5,seed=0,epochs=3000,formal_fits=10,
        run_directory=str(RUN),status_file=str(RUN/'STATUS.json'),dispatch=str(DISPATCH),
        all_downstream_trainable=True,old_checkpoint_loaded=False))
    print(dict(controller=controller.pid,reporter=reporter.pid,GPU_checks=2,formal_fits=10,
        status='submitted to shared queue; actual starts depend on memory'),flush=True)


if __name__=='__main__':main()

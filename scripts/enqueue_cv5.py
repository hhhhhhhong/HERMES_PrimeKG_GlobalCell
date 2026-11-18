"""Submit the explicitly requested ten GlobalCell seed0 CV1 fits.

Keeps old workers alive. GPU0/1 use the ordinary global limit of two; the
old PathBatch GPU3 exception is not inherited by GlobalCell.
"""
import os,sys,signal,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha,sources
RUNTIME=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT=Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
OLD=ROOT.with_name('HERMES_PrimeKG_PathBatch')
RUN=ROOT/'runs/global_cell_cv5_seed0_20261117'
DISPATCH=ROOT/'artifacts/formal_cv5_seed0_20261117'

def main():
    assert not RUN.exists() and not DISPATCH.exists();RUN.mkdir(parents=True);DISPATCH.mkdir()
    frozen=ROOT/'artifacts/source_freeze_20261117';m=read(frozen/'source_manifest.json')
    assert sources()==m['package'] and all(sha(p)==h for p,h in m['extra'].items())
    assert all(sha(p)==h for p,h in read(frozen/'input_manifest.json').items())
    assert all(read(ROOT/'artifacts'/f'cpu_check_{ds}/CHECKS.json')['status']=='passed' for ds in ['ALMANAC','ONEIL'])
    state=read(RUNTIME/'pipeline_progress.json');pid=state['pid'];assert state['status']=='running'
    previous_scope=ROOT/'artifacts/gpu_check_dispatch';previous=read(previous_scope/'LAUNCH.json');assert previous['controller']==pid
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0');assert cmd==previous['command']
    assert cmd[-2:]==['--gpu-process-limit','3=6']
    registry=read(RUNTIME/'auxiliary_jobs.json');write(DISPATCH/'REGISTRY_BEFORE.json',registry);write(DISPATCH/'STATE_BEFORE.json',state)
    jobs=[];checks=[str(ROOT/'artifacts'/f'gpu_check_{ds}/progress.json') for ds in ['ALMANAC','ONEIL']]
    for fold in range(5):
        for ds in ['ALMANAC','ONEIL']:
            requires=checks+([str(RUN/f'ENTRY_{ds}.json')] if fold>0 else [])
            jobs.append(dict(key=f'aux_hermes_global_cell_cv5_{ds}_fold{fold}_20261117',module='train_hermes_global_cell',dataset=ds,
                directory=str(RUN/ds/f'fold{fold}'),result_storage_root=str(ROOT),gpus=[0,1],memory_mib=20000,requires=requires,
                extra_args=['--stage','train','--fold',str(fold),'--expected-source-manifest',str(frozen/'source_manifest.json'),
                    '--expected-input-manifest',str(frozen/'input_manifest.json')]))
    assert not {j['key'] for j in jobs}&{j['key'] for j in registry['jobs']}
    # Existing GPU checks remain highest priority. All other pending work is
    # preserved; the new user-requested suite precedes the older waiting folds.
    newchecks=[j for j in registry['jobs'] if j['key'].startswith('aux_hermes_global_cell_check_')]
    others=[j for j in registry['jobs'] if j not in newchecks]
    registry['jobs']=newchecks+jobs+others
    write(DISPATCH/'PLAN.json',dict(datasets=['ALMANAC','ONEIL'],folds=list(range(5)),seed=0,formal_fits=10,epochs=3000,
        select_from_epoch=1000,all_downstream_trainable=True,old_checkpoint_loaded=False,internal_validation_only=True,
        test_evaluated=False,formal_fold0_continues_after_first_two_full_updates=True,jobs=jobs))
    shutil.copytree(ROOT/'global_cell',DISPATCH/'source_snapshot/global_cell',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'formal_cv5',DISPATCH/'source_snapshot/formal_cv5',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(frozen/'source_manifest.json',DISPATCH/'source_manifest.json');shutil.copy2(frozen/'input_manifest.json',DISPATCH/'input_manifest.json')
    expiry=read(previous_scope/'EXPIRY_PROCESS.json')['pid']
    def alive(p):
        try:return Path(f'/proc/{p}/stat').read_text().split()[2]!='Z'
        except FileNotFoundError:return False
    for cpu in [pid,expiry]:
        if alive(cpu):os.kill(cpu,signal.SIGTERM)
    for _ in range(100):
        if not any(alive(p) for p in [pid,expiry]):break
        time.sleep(.1)
    else:raise RuntimeError('CPU dispatcher/observer did not exit')
    write(RUNTIME/'auxiliary_jobs.json',registry)
    insert=cmd.index('--gpu-process-limit');newcmd=cmd[:insert]+[j['key'] for j in jobs]+cmd[insert:]
    default=newcmd[:]
    for _ in range(default.count('--gpu-process-limit')):
        i=default.index('--gpu-process-limit');del default[i:i+2]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(RUNTIME/'tmp'),HF_HOME=str(RUNTIME/'cache/huggingface'),TORCH_HOME=str(RUNTIME/'cache/torch'),XDG_CACHE_HOME=str(RUNTIME/'cache/xdg'),
        WANDB_DIR=str(RUNTIME/'wandb'),WANDB_CACHE_DIR=str(RUNTIME/'wandb_cache'),WANDB_DATA_DIR=str(RUNTIME/'wandb_data'),WANDB_CONFIG_DIR=str(RUNTIME/'wandb_config'))
    with (DISPATCH/'controller.log').open('a') as f:child=subprocess.Popen(newcmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(DISPATCH/'LAUNCH.json',dict(controller=child.pid,previous_controller=pid,command=newcmd,default_command=default,overrides={'3':6},
        exception_scope='old PathBatch CV5 only; GlobalCell uses GPU0/1 limit2',user_instruction='然后直接开启五折训练',jobs=jobs,
        GPU_workers_signalled=False,previous_workers=state['active'],new_GlobalCell_formal_fits=10))
    write(previous_scope/'CONTROLLER_ADOPTED.json',dict(controller=child.pid,record=str(DISPATCH),expiry_preserved=True))
    with (DISPATCH/'expiry.log').open('a') as f:
        observer=subprocess.Popen([sys.executable,'-B','-u',str(OLD/'scripts/expire_cv5_parallel_override.py'),'--record',str(DISPATCH)],cwd=OLD,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(DISPATCH/'EXPIRY_PROCESS.json',dict(pid=observer.pid,cpu_only=True,expires_old_PathBatch_exception=True))
    with (DISPATCH/'report.log').open('a') as f:
        reporter=subprocess.Popen([sys.executable,'-B','-u','-m','formal_cv5.report','--watch'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    write(DISPATCH/'REPORTER.json',dict(pid=reporter.pid,cpu_only=True))
    print(dict(controller=child.pid,formal_fits=10,existing_GPU_workers_preserved=True,reporter=reporter.pid,old_scope_expiry=observer.pid),flush=True)
if __name__=='__main__':main()

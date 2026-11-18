"""Explicitly approved GPU1 limit4 for ONEIL0 + ALMANAC1; no other fits."""
import os,sys,signal,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha,sources
RUNTIME=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT=Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
DISPATCH=ROOT/'artifacts/formal_cv5_seed0_20261117'
TARGETS=['aux_hermes_global_cell_cv5_ONEIL_fold0_20261117','aux_hermes_global_cell_cv5_ALMANAC_fold1_20261117']

def main():
    record=DISPATCH/'gpu1_two_more';assert record.exists() and not (record/'LAUNCH.json').exists()
    m=read(ROOT/'artifacts/source_freeze_20261117/source_manifest.json');assert sources()==m['package'] and all(sha(f)==h for f,h in m['extra'].items())
    assert all(sha(f)==h for f,h in read(ROOT/'artifacts/source_freeze_20261117/input_manifest.json').items())
    state=read(RUNTIME/'pipeline_progress.json');pid=state['pid'];previous=read(DISPATCH/'LAUNCH.json')
    assert state['status']=='running' and previous['controller']==pid
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0');assert cmd==previous['command']
    assert cmd[-2:]==['--gpu-process-limit','3=6']
    registry=read(RUNTIME/'auxiliary_jobs.json');write(record/'REGISTRY_BEFORE.json',registry);write(record/'STATE_BEFORE.json',state)
    pending={j['key'] for j in state['pending']};assert set(TARGETS)<=pending
    for j in registry['jobs']:
        if j['key'] in TARGETS:
            assert not (Path(j['directory'])/'progress.json').exists()
            assert all(read(Path(p)).get('status')=='complete' for p in j.get('requires',[]))
            j['gpus']=[1]
        elif j['key'] in pending:
            # Only the approved two jobs can consume slots3/4 on GPU1.
            j.setdefault('gpu_process_limits',{})['1']=2
    expiry=read(DISPATCH/'EXPIRY_PROCESS.json')['pid']
    def alive(p):
        try:return Path(f'/proc/{p}/stat').read_text().split()[2]!='Z'
        except FileNotFoundError:return False
    for cpu in [pid,expiry]:
        if alive(cpu):os.kill(cpu,signal.SIGTERM)
    for _ in range(100):
        if not any(alive(cpu) for cpu in [pid,expiry]):break
        time.sleep(.1)
    else:raise RuntimeError('CPU dispatcher/observer did not exit')
    write(RUNTIME/'auxiliary_jobs.json',registry)
    newcmd=cmd+['--gpu-process-limit','1=4']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(RUNTIME/'tmp'),HF_HOME=str(RUNTIME/'cache/huggingface'),TORCH_HOME=str(RUNTIME/'cache/torch'),XDG_CACHE_HOME=str(RUNTIME/'cache/xdg'),
        WANDB_DIR=str(RUNTIME/'wandb'),WANDB_CACHE_DIR=str(RUNTIME/'wandb_cache'),WANDB_DATA_DIR=str(RUNTIME/'wandb_data'),WANDB_CONFIG_DIR=str(RUNTIME/'wandb_config'))
    with (record/'controller.log').open('a') as f:child=subprocess.Popen(newcmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(record/'LAUNCH.json',dict(controller=child.pid,previous_controller=pid,command=newcmd,overrides={'1':4,'3':6},
        user_authorization='1卡临时放宽到4，启动这两折',approved_existing_keys=TARGETS,additional_fits=0,
        GPU_workers_signalled=False,previous_workers=state['active'],new_GlobalCell_supplement_scope_GPU1_only=True,model_sources_inputs_unchanged=True))
    oldrun=ROOT.with_name('HERMES_PrimeKG_PathBatch')/'runs/formal_path_batch_cv5_seed0_20261117'
    scopes={'1':dict(limit=4,scope='only newly approved ONEIL0 + ALMANAC1',completion_files=[
        str(ROOT/'runs/global_cell_cv5_seed0_20261117/ONEIL/fold0/progress.json'),str(ROOT/'runs/global_cell_cv5_seed0_20261117/ALMANAC/fold1/progress.json')]),
      '3':dict(limit=6,scope='previously running old PathBatch only; unchanged',completion_files=[str(oldrun/d/f'fold{f}/progress.json') for d in ['ALMANAC','ONEIL'] for f in range(5)])}
    write(record/'POLICY_STATE.json',dict(controller=child.pid,command=newcmd,active_scopes=scopes))
    with (record/'expiry.log').open('a') as f:
        watcher=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/expire_concurrency_scopes.py'),'--record',str(record)],cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(record/'EXPIRY_PROCESS.json',dict(pid=watcher.pid,cpu_only=True,independent_GPU1_GPU3_expiry=True))
    write(DISPATCH/'CONTROLLER_ADOPTED.json',dict(controller=child.pid,scope=str(record),previous_expiry_stopped=expiry))
    print(dict(controller=child.pid,newly_released=TARGETS,GPU1_limit=4,expiry=watcher.pid),flush=True)
if __name__=='__main__':main()

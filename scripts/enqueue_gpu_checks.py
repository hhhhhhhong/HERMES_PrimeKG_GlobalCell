"""Only two correctness checks; never borrow PathBatch's GPU3 exception."""
import os,sys,signal,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sources,sha
RUNTIME=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT=Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
OLD=ROOT.with_name('HERMES_PrimeKG_PathBatch')

def main():
 assert all(read(ROOT/'artifacts'/f'cpu_check_{ds}/CHECKS.json')['status']=='passed' for ds in ['ALMANAC','ONEIL'])
 frozen=ROOT/'artifacts/source_freeze_20261117';manifest=read(frozen/'source_manifest.json');assert sources()==manifest['package']
 assert all(sha(f)==h for f,h in manifest['extra'].items())
 record=ROOT/'artifacts/gpu_check_dispatch';assert not record.exists();record.mkdir()
 state=read(RUNTIME/'pipeline_progress.json');pid=state['pid'];assert state['status']=='running'
 previous_scope=OLD/'artifacts/formal_path_batch_cv5_seed0_20261117/gpu3_all_oneil'
 previous=read(previous_scope/'LAUNCH.json');assert previous['controller']==pid
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0');assert cmd==previous['command']
 registry=read(RUNTIME/'auxiliary_jobs.json');write(record/'REGISTRY_BEFORE.json',registry);write(record/'STATE_BEFORE.json',state)
 shared=['--stage','check','--fold','0','--expected-source-manifest',str(frozen/'source_manifest.json'),'--expected-input-manifest',str(frozen/'input_manifest.json')]
 jobs=[dict(key=f'aux_hermes_global_cell_check_{ds}_20261117',module='train_hermes_global_cell',dataset=ds,
  directory=str(ROOT/'artifacts'/f'gpu_check_{ds}'),result_storage_root=str(ROOT),gpus=[0,1],memory_mib=20000,extra_args=shared) for ds in ['ALMANAC','ONEIL']]
 assert not {j['key'] for j in jobs}&{j['key'] for j in registry['jobs']}
 # Priority only affects the next free ordinary slot; no training is stopped.
 registry['jobs']=jobs+registry['jobs']
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
 with (record/'controller.log').open('a') as f:child=subprocess.Popen(newcmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 # The scope remains the OLD PathBatch CV5 exception. Its expiry observer is
 # restarted solely so a CPU dispatcher restart cannot accidentally prolong it.
 write(record/'LAUNCH.json',dict(controller=child.pid,previous_controller=pid,command=newcmd,default_command=default,overrides={'3':6},
  exception_scope='old PathBatch CV5 only; new GlobalCell checks restricted to GPU0/1 at limit2',jobs=jobs,GPU_workers_signalled=False,
  previous_workers=state['active'],new_global_cell_formal_fits=0))
 write(previous_scope/'CONTROLLER_ADOPTED.json',dict(controller=child.pid,record=str(record),expiry_preserved=True))
 with (record/'expiry.log').open('a') as f:
  watcher=subprocess.Popen([sys.executable,'-B','-u',str(OLD/'scripts/expire_cv5_parallel_override.py'),'--record',str(record)],cwd=OLD,
   stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 write(record/'EXPIRY_PROCESS.json',dict(pid=watcher.pid,cpu_only=True,expires_old_PathBatch_exception=True))
 print(dict(controller=child.pid,GPU_checks=2,new_formal_fits=0,gpus=[0,1],old_PathBatch_expiry=watcher.pid),flush=True)
if __name__=='__main__':main()

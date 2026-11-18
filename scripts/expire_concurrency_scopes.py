"""Expire GPU1's two approved folds and the old GPU3 suite independently.

CPU policy observer only. It never signals a training worker or launches a
fit, and exits if another session takes ownership of the dispatcher.
"""
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_cell.common import read,write
RUNTIME=Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT=Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')

def remove_limits(command, expired):
    result=[];i=0
    while i<len(command):
        if command[i]=='--gpu-process-limit':
            gpu=int(command[i+1].split('=')[0])
            if str(gpu) not in expired:result.extend(command[i:i+2])
            i+=2
        else:result.append(command[i]);i+=1
    return result

def ready(scope):
    return all(p.exists() and read(p).get('status')=='complete' for p in map(Path,scope['completion_files']))

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);args=p.parse_args();record=args.record
    policy=read(record/'POLICY_STATE.json')
    while policy['active_scopes']:
        expired={gpu for gpu,scope in policy['active_scopes'].items() if ready(scope)}
        if not expired:time.sleep(15);continue
        state=read(RUNTIME/'pipeline_progress.json');pid=policy['controller']
        if state.get('pid')!=pid:
            write(record/'WATCH_STATUS.json',dict(status='superseded_by_later_controller',no_GPU_workers_signalled=True));return
        newcmd=remove_limits(policy['command'],expired);newpid=pid
        if alive(pid):
            current=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
            if current!=policy['command']:
                write(record/'WATCH_STATUS.json',dict(status='command_changed',no_GPU_workers_signalled=True));return
            os.kill(pid,signal.SIGTERM)
            for _ in range(500):
                if not alive(pid):break
                time.sleep(.1)
            else:raise RuntimeError('CPU dispatcher did not exit')
            with (record/'expiry_controller.log').open('a') as f:
                child=subprocess.Popen(newcmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
            newpid=child.pid
        else:
            resource=read(RUNTIME/'resource_policy.json')
            for gpu in expired:resource['gpu_process_limit_overrides'].pop(gpu,None)
            write(RUNTIME/'resource_policy.json',resource)
        for gpu in expired:
            write(record/f'GPU{gpu}_EXPIRED.json',dict(status='default_limit2_restored',scope=policy['active_scopes'].pop(gpu),
                controller=newpid,GPU_workers_signalled=False))
        policy.update(controller=newpid,command=newcmd);write(record/'POLICY_STATE.json',policy)
    write(record/'WATCH_STATUS.json',dict(status='all_scopes_expired',GPU_workers_signalled=False))
if __name__=='__main__':main()

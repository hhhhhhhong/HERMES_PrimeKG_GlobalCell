"""Submit one optimizer diagnostic through the shared queue; preserve CV5."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

DISPATCH = ROOT / 'artifacts/knowledge_decay_ALMANAC_fold0_seed0_20261119'
RUN = ROOT / 'runs/knowledge_decay_ALMANAC_fold0_seed0_20261119'
RUNTIME = ROOT.parent / 'runtime/paper_routes_20261009'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'


def alive(pid):
    try:
        return Path('/proc/%s/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    assert not (DISPATCH / 'LAUNCH.json').exists()
    assert read(DISPATCH / 'UNIT_CHECKS.json')['status'] == 'complete'
    assert read(DISPATCH / 'cpu_check/progress.json')['status'] == 'complete'
    assert read(DISPATCH / 'cpu_check/SELECTED_DIAGNOSTICS.json')['status'] == 'passed'
    manifest = read(DISPATCH / 'frozen/source_manifest.json')
    assert all(sha(ROOT / p) == h for p, h in manifest['package'].items())
    assert all(sha(Path(p)) == h for p, h in manifest['extra'].items())
    state = read(RUNTIME / 'pipeline_progress.json')
    assert state['status'] == 'running' and alive(state['pid'])
    previous_command = Path('/proc/%s/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in previous_command
    assert '--runtime-directory' in previous_command and str(RUNTIME) in previous_command
    active = [j for j in state['active'] if alive(j['pid'])]
    registry = read(RUNTIME / 'auxiliary_jobs.json')
    common = dict(module='train_hermes_knowledge_decay', result_storage_root=str(ROOT),
        dataset='ALMANAC', gpus=[1, 0, 2, 3], gpu_process_limits={str(i): 1 for i in range(4)},
        memory_mib=16384)
    args = ['--decay-mode', 'knowledge_no_decay', '--fold', '0',
        '--expected-source-manifest', str(DISPATCH / 'frozen/source_manifest.json'),
        '--expected-input-manifest', str(DISPATCH / 'frozen/input_manifest.json')]
    check = dict(common, key='aux_hermes_knowledge_decay_ALMANAC_fold0_check_20261119',
        directory=str(DISPATCH / 'gpu_check'),
        requires=[str(DISPATCH / 'UNIT_CHECKS.json'), str(DISPATCH / 'cpu_check/progress.json')],
        extra_args=['--stage', 'check'] + args)
    fit = dict(common, key='aux_hermes_knowledge_decay_ALMANAC_fold0_fit_20261119',
        directory=str(RUN / 'ALMANAC/fold0'), requires=[str(DISPATCH / 'gpu_check/progress.json')],
        extra_args=['--stage', 'train'] + args)
    assert not {check['key'], fit['key']} & {j['key'] for j in registry['jobs']}
    write(DISPATCH / 'REGISTRY_BEFORE.json', registry)
    write(DISPATCH / 'STATE_BEFORE.json', state)
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']):
            break
        time.sleep(.1)
    assert not alive(state['pid'])
    assert all(alive(j['pid']) for j in active), 'Preserve all active GPU workers'
    assert read(RUNTIME / 'auxiliary_jobs.json') == registry
    registry['jobs'] = [check, fit] + registry['jobs']
    write(RUNTIME / 'auxiliary_jobs.json', registry)
    # Prioritize this single-fold correctness experiment, then keep every old
    # selected job. This is the same queue and global user process accounting.
    position = previous_command.index('--auxiliary-keys')
    command = previous_command[:position + 1] + [check['key'], fit['key']] + previous_command[position + 1:]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    for key, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
        XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
        WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        env[key] = str(RUNTIME / suffix)
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'reporter.log').open('a') as stream:
        observer = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/report_knowledge_decay.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    scope = ROOT / 'artifacts/formal_cv5_seed0_20261117/gpu1_two_more/POLICY_STATE.json'
    policy = read(scope)
    assert not policy['active_scopes']
    policy.update(controller=controller.pid, command=command)
    write(scope, policy)
    record = dict(status='submitted', controller=controller.pid, command=command, reporter=observer.pid,
        check=check, fit=fit, preserved_gpu_workers=active, GPU_workers_signalled=False,
        seed=0, fold=0, formal_fits=1, architecture_changed=False, old_checkpoint_loaded=False,
        old_CV5_preserved=True, global_user_process_limit_per_GPU=2, job_process_ceiling=1)
    write(DISPATCH / 'LAUNCH.json', record)
    write(ROOT / 'OPTIMIZER_DIAGNOSTIC_RUN.json', dict(run_directory=str(RUN), dispatch=str(DISPATCH),
        variant='knowledge-only weight_decay=0', seed=0, dataset='ALMANAC', fold=0, formal_fits=1))
    print(dict(controller=controller.pid, reporter=observer.pid, preserved_gpu_pids=[j['pid'] for j in active],
        formal_fits=1, status='submitted; GPU availability determines actual start'), flush=True)


if __name__ == '__main__':
    main()

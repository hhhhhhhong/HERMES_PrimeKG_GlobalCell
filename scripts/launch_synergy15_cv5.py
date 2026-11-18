"""Register current in-place source, two GPU checks and ten fresh CV1 fits.

Only the existing shared queue launches GPU workers. Restarting that CPU
controller adopts its existing workers and preserves the older PathBatch
exception, which the new suite cannot use. All output stays on IFS2.
"""
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha, sources

RUNTIME = Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT = Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
RUN = ROOT/'runs/global_cell_synergy15_cv5_seed0_20261117'
DISPATCH = ROOT/'artifacts/synergy15_cv5_seed0_20261117'
FROZEN = DISPATCH/'frozen'
CHECKS = DISPATCH/'gpu_checks'
SCOPE = ROOT/'artifacts/formal_cv5_seed0_20261117/gpu1_two_more'


def argv(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except FileNotFoundError:
        return []


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def stop_cpu(pid):
    if not alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not alive(pid):
            return
        time.sleep(.1)
    raise RuntimeError('CPU process did not exit; no GPU worker signalled')


def main():
    assert not RUN.exists() and not DISPATCH.exists(), 'Use unique run paths; never overwrite old results'
    DISPATCH.mkdir(parents=True)
    FROZEN.mkdir()
    CHECKS.mkdir()
    RUN.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    for key, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
                            XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
                            WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        directory = RUNTIME/suffix
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)

    # Verify the actual current package rather than the archived pre-user-edit checks.
    with (DISPATCH/'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    inputs = read(ROOT/'artifacts/source_freeze_20261117/input_manifest.json')
    assert all(sha(p) == h for p, h in inputs.items()), 'Historical input changed'
    extra_paths = [Path(__file__).resolve(), ROOT/'scripts/report_synergy15_cv5.py', ROOT/'formal_cv5/report.py',
                   ROOT/'scripts/audit_synergy.py', ROOT/'tests/test_synergy.py', ROOT/'tests/test_global_cell.py',
                   PROJECT/'tools/paper_routes/train_hermes_global_cell.py']
    manifest = dict(package=sources(), extra={str(p): sha(p) for p in extra_paths})
    write(FROZEN/'source_manifest.json', manifest)
    write(FROZEN/'input_manifest.json', inputs)
    shutil.copytree(ROOT/'global_cell', FROZEN/'source_snapshot/global_cell', ignore=shutil.ignore_patterns('__pycache__'))
    (FROZEN/'supporting_sources').mkdir()
    for p in extra_paths:
        dest = FROZEN/'supporting_sources'/p.name
        shutil.copy2(p, dest)

    registry_path = RUNTIME/'auxiliary_jobs.json'
    registry = read(registry_path)
    checks = []
    jobs = []
    extra = ['--expected-source-manifest', str(FROZEN/'source_manifest.json'),
             '--expected-input-manifest', str(FROZEN/'input_manifest.json')]
    common = dict(module='train_hermes_global_cell', result_storage_root=str(ROOT),
                  gpus=[2, 0, 1], memory_mib=20000,
                  gpu_process_limits={'0': 2, '1': 2, '2': 2, '3': 2})
    for ds in ['ALMANAC', 'ONEIL']:
        checks.append(dict(common, key=f'aux_hermes_synergy15_gpu_check_{ds}_20261117',
                           dataset=ds, directory=str(CHECKS/ds), requires=[],
                           extra_args=['--stage', 'check', '--fold', '0']+extra))
    prerequisites = [str(CHECKS/ds/'progress.json') for ds in ['ALMANAC', 'ONEIL']]
    for fold in range(5):
        for ds in ['ALMANAC', 'ONEIL']:
            jobs.append(dict(common, key=f'aux_hermes_synergy15_cv5_{ds}_fold{fold}_20261117',
                             dataset=ds, directory=str(RUN/ds/f'fold{fold}'),
                             requires=prerequisites+([str(RUN/f'ENTRY_{ds}.json')] if fold else []),
                             extra_args=['--stage', 'train', '--fold', str(fold)]+extra))
    assert not {j['key'] for j in checks+jobs} & {j['key'] for j in registry['jobs']}
    write(DISPATCH/'PLAN.json', dict(seed=0, datasets=['ALMANAC', 'ONEIL'], folds=list(range(5)),
          formal_fits=10, checks=checks, jobs=jobs, epochs=3000, freeze=False, checkpoint_warm_start=False,
          exact_current_user_edits_preserved=True, tests='internal validation only',
          first_two_full_fit_updates_required_before_other_folds=True))

    # Quiesce only the single shared CPU controller and its CPU expiry observer.
    state = read(RUNTIME/'pipeline_progress.json')
    controller = state['pid']
    cmd = argv(controller)
    assert 'tools.paper_routes.queue' in cmd and str(RUNTIME) in cmd
    policy = read(SCOPE/'POLICY_STATE.json')
    assert policy['controller'] == controller and policy['command'] == cmd
    assert set(policy['active_scopes']) <= {'3'}
    write(DISPATCH/'REGISTRY_BEFORE.json', registry)
    write(DISPATCH/'STATE_BEFORE.json', state)
    write(DISPATCH/'POLICY_BEFORE.json', policy)
    expiry = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            a = argv(int(p.name))
            if str(ROOT/'scripts/expire_concurrency_scopes.py') in a and str(SCOPE) in a:
                expiry.append(int(p.name))
    for pid in expiry:
        stop_cpu(pid)
    # Re-read state after stopping the policy observer, in case it just expired a scope.
    state = read(RUNTIME/'pipeline_progress.json')
    assert state['pid'] == controller and argv(controller) == cmd
    stop_cpu(controller)
    current = read(registry_path)
    assert current == registry, 'Registry edited concurrently; do not overwrite'
    registry['jobs'] = checks+jobs+registry['jobs']
    write(registry_path, registry)

    newcmd = list(cmd)
    i = newcmd.index('--auxiliary-keys')+1
    newcmd[i:i] = [j['key'] for j in checks+jobs]
    i = newcmd.index('--gpus')+1
    if '2' not in newcmd[i:newcmd.index('--qa-gpu')]:
        newcmd.insert(i, '2')
    with (DISPATCH/'controller.log').open('a') as log:
        child = subprocess.Popen(newcmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    policy.update(controller=child.pid, command=newcmd)
    write(SCOPE/'POLICY_STATE.json', policy)
    watcher = None
    if policy['active_scopes']:
        with (DISPATCH/'expiry.log').open('a') as log:
            watcher = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT/'scripts/expire_concurrency_scopes.py'),
                                        '--record', str(SCOPE)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env).pid
    report_cmd = [sys.executable, '-B', '-u', str(ROOT/'scripts/report_synergy15_cv5.py'),
                  '--run', str(RUN), '--frozen', str(FROZEN), '--checks', str(CHECKS), '--watch']
    with (DISPATCH/'report.log').open('a') as log:
        reporter = subprocess.Popen(report_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=dict(env, CUDA_VISIBLE_DEVICES='')).pid
    write(DISPATCH/'LAUNCH.json', dict(controller=child.pid, previous_controller=controller,
          command=newcmd, report_command=report_cmd, reporter=reporter, scope_observer=watcher,
          source_manifest=str(FROZEN/'source_manifest.json'), run=str(RUN),
          GPU_workers_signalled=False, previous_workers=state.get('active', []),
          new_suite_process_limit=2, new_suite_gpus=[2, 0, 1], old_PathBatch_exception_preserved=True))
    print(dict(controller=child.pid, reporter=reporter, GPU_checks=2, formal_fits=10,
               new_suite_limit_per_GPU=2, run=str(RUN)), flush=True)


if __name__ == '__main__':
    main()

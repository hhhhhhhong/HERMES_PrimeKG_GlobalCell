"""Register two checks and two seed0/fold0 fits in the existing shared queue."""
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment, RUNTIME
from global_cell.common import read, write, sha

DISPATCH = ROOT / 'artifacts/transformer_repair_20261120'
RUN = ROOT / 'runs/transformer_repair_fold0_seed0_20261120'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'
BRIDGE = PROJECT / 'tools/paper_routes/train_hermes_transformer_repair.py'


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    environment()
    assert not (DISPATCH / 'LAUNCH.json').exists(), 'Do not submit duplicate pilots'
    previous = read(ROOT / 'artifacts/dual_branch_transformer_cv5_seed0_20261120/frozen/manifest.json')
    assert all(sha(p) == h for p, h in previous['source'].items())
    assert all(sha(p) == h for p, h in previous['inputs'].items())
    source = dict(previous['source'])
    extra = [ROOT / ('dual_branch/' + n) for n in ['knowledge_selection.py', 'stable_prediction.py', 'stable_training.py', 'repair_cli.py']]
    extra += [ROOT / ('tests/' + n) for n in ['test_knowledge_selection.py', 'test_stable_prediction.py']]
    extra += [Path(__file__).resolve(), ROOT / 'scripts/report_transformer_repair.py', BRIDGE]
    source.update({str(p): sha(p) for p in extra})
    manifest = DISPATCH / 'frozen/manifest.json'
    write(manifest, dict(source=source, inputs=previous['inputs'],
        version='bounded_prediction_repair_20261120', original_frozen_sources_unchanged=True))
    for name in source:
        path = Path(name)
        rel = str(path.relative_to(ROOT)) if ROOT in path.parents else 'external/' + str(path).lstrip('/')
        target = DISPATCH / 'frozen/source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    modes = ['full', 'selected']
    common = dict(module='train_hermes_transformer_repair', dataset='ALMANAC',
        result_storage_root=str(ROOT), gpus=[0, 1, 2, 3])
    jobs = []
    for mode in modes:
        jobs.append(dict(common, key='aux_hermes_transformer_repair_%s_check_20261120' % mode,
            directory=str(DISPATCH / ('gpu_' + mode)), memory_mib=10240 if mode == 'full' else 8192,
            gpu_process_limits={str(i): 2 for i in range(4)},
            requires=[str(DISPATCH / 'CPU_READY.json')],
            extra_args=['--stage', 'check', '--check-rows', '1024', '--fold', '0',
                '--knowledge-mode', mode, '--expected-manifest', str(manifest)]))
    for mode in modes:
        jobs.append(dict(common, key='aux_hermes_transformer_repair_%s_fold0_20261120' % mode,
            directory=str(RUN / 'ALMANAC' / mode / 'fold0'), memory_mib=12288 if mode == 'full' else 8192,
            gpu_process_limits={str(i): 1 for i in range(4)},
            requires=[str(DISPATCH / 'GPU_READY.json')],
            extra_args=['--stage', 'train', '--fold', '0', '--knowledge-mode', mode,
                '--expected-manifest', str(manifest)]))
    state = read(RUNTIME / 'pipeline_progress.json')
    assert state['status'] == 'running' and alive(state['pid'])
    policy = read(RUNTIME / 'resource_policy.json')
    assert not policy.get('gpu_process_limit_overrides')
    command = Path('/proc/%d/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in command and '--auxiliary-keys' in command
    assert '--runtime-directory' in command and str(RUNTIME) in command and '--gpu-process-limit' not in command
    before = read(RUNTIME / 'auxiliary_jobs.json')
    assert not {j['key'] for j in before['jobs']} & {j['key'] for j in jobs}
    write(DISPATCH / 'REGISTRY_BEFORE.json', before)
    write(DISPATCH / 'STATE_BEFORE.json', state)
    old_workers = {j['pid']: j for j in state['active'] if alive(j['pid'])}
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']):
            break
        time.sleep(.05)
    assert not alive(state['pid'])
    registry = read(RUNTIME / 'auxiliary_jobs.json')
    assert registry == before
    # Give the new diagnostic task priority over waiting fits; keep every old
    # job, its dependencies, worker and relative order unchanged.
    registry['jobs'] = jobs + registry['jobs']
    registry['transformer_repair_authorization_20261120'] = dict(user_request='你帮我修复一下吧',
        scope='ALMANAC seed0/fold0 full vs selected knowledge, 2 fits only',
        old_training_preserved=True, no_automatic_cv5_or_extra_seed=True)
    write(RUNTIME / 'auxiliary_jobs.json', registry)
    command += [j['key'] for j in jobs]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'report.log').open('a') as stream:
        reporter = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/report_transformer_repair.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(DISPATCH / 'LAUNCH.json', dict(controller=controller.pid, reporter=reporter.pid,
        command=command, jobs=jobs, previous_controller=state['pid'],
        old_workers=old_workers, old_GPU_workers_signalled=False, formal_fits=2,
        dataset='ALMANAC', seed=0, fold=0, global_user_process_limit_per_GPU=2))
    write(ROOT / 'REPAIR_RUN.json', dict(status='submitted', run_directory=str(RUN),
        dispatch=str(DISPATCH), status_file=str(RUN / 'STATUS.json'), formal_fits=2,
        old_CURRENT_RUN_unchanged=True, all_outputs_on_IFS2=True))
    for pid, job in old_workers.items():
        assert alive(pid) or read(Path(job['directory']) / 'progress.json').get('status') == 'complete'
    print(dict(controller=controller.pid, reporter=reporter.pid, GPU_checks=2,
        formal_fits=2, output=str(RUN)), flush=True)


if __name__ == '__main__':
    main()

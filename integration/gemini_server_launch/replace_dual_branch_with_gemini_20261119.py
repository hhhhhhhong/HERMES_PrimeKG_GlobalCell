"""Stop only the replaced suite; stage the new CV5 in the same GPU queue."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment, RUNTIME
from global_cell.common import read, write, sha
environment()
OLD = ROOT / 'artifacts/dual_branch_cv5_seed0_20261119'
OLD_RUN = ROOT / 'runs/dual_branch_cv5_seed0_20261119'
NEW = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


assert not (NEW / 'REPLACEMENT.json').exists()
state = read(RUNTIME / 'pipeline_progress.json')
launch = read(OLD / 'LAUNCH.json')
assert state['status'] == 'running' and state['pid'] == launch['controller']
registry = read(RUNTIME / 'auxiliary_jobs.json')
old_jobs = [job for job in registry['jobs'] if job['module'] == 'train_hermes_dual_branch']
assert len(old_jobs) == 12
old_keys = {job['key'] for job in old_jobs}
assert all(str(Path(job['directory'])).startswith(str(OLD_RUN) + '/') or
    str(Path(job['directory'])).startswith(str(OLD) + '/') for job in old_jobs)
write(NEW / 'REGISTRY_BEFORE.json', registry)
write(NEW / 'STATE_BEFORE.json', state)
write(NEW / 'OLD_CURRENT_RUN.json', read(ROOT / 'CURRENT_RUN.json'))
backup = NEW / 'before_overwrite'
sources = list((ROOT / 'dual_branch').glob('*.py')) + [ROOT / 'tests/test_dual_branch.py']
for path in sources:
    destination = backup / path.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
write(backup / 'source_sha256.json', {str(path): sha(path) for path in sources})
command = Path('/proc/%d/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
assert 'tools.paper_routes.queue' in command and '--gpu-process-limit' not in command
assert '--runtime-directory' in command and str(RUNTIME) in command
for pid, expected in [(state['pid'], 'tools.paper_routes.queue'), (launch['reporter'], 'report_dual_branch_per_dataset.py')]:
    if alive(pid):
        assert expected in Path('/proc/%d/cmdline' % pid).read_bytes().decode()
        os.kill(pid, signal.SIGTERM)
for _ in range(100):
    if not alive(state['pid']) and not alive(launch['reporter']):
        break
    time.sleep(.05)
assert not alive(state['pid']) and not alive(launch['reporter'])
stopped = []
for process_file in (RUNTIME / 'processes').glob('aux_hermes_dual_branch_cv5_*.json'):
    process = read(process_file)
    if process['key'] in old_keys and alive(process['pid']):
        argv = Path('/proc/%d/cmdline' % process['pid']).read_bytes().decode()
        assert 'tools.paper_routes.train_hermes_dual_branch' in argv
        assert process['directory'] in argv
        os.kill(process['pid'], signal.SIGTERM)
        stopped.append(process)
for _ in range(100):
    if all(not alive(process['pid']) for process in stopped):
        break
    time.sleep(.05)
assert all(not alive(process['pid']) for process in stopped)
for job in old_jobs:
    path = Path(job['directory']) / 'progress.json'
    if path.exists():
        progress = read(path)
        if progress.get('status') != 'complete':
            progress.update(status='cancelled', stopped_by_user=True,
                reason='User requested overwrite with corrected Gemini model and restart from scratch')
            write(path, progress)
registry = read(RUNTIME / 'auxiliary_jobs.json')
assert registry == read(NEW / 'REGISTRY_BEFORE.json')
registry['jobs'] = [job for job in registry['jobs'] if job['key'] not in old_keys]
registry.setdefault('shelved_experiments', []).append(dict(method='replaced first dual-branch version',
    reason='User explicitly requested overwrite with corrected Gemini code and rerun',
    jobs=old_jobs, old_run=str(OLD_RUN), automatic_resume_allowed=False))
common = dict(module='train_hermes_dual_branch', result_storage_root=str(ROOT),
    gpus=[1, 2, 3, 0], gpu_process_limits={str(i): 2 for i in range(4)})
extra = ['--expected-manifest', str(NEW / 'frozen/manifest.json')]
checks = [dict(common, key='aux_hermes_dual_branch_gemini_%s_check_20261119' % dataset,
    dataset=dataset, directory=str(NEW / ('gpu_check_' + dataset)),
    memory_mib=8192 if dataset == 'ALMANAC' else 5120,
    requires=[str(NEW / ('cpu_check_' + dataset) / 'progress.json')],
    extra_args=['--stage', 'check', '--fold', '0', '--check-rows', '1024'] + extra)
    for dataset in ['ALMANAC', 'ONEIL']]
fits = [dict(common, key='aux_hermes_dual_branch_gemini_%s_fold%d_20261119' % (dataset, fold),
    dataset=dataset, directory=str(RUN / dataset / ('fold%d' % fold)),
    memory_mib=12288 if dataset == 'ALMANAC' else 10240,
    requires=[str(NEW / ('gpu_check_' + dataset) / 'progress.json'), str(NEW / ('MEMORY_READY_' + dataset + '.json'))] +
        ([] if fold == 0 else [str(RUN / ('ENTRY_' + dataset + '.json'))]),
    extra_args=['--stage', 'train', '--fold', str(fold)] + extra)
    for fold in range(5) for dataset in ['ALMANAC', 'ONEIL']]
registry['jobs'] += checks + fits
write(RUNTIME / 'auxiliary_jobs.json', registry)
command = [argument for argument in command if argument not in old_keys] + [job['key'] for job in checks + fits]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
with (NEW / 'controller.log').open('a') as stream:
    controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
for job in state.get('active', []):
    if job['key'] not in old_keys:
        progress = read(Path(job['directory']) / 'progress.json')
        assert alive(job['pid']) or progress['status'] == 'complete'
write(OLD / 'STOPPED_BY_REPLACEMENT.json', dict(status='stopped_by_user', old_results_preserved=True,
    new_run=str(RUN), GPU_workers_stopped=[process['pid'] for process in stopped], automatic_resume_allowed=False))
write(OLD_RUN / 'STATUS.json', dict(status='stopped_by_user', replacement_run=str(RUN),
    old_results_preserved=True, no_complete_fivefold_claim=True))
write(NEW / 'REPLACEMENT.json', dict(status='staged_waiting_checks', controller=controller.pid, command=command,
    checks=checks, jobs=fits, stopped_GPU_workers=stopped,
    previous_other_workers=[job for job in state.get('active', []) if job['key'] not in old_keys],
    old_results_preserved=True, only_old_dual_branch_stopped=True,
    all_new_fits_from_scratch=True, old_checkpoint_loaded=False))
write(ROOT / 'CURRENT_RUN.json', dict(variant='Corrected Gemini dual branch', status='preparing_checks',
    datasets=['ALMANAC', 'ONEIL'], folds_per_dataset=5, seed=0, epochs=3000,
    run_directory=str(RUN), dispatch=str(NEW), status_file=str(RUN / 'STATUS.json'),
    all_downstream_trainable=True, old_checkpoint_loaded=False, previous_run=str(OLD_RUN)))
print(dict(controller=controller.pid, stopped_only_dual_branch=[process['pid'] for process in stopped],
    new_GPU_checks=2, new_formal_fits=10, status='awaiting_new_CPU_and_GPU_checks'), flush=True)

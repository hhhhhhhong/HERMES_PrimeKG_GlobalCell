"""Recover an allocation failure without changing the frozen training recipe.

Only the existing shared queue starts GPU workers. Failed outputs are retained,
and a canonical symlink lets the unchanged observer read the technical retry.
"""
import copy
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment, RUNTIME
from global_cell.common import read, write, sha

environment()
DISPATCH = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'
OLD_KEY = 'aux_hermes_dual_branch_gemini_ALMANAC_fold1_20261119'
NEW_KEY = 'aux_hermes_dual_branch_gemini_ALMANAC_fold1_memory_retry_20261120'


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


assert not (DISPATCH / 'MEMORY_RECOVERY_20261120.json').exists()
launch = read(DISPATCH / 'LAUNCH.json')
state = read(RUNTIME / 'pipeline_progress.json')
assert not alive(launch['controller'])
assert state['status'] == 'failed' and state['failure']['key'] == OLD_KEY
assert alive(launch['reporter'])
manifest = read(DISPATCH / 'frozen/manifest.json')
for kind in ['source', 'inputs']:
    for filename, expected in manifest[kind].items():
        assert sha(filename) == expected, filename

completed = {}
for dataset, fold in [('ALMANAC', 0), ('ONEIL', 0), ('ONEIL', 1)]:
    directory = RUN / dataset / ('fold%d' % fold)
    progress = read(directory / 'progress.json')
    assert progress['status'] == 'complete' and progress['optimizer_updates'] == 3000
    verification = read(directory / 'FINAL_VERIFICATION.json')
    assert verification['status'] == 'passed' and not verification['test_evaluated']
    assert sha(directory / 'selected.pt') == verification['selection']['checkpoint_sha256']
    completed[dataset + '/fold%d' % fold] = verification

failed = RUN / 'ALMANAC/fold1'
failure = read(failed / 'progress.json')
assert failure['status'] == 'failed' and 'out of memory' in failure['error']
assert not alive(failure['pid'])
assert not (failed / 'history.jsonl').exists() and not (failed / 'selected.pt').exists()
assert set(p.name for p in failed.iterdir()) == {'progress.json', 'EXPERIMENT.json', 'OPTIMIZER_GROUPS.json'}
archive = RUN / 'failed_attempts/ALMANAC_fold1_oom_20261120'
retry = RUN / 'technical_retries/ALMANAC/fold1_memory_retry_20261120'
assert not archive.exists() and not retry.exists()

# Hold the same shared lock while editing the canonical registry. Do not replace
# compatibility symlinks under the old, read-only results path.
lock = (RUNTIME / 'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
registry_path = RUNTIME / 'auxiliary_jobs.json'
before = read(registry_path)
registry = copy.deepcopy(before)
suite_keys = {job['key'] for job in registry['jobs'] if 'dual_branch_gemini_' in job['key']}
assert len(suite_keys) == 12 and OLD_KEY in suite_keys and NEW_KEY not in suite_keys
write(DISPATCH / 'REGISTRY_BEFORE_MEMORY_RECOVERY_20261120.json', before)
write(DISPATCH / 'STATE_BEFORE_MEMORY_RECOVERY_20261120.json', state)
write(DISPATCH / 'LAUNCH_BEFORE_MEMORY_RECOVERY_20261120.json', launch)
old_job = next(job for job in registry['jobs'] if job['key'] == OLD_KEY)
old_job = copy.deepcopy(old_job)
shutil.copy2(state['failure']['log'], DISPATCH / 'ALMANAC_fold1_oom_20261120.log')
archive.parent.mkdir(parents=True, exist_ok=True)
failed.rename(archive)
retry.mkdir(parents=True)
failed.symlink_to(retry, target_is_directory=True)

changed = []
for job in registry['jobs']:
    if job['key'] not in suite_keys or '--stage' not in job.get('extra_args', []):
        continue
    if job['extra_args'][job['extra_args'].index('--stage') + 1] != 'train':
        continue
    progress_path = Path(job['directory']) / 'progress.json'
    if progress_path.exists() and read(progress_path).get('status') == 'complete':
        continue
    original = copy.deepcopy(job)
    if job['key'] == OLD_KEY:
        job.update(key=NEW_KEY, directory=str(retry))
    # A suite worker has a whole free user slot on a card. This tightens the
    # existing global two-process ceiling and avoids concurrent initialization.
    job['gpu_process_limits'] = {str(i): 1 for i in range(4)}
    job['memory_mib'] = 12288 if job['dataset'] == 'ALMANAC' else 8192
    changed.append(dict(before=original, after=copy.deepcopy(job)))
assert len(changed) == 7
assert [j for j in registry['jobs'] if j['key'] not in suite_keys | {NEW_KEY}] == [
    j for j in before['jobs'] if j['key'] not in suite_keys]
registry.setdefault('technical_recoveries', []).append(dict(
    reason='ALMANAC fold1 shared-device CUDA OOM before first optimizer update',
    superseded_job=old_job, replacement_key=NEW_KEY, failed_directory=str(archive),
    interrupted_checkpoint_loaded=False, original_recipe_preserved=True))
assert read(registry_path) == before
write(registry_path, registry)
command = [NEW_KEY if arg == OLD_KEY else arg for arg in launch['command']]
assert command.count(NEW_KEY) == 1 and OLD_KEY not in command
assert '--gpu-process-limit' not in command and '--runtime-directory' in command
assert str(RUNTIME) in command and '--auxiliary-only' in command
fcntl.flock(lock, fcntl.LOCK_UN)
lock.close()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
with (DISPATCH / 'controller_memory_recovery_20261120.log').open('a') as stream:
    controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
launch.update(controller=controller.pid, command=command,
              jobs=[job for job in registry['jobs'] if job['key'] in suite_keys | {NEW_KEY}
                    and '--stage' in job.get('extra_args', []) and
                    job['extra_args'][job['extra_args'].index('--stage') + 1] == 'train'])
write(DISPATCH / 'LAUNCH.json', launch)
current = read(ROOT / 'CURRENT_RUN.json')
current.update(status='training_queue_recovered', controller=controller.pid,
               technical_recovery=str(DISPATCH / 'MEMORY_RECOVERY_20261120.json'))
write(ROOT / 'CURRENT_RUN.json', current)
record = dict(status='queue_restarted_waiting_actual_updates',
              timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              controller=controller.pid, reporter=launch['reporter'], command=command,
              completed_unchanged=completed, remaining_unique_fits=7,
              failed_directory=str(archive), retry_directory=str(retry),
              canonical_directory=str(failed), changes=changed,
              source_files_rehashed=len(manifest['source']), input_files_rehashed=len(manifest['inputs']),
              model_and_recipe_unchanged=True, checkpoint_loaded=False,
              global_process_limit=2, pending_job_ceiling=1,
              other_registry_jobs_unchanged=True, all_outputs_on_ifs2=True)
write(DISPATCH / 'MEMORY_RECOVERY_20261120.json', record)
print(dict(controller=controller.pid, preserved_complete_folds=3, remaining_folds=7,
           retry_key=NEW_KEY, source_and_inputs_unchanged=True), flush=True)

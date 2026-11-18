"""Run only current ALMANAC seed0/fold0 after a same-source GPU check."""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sources, sha

RUNTIME = Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
PROJECT = Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
RUN = ROOT/'runs/almanac_fold0_synergy002_drop20_seed0_20261118'
DISPATCH = ROOT/'artifacts/almanac_fold0_synergy002_drop20_seed0_20261118'


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


def main():
    assert not RUN.exists() and not DISPATCH.exists()
    DISPATCH.mkdir(parents=True)
    RUN.mkdir(parents=True)
    frozen = DISPATCH/'frozen'
    frozen.mkdir()
    (frozen/'supporting_sources').mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    for key, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
                           XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
                           WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        path = RUNTIME/suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)

    before = sources()
    with (DISPATCH/'current_variant_tests.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                        '-p', 'test_synergy002_dropout.py', '-v'], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)
    assert before == sources(), 'Model changed during validation; do not mix revisions'
    inputs = {p: h for p, h in read(ROOT/'artifacts/source_freeze_20261117/input_manifest.json').items()
              if '/ONEIL/' not in p and '/ONEIL-COSMIC/' not in p}
    assert all(sha(p) == h for p, h in inputs.items())
    extras = [Path(__file__).resolve(), ROOT/'scripts/report_synergy002_fold0.py',
              ROOT/'tests/test_synergy002_dropout.py', ROOT/'tests/test_global_cell.py',
              PROJECT/'tools/paper_routes/train_hermes_global_cell.py']
    manifest = dict(package=before, extra={str(p): sha(p) for p in extras})
    write(frozen/'source_manifest.json', manifest)
    write(frozen/'input_manifest.json', inputs)
    shutil.copytree(ROOT/'global_cell', frozen/'source_snapshot/global_cell', ignore=shutil.ignore_patterns('__pycache__'))
    for p in extras:
        shutil.copy2(p, frozen/'supporting_sources'/p.name)

    check_dir = DISPATCH/'gpu_check'
    extra = ['--expected-source-manifest', str(frozen/'source_manifest.json'),
             '--expected-input-manifest', str(frozen/'input_manifest.json')]
    common = dict(module='train_hermes_global_cell', dataset='ALMANAC', result_storage_root=str(ROOT),
                  gpus=[1, 0, 3, 2], gpu_process_limits={'0': 2, '1': 2, '2': 2, '3': 2})
    check = dict(common, key='aux_hermes_synergy002_drop20_ALMANAC_fold0_check_20261118',
                 directory=str(check_dir), memory_mib=12000, requires=[],
                 extra_args=['--stage', 'check', '--fold', '0']+extra)
    fit = dict(common, key='aux_hermes_synergy002_drop20_ALMANAC_fold0_fit_20261118',
               directory=str(RUN), memory_mib=16000, requires=[str(check_dir/'progress.json')],
               extra_args=['--stage', 'train', '--fold', '0']+extra)
    state = read(RUNTIME/'pipeline_progress.json')
    assert state['status'] == 'failed' and not alive(state['pid'])
    assert state.get('failure', {}).get('key') == 'aux_hermes_synergy15_cv5_ALMANAC_fold3_20261117'
    previous = read(ROOT/'artifacts/synergy15_cv5_seed0_20261117/LAUNCH.json')
    command = previous['command']
    assert '--runtime-directory' in command and str(RUNTIME) in command
    assert '--gpu-process-limit' not in command
    registry = read(RUNTIME/'auxiliary_jobs.json')
    write(DISPATCH/'REGISTRY_BEFORE.json', registry)
    write(DISPATCH/'STATE_BEFORE.json', state)
    old_waiting = [j for j in registry['jobs'] if j['key'].startswith('aux_hermes_synergy15_cv5_')
                   and j['extra_args'][j['extra_args'].index('--fold')+1] in ['3', '4']]
    assert len(old_waiting) == 4
    old_keys = {j['key'] for j in old_waiting}
    assert not {check['key'], fit['key']} & {j['key'] for j in registry['jobs']}
    # Do not restart old fits that failed before training or promote this pilot.
    registry['jobs'] = [check, fit]+[j for j in registry['jobs'] if j['key'] not in old_keys]
    registry.setdefault('shelved_experiments', []).append(dict(
        directory=str(ROOT/'runs/global_cell_synergy15_cv5_seed0_20261117'),
        reason='User requested only current [0,0,2]+20% dropout ALMANAC fold0; old remaining four folds stay shelved',
        jobs=old_waiting, archive=str(DISPATCH)))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        a = argv(pid)
        if str(ROOT/'scripts/report_synergy15_cv5.py') in a and '--watch' in a:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                if not alive(pid):
                    break
                time.sleep(.1)
            assert not alive(pid)
    assert read(RUNTIME/'auxiliary_jobs.json') == read(DISPATCH/'REGISTRY_BEFORE.json')
    write(RUNTIME/'auxiliary_jobs.json', registry)
    newcmd = [arg for arg in command if arg not in old_keys]
    i = newcmd.index('--auxiliary-keys')+1
    newcmd[i:i] = [check['key'], fit['key']]
    write(DISPATCH/'PLAN.json', dict(dataset='ALMANAC', training_seed=0, folds=[0], formal_fits=1,
          epochs=3000, select_from_epoch=1000, synergy_positions=[0, 0, 2], synergy_message_dropout=0.2,
          dropout_applied='after beta; inverse keep probability 1/0.8; disabled in eval',
          fresh_initialization=True, all_downstream_trainable=True, old_checkpoint_loaded=False,
          user_model_code_unchanged=True, test_evaluated=False, checks=[check], jobs=[fit],
          old_remaining_folds_shelved=sorted(old_keys), max_user_processes_per_GPU=2,
          memory_budget_note='ALMANAC full-graph predecessor GPU check peak 7307 MiB; one-worker check reserve12GiB, fit reserve16GiB'))
    with (DISPATCH/'controller.log').open('a') as log:
        child = subprocess.Popen(newcmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    scope = ROOT/'artifacts/formal_cv5_seed0_20261117/gpu1_two_more/POLICY_STATE.json'
    policy = read(scope)
    assert not policy['active_scopes']
    policy.update(controller=child.pid, command=newcmd)
    write(scope, policy)
    report_cmd = [sys.executable, '-B', '-u', str(ROOT/'scripts/report_synergy002_fold0.py'),
                  '--run', str(RUN), '--check', str(check_dir), '--watch']
    with (DISPATCH/'report.log').open('a') as log:
        reporter = subprocess.Popen(report_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=dict(env, CUDA_VISIBLE_DEVICES='')).pid
    write(DISPATCH/'LAUNCH.json', dict(controller=child.pid, command=newcmd, reporter=reporter,
          run=str(RUN), source_manifest=str(frozen/'source_manifest.json'), check=check,
          fit=fit, old_GPU_workers_signalled=False, other_new_folds=0))
    print(dict(controller=child.pid, reporter=reporter, GPU_checks=1, formal_fits=1,
               dataset='ALMANAC', seed=0, fold=0, run=str(RUN)), flush=True)


if __name__ == '__main__':
    main()

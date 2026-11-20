"""Freeze/submit the authorized ONEIL optimizer-only CV5 to the shared queue."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha, sources

DISPATCH = ROOT / 'artifacts/knowledge_decay_ONEIL_cv5_seed0_20261119'
RUN = ROOT / 'runs/knowledge_decay_ONEIL_cv5_seed0_20261119'
RUNTIME = ROOT.parent / 'runtime/paper_routes_20261009'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'


def alive(pid):
    try:
        return Path('/proc/%s/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def prepare():
    """Reuse audited, unchanged inputs; freeze new adapters without editing old ones."""
    frozen = DISPATCH / 'frozen'
    assert not frozen.exists()
    old = read(ROOT / 'artifacts/zero_gated_hgnn3_cv5_seed0_20261118/frozen/source_manifest.json')
    almanac = read(ROOT / 'artifacts/knowledge_decay_ALMANAC_fold0_seed0_20261119/frozen/source_manifest.json')
    assert sources() == old['package'] == almanac['package']
    for manifest in (old, almanac):
        assert all(sha(p) == h for p, h in manifest['extra'].items())
    extras = dict(old['extra'])
    for p in [ROOT / 'scripts/knowledge_decay.py', ROOT / 'scripts/train_knowledge_decay_oneil_cv5.py',
              ROOT / 'scripts/report_knowledge_decay_oneil_cv5.py', Path(__file__).resolve(),
              PROJECT / 'tools/paper_routes/train_hermes_knowledge_decay_oneil_cv5.py']:
        extras[str(p)] = sha(p)
    write(frozen / 'source_manifest.json', dict(package=sources(), extra=extras))
    inputs = read(ROOT / 'artifacts/zero_gated_hgnn3_cv5_seed0_20261118/frozen/input_manifest.json')
    assert all(sha(p) == h for p, h in inputs.items())
    write(frozen / 'input_manifest.json', inputs)
    for p in [ROOT / name for name in sources()] + [Path(p) for p in extras]:
        relative = 'project/' + str(p.relative_to(ROOT)) if ROOT in p.parents else 'bridge/' + p.name
        target = frozen / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    prior = ROOT / 'artifacts/zero_gated_hgnn3_cv5_seed0_20261118/FIVEFOLD_INPUT_AUDIT.json'
    audit = read(prior)
    assert audit['status'] == 'passed'
    folds = {k: v for k, v in audit['folds'].items() if k.startswith('ONEIL/')}
    assert set(folds) == {'ONEIL/fold%d' % i for i in range(5)}
    references = {}
    for fold in range(5):
        path = ROOT / 'runs/zero_gated_hgnn3_cv5_seed0_20261118/ONEIL' / ('fold%d' % fold) / 'EXPERIMENT.json'
        reference = read(path)
        assert reference['baseline_metadata']['split'] == folds['ONEIL/fold%d' % fold]['split']
        references[str(fold)] = dict(path=str(path), initialization_sha256=reference['initialization_sha256'],
            split=reference['baseline_metadata']['split'])
    write(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json', dict(status='passed', dataset='ONEIL', seed=0,
        folds=folds, references=references, inputs_sha256_reverified=True,
        prior_audit=str(prior), prior_audit_sha256=sha(prior), test_evaluated=False,
        original_and_ALMANAC_frozen_sources_unchanged=True))
    write(DISPATCH / 'PLAN.json', dict(dataset='ONEIL', seed=0, folds=list(range(5)), formal_fits=5,
        epochs=3000, select_from_epoch=1000, selection_metric='internal validation AUROC',
        all_downstream_trainable=True, old_checkpoint_loaded=False, architecture_changed=False,
        knowledge_weight_decay=0., initializer_and_head_weight_decay=.01,
        optimizer='original torch.optim.Adam with coupled decay', gpus=[0, 1],
        global_user_process_limit_per_gpu=2, memory_reserve_mib=6144,
        checks_before_fit=True, fold0_two_full_updates_before_other_folds=True,
        ALMANAC_stays_single_fold=True, extra_seed_or_grid=False, test_evaluated=False))


def submit():
    assert not (DISPATCH / 'LAUNCH.json').exists()
    assert read(DISPATCH / 'cpu_check/progress.json')['status'] == 'complete'
    assert read(DISPATCH / 'cpu_check/SELECTED_DIAGNOSTICS.json')['status'] == 'passed'
    manifest = read(DISPATCH / 'frozen/source_manifest.json')
    assert sources() == manifest['package']
    assert all(sha(p) == h for p, h in manifest['extra'].items())
    state = read(RUNTIME / 'pipeline_progress.json')
    assert state['status'] == 'running' and alive(state['pid'])
    command = Path('/proc/%s/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in command
    assert '--runtime-directory' in command and str(RUNTIME) in command
    assert '--gpu-process-limit' not in command
    active = [job for job in state['active'] if alive(job['pid'])]
    registry = read(RUNTIME / 'auxiliary_jobs.json')
    common = dict(module='train_hermes_knowledge_decay_oneil_cv5', result_storage_root=str(ROOT),
        dataset='ONEIL', gpus=[0, 1], gpu_process_limits={str(i): 2 for i in range(4)}, memory_mib=6144)
    args = ['--decay-mode', 'knowledge_no_decay', '--expected-source-manifest', str(DISPATCH / 'frozen/source_manifest.json'),
        '--expected-input-manifest', str(DISPATCH / 'frozen/input_manifest.json')]
    check = dict(common, key='aux_hermes_knowledge_decay_ONEIL_cv5_check_20261119',
        directory=str(DISPATCH / 'gpu_check'), requires=[str(DISPATCH / 'cpu_check/progress.json')],
        extra_args=['--stage', 'check', '--fold', '0'] + args)
    fits = [dict(common, key='aux_hermes_knowledge_decay_ONEIL_cv5_fold%d_20261119' % fold,
        directory=str(RUN / 'ONEIL' / ('fold%d' % fold)),
        requires=[str(DISPATCH / 'gpu_check/progress.json')] + ([] if fold == 0 else [str(RUN / 'ENTRY_ONEIL.json')]),
        extra_args=['--stage', 'train', '--fold', str(fold)] + args) for fold in range(5)]
    new = [check] + fits
    assert len({job['directory'] for job in new}) == 6
    assert not {job['key'] for job in new} & {job['key'] for job in registry['jobs']}
    write(DISPATCH / 'REGISTRY_BEFORE.json', registry)
    write(DISPATCH / 'STATE_BEFORE.json', state)
    # Restart only the same CPU controller so the new keys are selected. All
    # active GPU workers survive and are re-adopted through process records.
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']):
            break
        time.sleep(.1)
    assert not alive(state['pid'])
    assert all(alive(job['pid']) for job in active)
    assert read(RUNTIME / 'auxiliary_jobs.json') == registry
    registry['jobs'] = new + registry['jobs']
    write(RUNTIME / 'auxiliary_jobs.json', registry)
    position = command.index('--auxiliary-keys')
    command = command[:position + 1] + [job['key'] for job in new] + command[position + 1:]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    for key, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
        XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
        WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        env[key] = str(RUNTIME / suffix)
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'reporter.log').open('a') as stream:
        observer = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/report_knowledge_decay_oneil_cv5.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    scope = ROOT / 'artifacts/formal_cv5_seed0_20261117/gpu1_two_more/POLICY_STATE.json'
    policy = read(scope)
    assert not policy['active_scopes']
    policy.update(controller=controller.pid, command=command)
    write(scope, policy)
    write(DISPATCH / 'LAUNCH.json', dict(status='submitted', controller=controller.pid, command=command,
        reporter=observer.pid, check=check, jobs=fits, formal_fits=5,
        preserved_GPU_workers=active, GPU_workers_signalled=False,
        global_user_process_limit_per_GPU=2, test_evaluated=False))
    write(ROOT / 'ONEIL_OPTIMIZER_CV5_RUN.json', dict(run_directory=str(RUN), dispatch=str(DISPATCH),
        dataset='ONEIL', seed=0, folds=list(range(5)), formal_fits=5,
        variant='knowledge-only weight_decay=0', preferred_gpus=[0, 1], epochs=3000,
        all_downstream_trainable=True, original_checkpoint_loaded=False))
    print(dict(controller=controller.pid, reporter=observer.pid, preserved_GPU_pids=[job['pid'] for job in active],
        formal_fits=5, GPU_checks=1, status='submitted; GPU availability determines actual start'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    prepare() if options.prepare_only else submit()

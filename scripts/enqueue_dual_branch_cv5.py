"""Register the authorized dual branch with the existing, shared GPU queue."""
import hashlib
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

PREPARATION = ROOT / 'artifacts/dual_branch_v1_20261119'
DISPATCH = ROOT / 'artifacts/dual_branch_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_cv5_seed0_20261119'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    environment()
    assert not (DISPATCH / 'LAUNCH.json').exists(), 'Do not submit duplicate fits'
    assert read(PREPARATION / 'DELIVERY_VERIFIED.json')['status'] == 'passed'
    assert read(PREPARATION / 'CPU_VALIDATION_FINISHED.json')['status'] == 'passed'
    old_manifest = read(PREPARATION / 'frozen/manifest.json')
    changed = [p for p, digest in old_manifest['source'].items() if sha(p) != digest]
    assert changed == [str(ROOT / 'dual_branch/cli.py')], changed
    # The only post-CPU change adds an explicit GPU correctness batch size and
    # reserved-memory reporting. No learned module or training formula changes.
    assert all(sha(p) == digest for p, digest in old_manifest['inputs'].items())
    for name in ['zero_gated_hgnn3_cv5_seed0_20261118',
                 'knowledge_decay_ALMANAC_fold0_seed0_20261119',
                 'knowledge_decay_ONEIL_cv5_seed0_20261119']:
        historical = read(ROOT / 'artifacts' / name / 'frozen/source_manifest.json')
        assert all(sha(ROOT / p) == digest for p, digest in historical['package'].items())
        assert all(sha(p) == digest for p, digest in historical['extra'].items())

    import torch
    from dual_branch.data import load
    historical_audit = read(ROOT / 'artifacts/zero_gated_hgnn3_cv5_seed0_20261118/FIVEFOLD_INPUT_AUDIT.json')
    folds = {}
    for dataset in ['ALMANAC', 'ONEIL']:
        for fold in range(5):
            data = load(dataset, fold)
            key = '%s/fold%d' % (dataset, fold)
            split = data['raw']['metadata']['split']
            assert split == historical_audit['folds'][key]['split']
            for part, label in [('fit', 'train'), ('valid', 'valid')]:
                rows = data['raw'][part]['rows']
                assert hashlib.sha256(rows.numpy().tobytes()).hexdigest() == split[label + '_array_sha256']
                assert len(rows) == split[label + '_samples']
            folds[key] = dict(split=split, synergy_audit=data['graph']['synergy_audit'],
                semantic_sha256=data['input_provenance']['semantic_sha256'],
                baseline=historical_audit['folds'][key]['baseline'])
            del data
    write(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json', dict(status='passed', folds=folds,
        seed=0, fold_count=10, original_split_hashes_matched=True,
        test_evaluated=False, old_checkpoints_loaded=False))
    source = {p: sha(p) for p in old_manifest['source']}
    for path in [Path(__file__).resolve(), ROOT / 'scripts/report_dual_branch_cv5.py']:
        source[str(path)] = sha(path)
    manifest = dict(source=source, inputs=old_manifest['inputs'],
        version='dual_branch_v1_20261119', previous_CPU_manifest=str(PREPARATION / 'frozen/manifest.json'),
        post_CPU_change='CLI check-rows argument and memory reporting only; model unchanged',
        historical_frozen_sources_unchanged=True)
    frozen = DISPATCH / 'frozen/manifest.json'
    write(frozen, manifest)
    for path in map(Path, source):
        relative = str(path.relative_to(ROOT)) if ROOT in path.parents else 'external/' + str(path).lstrip('/')
        destination = DISPATCH / 'frozen/source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    common = dict(module='train_hermes_dual_branch', result_storage_root=str(ROOT),
        gpus=[1, 3, 2, 0], gpu_process_limits={str(i): 2 for i in range(4)})
    extra = ['--expected-manifest', str(frozen)]
    checks = [dict(common, key='aux_hermes_dual_branch_cv5_%s_check_20261119' % dataset,
        dataset=dataset, directory=str(DISPATCH / ('gpu_check_' + dataset)),
        memory_mib=8192 if dataset == 'ALMANAC' else 6144, requires=[],
        extra_args=['--stage', 'check', '--fold', '0', '--check-rows', '1024'] + extra)
        for dataset in ['ALMANAC', 'ONEIL']]
    dependencies = [str(Path(job['directory']) / 'progress.json') for job in checks]
    fits = [dict(common, key='aux_hermes_dual_branch_cv5_%s_fold%d_20261119' % (dataset, fold),
        dataset=dataset, directory=str(RUN / dataset / ('fold%d' % fold)),
        memory_mib=12288 if dataset == 'ALMANAC' else 10240,
        requires=dependencies + [str(DISPATCH / 'MEMORY_READY.json')] +
            ([] if fold == 0 else [str(RUN / ('ENTRY_' + dataset + '.json'))]),
        extra_args=['--stage', 'train', '--fold', str(fold)] + extra)
        for fold in range(5) for dataset in ['ALMANAC', 'ONEIL']]
    registry = read(RUNTIME / 'auxiliary_jobs.json')
    state = read(RUNTIME / 'pipeline_progress.json')
    assert state['status'] == 'running' and alive(state['pid'])
    assert not state.get('gpu_process_limit_overrides')
    assert not any(job['module'] == 'train_hermes_dual_branch' for job in registry['jobs'])
    assert len({job['directory'] for job in checks + fits}) == 12
    assert not set(job['key'] for job in checks + fits) & set(job['key'] for job in registry['jobs'])
    command = Path('/proc/%d/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
    assert 'tools.paper_routes.queue' in command and '--auxiliary-keys' in command
    assert '--runtime-directory' in command and str(RUNTIME) in command
    assert '--gpu-process-limit' not in command
    command += [job['key'] for job in checks + fits]
    write(DISPATCH / 'REGISTRY_BEFORE.json', registry)
    write(DISPATCH / 'STATE_BEFORE.json', state)
    # Restart only the CPU controller; all live GPU children are adopted.
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(100):
        if not alive(state['pid']):
            break
        time.sleep(.05)
    assert not alive(state['pid'])
    assert read(RUNTIME / 'auxiliary_jobs.json') == registry
    registry['jobs'] += checks + fits
    registry['hermes_dual_branch_authorization_20261119'] = dict(
        instruction='Find available space and run the implemented dual-branch version',
        datasets=['ALMANAC', 'ONEIL'], seed=0, folds=list(range(5)),
        checks_before_fit=True, max_user_GPU_processes_per_card=2)
    write(RUNTIME / 'auxiliary_jobs.json', registry)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'report.log').open('a') as stream:
        reporter = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/report_dual_branch_cv5.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(controller=controller.pid, reporter=reporter.pid, command=command,
        checks=checks, jobs=fits, run_directory=str(RUN), source_manifest=str(frozen),
        old_GPU_workers_signalled=False, previous_active_workers=state.get('active', []),
        previous_controller=state['pid'])
    write(DISPATCH / 'LAUNCH.json', launch)
    write(DISPATCH / 'PLAN.json', dict(launch, datasets=['ALMANAC', 'ONEIL'], seed=0,
        folds_per_dataset=5, formal_fits=10, epochs=3000, select_from_epoch=1000,
        all_downstream_trainable=True, old_checkpoint_loaded=False,
        fixed_pretrained_inputs=['CODER', 'Uni-Mol'], test_evaluated=False,
        first_fold_full_fit_updates_before_other_folds=3,
        max_user_GPU_processes_per_card=2, GPU_check_query_rows=1024,
        no_extra_seeds_or_parameter_grid=True))
    previous = read(ROOT / 'CURRENT_RUN.json')
    write(ROOT / 'CURRENT_RUN.json', dict(variant='All-node CODER + 2D/3D/Cell dual branch',
        run_directory=str(RUN), dispatch=str(DISPATCH), status_file=str(RUN / 'STATUS.json'),
        datasets=['ALMANAC', 'ONEIL'], folds_per_dataset=5, seed=0, epochs=3000, formal_fits=10,
        all_downstream_trainable=True, old_checkpoint_loaded=False, previous_run=previous))
    print(dict(status='submitted', controller=controller.pid, reporter=reporter.pid,
        checks=2, formal_fits=10, GPU_workers_signalled=False), flush=True)


if __name__ == '__main__':
    main()

"""Submit the authorized ten Transformer folds to the single shared queue."""
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

DISPATCH = ROOT / 'artifacts/dual_branch_transformer_cv5_seed0_20261120'
RUN = ROOT / 'runs/dual_branch_transformer_cv5_seed0_20261120'
PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'
MODULE = 'train_hermes_prediction_transformer'


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    environment()
    assert not (DISPATCH / 'LAUNCH.json').exists(), 'Never submit duplicate folds'
    assert read(ROOT / 'artifacts/prediction_transformer_20261120/DELIVERY.json')['status'] == 'implemented_and_CPU_verified'
    before = read(ROOT / 'artifacts/prediction_transformer_20261120/manifest.json')
    assert all(sha(p) == h for p, h in before['source'].items())
    # Rehash every reused cache, including all ten fold inputs. Immutable cache
    # identity permits reusing the prior per-row split and graph-label audit.
    assert all(sha(p) == h for p, h in before['inputs'].items())
    previous_audit = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119/FIVEFOLD_INPUT_AUDIT.json'
    audit = read(previous_audit)
    assert audit['status'] == 'passed' and len(audit['folds']) == 10
    audit.update(reused_after_all_input_files_rehashed=True,
        previous_audit=str(previous_audit), current_input_manifest=str(DISPATCH / 'frozen/manifest.json'),
        neural_checkpoint_reused=False, prediction_Transformer_added=True,
        Cell_inputs_unchanged=True, PrimeKG_selection_unchanged=True)
    write(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json', audit)
    source = dict(before['source'])
    for p in [Path(__file__).resolve(), ROOT / 'scripts/report_prediction_transformer_cv5.py',
              PROJECT / 'tools/paper_routes/train_hermes_prediction_transformer.py']:
        source[str(p)] = sha(p)
    manifest = DISPATCH / 'frozen/manifest.json'
    write(manifest, dict(source=source, inputs=before['inputs'],
        version='dual_branch_prediction_transformer_20261120',
        previous_CPU_manifest=str(ROOT / 'artifacts/prediction_transformer_20261120/manifest.json'),
        neural_checkpoint_loaded=False, original_frozen_sources_unchanged=True))
    for filename in source:
        p = Path(filename)
        rel = str(p.relative_to(ROOT)) if ROOT in p.parents else 'external/' + str(p).lstrip('/')
        target = DISPATCH / 'frozen/source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    common = dict(module=MODULE, result_storage_root=str(ROOT),
        gpus=[0, 1, 2, 3], gpu_process_limits={str(i): 2 for i in range(4)})
    extra = ['--expected-manifest', str(manifest)]
    checks = [dict(common, key='aux_hermes_prediction_transformer_%s_check_20261120' % ds,
        dataset=ds, directory=str(DISPATCH / ('gpu_check_' + ds)),
        gpus=[0] if ds == 'ALMANAC' else [1, 2, 3, 0],
        memory_mib=10240 if ds == 'ALMANAC' else 7680,
        requires=[str(DISPATCH / ('CPU_READY_' + ds + '.json'))],
        extra_args=['--stage', 'check', '--fold', '0', '--check-rows', '1024'] + extra)
        for ds in ['ALMANAC', 'ONEIL']]
    fits = [dict(common, key='aux_hermes_prediction_transformer_%s_fold%d_20261120' % (ds, fold),
        dataset=ds, directory=str(RUN / ds / ('fold%d' % fold)),
        memory_mib=12288 if ds == 'ALMANAC' else 8192,
        requires=[str(DISPATCH / ('gpu_check_' + ds) / 'progress.json'),
                  str(DISPATCH / ('MEMORY_READY_' + ds + '.json'))]
                 + ([] if fold == 0 else [str(RUN / ('ENTRY_' + ds + '.json'))]),
        extra_args=['--stage', 'train', '--fold', str(fold)] + extra)
        for fold in range(5) for ds in ['ALMANAC', 'ONEIL']]
    new_jobs = checks + fits
    assert len({j['directory'] for j in new_jobs}) == 12
    assert all(str(Path(j['directory']).resolve()).startswith('/ifs2/') for j in new_jobs)
    registry = read(RUNTIME / 'auxiliary_jobs.json')
    assert not set(j['key'] for j in new_jobs) & {j['key'] for j in registry['jobs']}
    state = read(RUNTIME / 'pipeline_progress.json')
    policy = read(RUNTIME / 'resource_policy.json')
    assert not policy.get('gpu_process_limit_overrides')
    if alive(state['pid']):
        cmd = Path('/proc/%d/cmdline' % state['pid']).read_bytes().decode().rstrip('\0').split('\0')
        assert 'tools.paper_routes.queue' in cmd and '--auxiliary-keys' in cmd
        assert '--runtime-directory' in cmd and str(RUNTIME) in cmd
        assert '--gpu-process-limit' not in cmd
    else:
        assert state['status'] == 'complete'
        cmd = [sys.executable, '-B', '-u', '-m', 'tools.paper_routes.queue',
            '--output', '/ifs/data/liushuhong/work2/results/paper_routes_20261009',
            '--runtime-directory', str(RUNTIME), '--gpus', '1', '0', '2', '3',
            '--qa-gpu', '-1', '--auxiliary-only', '--no-source-checks', '--auxiliary-keys']
        cmd += policy['selected_auxiliary_keys']
    cmd += [j['key'] for j in new_jobs]
    write(DISPATCH / 'REGISTRY_BEFORE.json', registry)
    write(DISPATCH / 'STATE_BEFORE.json', state)
    write(DISPATCH / 'POLICY_BEFORE.json', policy)
    old_workers = {j['pid']: Path('/proc/%d/cmdline' % j['pid']).read_bytes().decode().split('\0')
                   for j in state.get('active', []) if alive(j['pid'])}
    # Only restart the CPU controller. GPU children use independent sessions
    # and are adopted by the same queue after its shared lock is released.
    if alive(state['pid']):
        try:
            os.kill(state['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(100):
            if not alive(state['pid']):
                break
            time.sleep(.05)
        assert not alive(state['pid'])
    assert read(RUNTIME / 'auxiliary_jobs.json') == registry
    registry['jobs'] += new_jobs
    registry['prediction_transformer_authorization_20261120'] = dict(
        user_request='直接运行五折；结果存到IFS2不要存IFS',
        datasets=['ALMANAC', 'ONEIL'], seed=0, folds=list(range(5)),
        formal_fits=10, CPU_and_GPU_checks_required=True,
        all_downstream_trainable=True, old_checkpoint_loaded=False,
        max_user_GPU_processes_per_card=2)
    write(RUNTIME / 'auxiliary_jobs.json', registry)
    write(ROOT / 'PREDICTION_TRANSFORMER_VERSION.json', dict(
        version='dual_branch_prediction_transformer_20261120', status='submitted',
        run_directory=str(RUN), dispatch=str(DISPATCH), formal_fits=10,
        documentation=str(ROOT / 'TRANSFORMER_PREDICTION.md')))
    previous = read(ROOT / 'CURRENT_RUN.json')
    write(ROOT / 'CURRENT_RUN.json', dict(variant='Dual branch + prediction Transformer',
        status='submitted', datasets=['ALMANAC', 'ONEIL'], folds_per_dataset=5,
        seed=0, epochs=3000, run_directory=str(RUN), dispatch=str(DISPATCH),
        status_file=str(RUN / 'STATUS.json'), all_downstream_trainable=True,
        old_checkpoint_loaded=False, previous_run=previous))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'report.log').open('a') as stream:
        reporter = subprocess.Popen([sys.executable, '-B', '-u',
            str(ROOT / 'scripts/report_prediction_transformer_cv5.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(controller=controller.pid, reporter=reporter.pid, command=cmd,
        checks=checks, jobs=fits, run_directory=str(RUN), source_manifest=str(manifest),
        previous_controller=state['pid'], previous_active_workers=old_workers,
        old_GPU_workers_signalled=False, per_card_global_process_limit=2)
    write(DISPATCH / 'LAUNCH.json', launch)
    write(DISPATCH / 'PLAN.json', dict(launch, datasets=['ALMANAC', 'ONEIL'],
        seed=0, folds_per_dataset=5, formal_fits=10, epochs=3000,
        select_from_epoch=1000, selection_metric='internal validation AUROC',
        all_downstream_trainable=True, old_checkpoint_loaded=False,
        fixed_pretrained_inputs=['CODER', 'Uni-Mol'], test_evaluated=False,
        first_fold_full_fit_updates_before_other_folds=3,
        all_new_outputs_under_IFS2=True, no_extra_seeds_or_grid=True))
    for old in state.get('active', []):
        if old['pid'] in old_workers:
            assert alive(old['pid']) or read(Path(old['directory']) / 'progress.json').get('status') == 'complete'
    print(dict(status='submitted', controller=controller.pid, reporter=reporter.pid,
        GPU_checks=2, formal_fits=10, output=str(RUN)), flush=True)


if __name__ == '__main__':
    main()

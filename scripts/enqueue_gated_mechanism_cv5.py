"""Submit exactly two checks and ten fresh folds to the existing shared queue."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import RUNTIME, environment
from dual_branch.gated_common import DISPATCH, RUN, VERSION
from global_cell.common import read, write, sha

PROJECT = ROOT.parent / 'mcdsp_moe/MCDSP-main'
MODULE = 'train_hermes_gated_mechanism_cv5'
SHARED_OUTPUT = '/ifs/data/liushuhong/work2/results/paper_routes_20261009'


def main():
    environment()
    assert not (DISPATCH / 'LAUNCH.json').exists(), 'No duplicate folds'
    assert read(DISPATCH / 'UNIT_CHECKS.json')['status'] == 'passed'
    manifest = DISPATCH / 'frozen/manifest.json'
    frozen = read(manifest)
    for kind in ['source', 'inputs']:
        assert all(sha(p) == digest for p, digest in frozen[kind].items()), kind
    for ds in ['ALMANAC', 'ONEIL']:
        checks = read(DISPATCH / ('cpu_' + ds) / 'CHECKS.json')
        assert checks['status'] == 'passed' and checks['fusion_gate']['status'] == 'passed'
        write(DISPATCH / ('CPU_READY_' + ds + '.json'), dict(status='complete', checks=checks))
    common = dict(module=MODULE, result_storage_root=str(ROOT), gpus=[2, 0, 1, 3],
                  gpu_process_limits={str(i): 2 for i in range(4)})
    extra = ['--expected-manifest', str(manifest)]
    checks = [dict(common, key='aux_hermes_gated_mechanism_%s_check_20261122' % ds, dataset=ds,
        directory=str(DISPATCH / ('gpu_check_' + ds)), memory_mib=8192 if ds == 'ALMANAC' else 6144,
        requires=[str(DISPATCH / ('CPU_READY_' + ds + '.json'))],
        extra_args=['--stage', 'check', '--fold', '0', '--check-rows', '1024'] + extra)
        for ds in ['ALMANAC', 'ONEIL']]
    fits = [dict(common, key='aux_hermes_gated_mechanism_%s_fold%d_20261122' % (ds, fold), dataset=ds,
        directory=str(RUN / ds / ('fold%d' % fold)), memory_mib=8192 if ds == 'ALMANAC' else 6144,
        requires=[str(DISPATCH / ('GPU_READY_' + ds + '.json'))] + ([] if fold == 0 else [str(RUN / ('ENTRY_' + ds + '.json'))]),
        extra_args=['--stage', 'train', '--fold', str(fold)] + extra)
        for fold in range(5) for ds in ['ALMANAC', 'ONEIL']]
    jobs = checks + fits
    assert len({job['directory'] for job in jobs}) == 12
    lock = (RUNTIME / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        state = read(RUNTIME / 'pipeline_progress.json')
        policy = read(RUNTIME / 'resource_policy.json')
        assert state['status'] == 'complete', 'An existing controller needs safe adoption first'
        assert policy['max_user_processes_per_gpu'] == 2 and not policy.get('gpu_process_limit_overrides')
        registry = read(RUNTIME / 'auxiliary_jobs.json')
        assert not {j['key'] for j in jobs} & {j['key'] for j in registry['jobs']}
        write(DISPATCH / 'REGISTRY_BEFORE.json', registry)
        write(DISPATCH / 'STATE_BEFORE.json', state)
        write(DISPATCH / 'POLICY_BEFORE.json', policy)
        registry['jobs'] = jobs + registry['jobs']
        registry['gated_mechanism_cv5_authorization_20261122'] = dict(
            user_request='你先运行一下五折', interpretation='last-discussed conditioned-gate version',
            datasets=['ALMANAC', 'ONEIL'], seed=0, folds=list(range(5)), formal_fits=10,
            all_downstream_trainable=True, old_checkpoint_loaded=False, sealed_test_evaluated=False)
        write(RUNTIME / 'auxiliary_jobs.json', registry)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    command = [sys.executable, '-B', '-u', '-m', 'tools.paper_routes.queue',
        '--output', SHARED_OUTPUT, '--runtime-directory', str(RUNTIME),
        '--gpus', '2', '0', '1', '3', '--qa-gpu', '-1', '--auxiliary-only', '--no-source-checks',
        '--auxiliary-keys'] + [job['key'] for job in jobs]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (DISPATCH / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    with (DISPATCH / 'report.log').open('a') as stream:
        reporter = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/report_gated_mechanism_cv5.py'), '--watch'],
            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(DISPATCH / 'LAUNCH.json', dict(controller=controller.pid, reporter=reporter.pid, command=command,
        jobs=jobs, GPU_checks=2, formal_fits=10, all_user_GPU_process_cap=2, old_GPU_workers_signalled=False,
        source_manifest=str(manifest), old_registry_jobs_preserved=len(registry['jobs']) - 12))
    write(DISPATCH / 'PREVIOUS_CURRENT_RUN.json', read(ROOT / 'CURRENT_RUN.json'))
    write(DISPATCH / 'PREVIOUS_CURRENT_MODEL.json', read(ROOT / 'CURRENT_MODEL.json'))
    pointer = dict(variant='Conditioned-gate multimodal + author Refiner + mechanism evidence',
        status='submitted', run_directory=str(RUN), dispatch=str(DISPATCH), status_file=str(RUN / 'STATUS.json'),
        datasets=['ALMANAC', 'ONEIL'], folds_per_dataset=5, seed=0, epochs=3000,
        all_downstream_trainable=True, old_checkpoint_loaded=False, all_outputs_on_IFS2=True,
        primary_selection='Val AUPRC from first evaluation', source_manifest=str(manifest))
    write(ROOT / 'GATED_MECHANISM_RUN.json', pointer)
    write(ROOT / 'CURRENT_RUN.json', pointer)
    write(ROOT / 'CURRENT_MODEL.json', dict(pointer, model_class='dual_branch.gated_mechanism.GatedMechanismModel',
        code=str(ROOT / 'dual_branch/gated_mechanism.py'), training_entry='dual_branch.gated_cli'))
    print(dict(status='submitted', controller=controller.pid, reporter=reporter.pid, GPU_checks=2, formal_fits=10), flush=True)


if __name__ == '__main__':
    main()

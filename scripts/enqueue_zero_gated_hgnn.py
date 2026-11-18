"""Submit only the authorized three-round ALMANAC seed0/fold0 pilot."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sources, sha

PROJECT = Path('/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
RUNTIME = Path('/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009')
DISPATCH = ROOT/'artifacts/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118'
RUN = ROOT/'runs/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118'
PREDECESSOR = ROOT/'runs/almanac_fold0_synergy002_drop20_seed0_20261118'


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    assert not (DISPATCH/'LAUNCH.json').exists()
    assert read(PREDECESSOR/'FINAL_VERIFICATION.json')['status'] == 'complete_verified'
    assert read(PREDECESSOR/'metrics.json')['valid_aupr'] < read(PREDECESSOR/'EXPERIMENT.json')['baseline_metadata']['reference_valid_aupr']
    assert read(DISPATCH/'UNIT_TESTS.json')['status'] == 'passed'
    assert read(DISPATCH/'cpu_check/CHECKS.json')['status'] == 'passed'
    assert all(x['delta_rms'] == 0. for x in read(DISPATCH/'cpu_check/INITIAL_ZERO_GATE_AUDIT.json')['layers'])
    previous = read(ROOT/'artifacts/similarity_refiner_fold0_seed0_20261118/WITHDRAWN.json')
    assert not alive(previous['controller'])
    assert not alive(previous['predecessor_worker'])
    state = read(RUNTIME/'pipeline_progress.json')
    assert state['status'] == 'complete'
    baseline_package = read(ROOT/'artifacts/almanac_fold0_synergy002_drop20_seed0_20261118/frozen/source_manifest.json')['package']
    assert sources() == baseline_package
    frozen = DISPATCH/'frozen'
    frozen.mkdir()
    extras = [ROOT/'models/zero_gated_hgnn.py', ROOT/'scripts/train_zero_gated_hgnn.py',
        Path(__file__).resolve(), ROOT/'scripts/report_zero_gated_hgnn.py',
        ROOT/'tests/test_zero_gated_hgnn.py', PROJECT/'tools/paper_routes/train_hermes_zero_gated_hgnn.py']
    write(frozen/'source_manifest.json', dict(package=sources(), extra={str(p): sha(p) for p in extras}))
    inputs = {p: h for p, h in read(ROOT/'artifacts/source_freeze_20261117/input_manifest.json').items()
              if '/ONEIL/' not in p and '/ONEIL-COSMIC/' not in p}
    assert all(sha(p) == h for p, h in inputs.items())
    write(frozen/'input_manifest.json', inputs)
    for p in [ROOT/name for name in sources()] + extras:
        relative = 'project/'+str(p.relative_to(ROOT)) if ROOT in p.parents else 'bridge/'+p.name
        destination = frozen/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    shared = dict(module='train_hermes_zero_gated_hgnn', dataset='ALMANAC',
        result_storage_root=str(ROOT), gpus=[1,0,3,2],
        gpu_process_limits={str(i):2 for i in range(4)})
    extra = ['--expected-source-manifest', str(frozen/'source_manifest.json'),
             '--expected-input-manifest', str(frozen/'input_manifest.json')]
    check = dict(shared, key='aux_hermes_zero_gated_hgnn3_ALMANAC_fold0_check_20261118',
        directory=str(DISPATCH/'gpu_check'), memory_mib=26000,
        requires=[str(PREDECESSOR/'progress.json')], extra_args=['--stage','check','--fold','0']+extra)
    fit = dict(shared, key='aux_hermes_zero_gated_hgnn3_ALMANAC_fold0_fit_20261118',
        directory=str(RUN), memory_mib=32000,
        requires=[str(DISPATCH/'gpu_check/progress.json')], extra_args=['--stage','train','--fold','0']+extra)
    registry = read(RUNTIME/'auxiliary_jobs.json')
    assert not {check['key'],fit['key']} & {j['key'] for j in registry['jobs']}
    assert not any(j['module']=='train_hermes_similarity_refiner' for j in registry['jobs'])
    write(DISPATCH/'REGISTRY_BEFORE.json', registry)
    registry['jobs'].extend([check,fit])
    write(RUNTIME/'auxiliary_jobs.json', registry)
    command = previous['command']+[check['key'],fit['key']]
    assert '--runtime-directory' in command and str(RUNTIME) in command
    assert '--gpu-process-limit' not in command
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    for key, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
        XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
        WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        path = RUNTIME/suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    write(DISPATCH/'PLAN.json', dict(dataset='ALMANAC', seed=0, folds=[0], formal_fits=1,
        epochs=3000, select_from_epoch=1000, propagation_layers=3,
        zero_initialized_propagation_gates=True, original_readout_gate_initialization=.05,
        test_evaluated=False, all_downstream_trainable=True, old_checkpoint_loaded=False,
        checks=[check], jobs=[fit], predecessor_final_selection=read(PREDECESSOR/'selection.json'),
        max_user_gpu_processes_per_card=2,
        memory_note='Conservative reserves for three live graph rounds; do not bypass resource limits'))
    with (DISPATCH/'controller.log').open('a') as f:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=f,
            stderr=subprocess.STDOUT, start_new_session=True)
    scope = ROOT/'artifacts/formal_cv5_seed0_20261117/gpu1_two_more/POLICY_STATE.json'
    policy = read(scope)
    assert not policy['active_scopes']
    policy.update(controller=child.pid, command=command)
    write(scope, policy)
    with (DISPATCH/'report.log').open('a') as f:
        reporter = subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/report_zero_gated_hgnn.py'),'--watch'],
            cwd=ROOT, env=dict(env,CUDA_VISIBLE_DEVICES=''), stdout=f,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(DISPATCH/'LAUNCH.json', dict(controller=child.pid, command=command,
        reporter=reporter.pid, checks=[check], jobs=[fit], run=str(RUN),
        source_manifest=str(frozen/'source_manifest.json'), old_worker_signalled=False))
    write(ROOT/'CURRENT_RUN.json', dict(variant='3-layer real alpha/beta with zero-initialized propagation gates',
        dataset='ALMANAC', fold=0, training_seed=0, formal_fits=1, epochs=3000,
        run_directory=str(RUN), status_file=str(RUN/'STATUS.json'), dispatch=str(DISPATCH),
        previous_completed_run=str(PREDECESSOR), old_checkpoint_loaded=False,
        all_downstream_trainable=True, original_readout_gate_initialization=.05))
    print(dict(controller=child.pid, reporter=reporter.pid, GPU_checks=1, formal_fits=1,
        status='queued; wait for memory and shared process slots'), flush=True)


if __name__ == '__main__':
    main()

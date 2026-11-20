"""Freeze the corrected model and start only CPU checks/reporting.

The shared queue, already staged by the replacement script, is the sole GPU
dispatcher and waits for each actual CPU check to complete.
"""
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
NEW = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119'
OLD = ROOT / 'artifacts/dual_branch_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119'
PYTHON = '/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python'

assert not (NEW / 'CPU_CHECKS_LAUNCH.json').exists()
unit = (NEW / 'unit_checks.log').read_text()
assert 'Ran 6 tests' in unit and '\nOK\n' in unit
write(NEW / 'UNIT_CHECKS.json', dict(status='passed', tests=6, returncode=0))
for name in ['zero_gated_hgnn3_cv5_seed0_20261118',
             'knowledge_decay_ALMANAC_fold0_seed0_20261119',
             'knowledge_decay_ONEIL_cv5_seed0_20261119']:
    old = read(ROOT / 'artifacts' / name / 'frozen/source_manifest.json')
    assert all(sha(ROOT / path) == digest for path, digest in old['package'].items())
    assert all(sha(path) == digest for path, digest in old['extra'].items())
previous = read(OLD / 'frozen/manifest.json')
assert all(sha(path) == digest for path, digest in previous['inputs'].items())
source_paths = [Path(name) for name in previous['source']]
source_paths += [Path(__file__).resolve(), ROOT / 'scripts/report_dual_branch_gemini_cv5.py',
    ROOT / 'artifacts/replace_dual_branch_with_gemini_20261119.py']
manifest = dict(source={str(path): sha(path) for path in source_paths}, inputs=previous['inputs'],
    version='dual_branch_gemini_v2_20261119', original_input_hashes_unchanged=True,
    source_of_supplied_code='/ifs2/home/liushuhong/.codex/attachments/670d0592-97ff-4520-bf42-4f7c553cbae2/已粘贴的文本.txt',
    old_three_historical_model_source_manifests_unchanged=True)
assert all(Path(name).is_absolute() for name in manifest['source'])
frozen = NEW / 'frozen/manifest.json'
write(frozen, manifest)
for path in source_paths:
    relative = str(path.relative_to(ROOT)) if ROOT in path.parents else 'external/' + str(path).lstrip('/')
    destination = NEW / 'frozen/source_snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
audit = read(OLD / 'FIVEFOLD_INPUT_AUDIT.json')
assert audit['status'] == 'passed' and len(audit['folds']) == 10
audit.update(reused_after_all_input_files_rehashed=True,
    previous_audit=str(OLD / 'FIVEFOLD_INPUT_AUDIT.json'),
    current_input_manifest=str(frozen), neural_checkpoint_reused=False)
write(NEW / 'FIVEFOLD_INPUT_AUDIT.json', audit)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
checks = []
for dataset in ['ALMANAC', 'ONEIL']:
    directory = NEW / ('cpu_check_' + dataset)
    command = [PYTHON, '-B', '-u', '-m', 'dual_branch.cli', '--dataset', dataset,
        '--fold', '0', '--stage', 'check', '--device', 'cpu', '--check-rows', '128',
        '--result-directory', str(directory), '--expected-manifest', str(frozen)]
    with (NEW / ('cpu_check_' + dataset + '.log')).open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    checks.append(dict(dataset=dataset, pid=child.pid, command=command, physical_device='cpu'))
write(NEW / 'CPU_CHECKS_LAUNCH.json', dict(status='running', checks=checks,
    GPU_dispatcher='existing shared queue.py', no_GPU_started_by_this_script=True))
write(ROOT / 'DUAL_BRANCH_VERSION.json', dict(version='dual_branch_gemini_v2_20261119',
    status='checking', code_directory=str(ROOT / 'dual_branch'), documentation=str(ROOT / 'DUAL_BRANCH_GEMINI_V2.md'),
    unit_tests=read(NEW / 'UNIT_CHECKS.json'), preparation_cache_reused=True,
    preparation=read(ROOT / 'artifacts/dual_branch_v1_20261119/PREPARATION.json'),
    new_source_manifest=str(frozen), formal_training_fits_submitted=10,
    all_downstream_trainable=True, old_checkpoint_loaded=False, test_evaluated=False))
with (NEW / 'report.log').open('a') as stream:
    observer = subprocess.Popen(['/usr/bin/python3', '-B', '-u', str(ROOT / 'scripts/report_dual_branch_gemini_cv5.py'), '--watch'],
        cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
replacement = read(NEW / 'REPLACEMENT.json')
write(NEW / 'LAUNCH.json', dict(controller=replacement['controller'], reporter=observer.pid,
    command=replacement['command'], checks=replacement['checks'], jobs=replacement['jobs'],
    run_directory=str(RUN), source_manifest=str(frozen), CPU_checks=checks,
    old_only_dual_branch_workers_stopped=True, previous_other_workers_preserved=True))
write(NEW / 'PLAN.json', dict(seed=0, datasets=['ALMANAC', 'ONEIL'], folds_per_dataset=5,
    formal_fits=10, epochs=3000, select_from_epoch=1000, selection='internal validation AUROC',
    query_chunk=1024, original_data_splits_and_graphs=True,
    graph_gate_initial=.1, fusion_output_initial=0., norm_scale='tanh(context RMS)',
    optimizer='Adam grouped', modal_lr=1e-3, knowledge_lr=2e-3,
    weight_decay=1e-4, gate_and_output_weight_decay=0.,
    all_downstream_trainable=True, old_checkpoint_loaded=False,
    no_extra_seeds_or_grid=True, test_evaluated=False,
    max_user_GPU_processes_per_card=2, each_dataset_requires_own_CPU_and_GPU_checks=True))
print(dict(CPU_checks=checks, reporter=observer.pid, controller=replacement['controller'],
    source_files=len(manifest['source']), input_files=len(manifest['inputs'])), flush=True)

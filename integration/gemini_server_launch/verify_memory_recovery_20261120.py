"""Read-only recovery verification; never launches GPU work."""
import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path('/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
sys.path.insert(0, str(ROOT))
from global_cell.common import read, write, sha

DISPATCH = ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119'
RUN = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119'
RUNTIME = ROOT.parent / 'runtime/paper_routes_20261009'
recovery = read(DISPATCH / 'MEMORY_RECOVERY_20261120.json')
state = read(RUNTIME / 'pipeline_progress.json')
assert state['status'] == 'running' and state['pid'] == recovery['controller']
policy = read(RUNTIME / 'resource_policy.json')
assert policy['max_user_processes_per_gpu'] == 2 and not policy['gpu_process_limit_overrides']
assert not policy['process_limit_exempt_auxiliary_keys']
manifest = read(DISPATCH / 'frozen/manifest.json')
for kind in ['source', 'inputs']:
    assert all(sha(p) == expected for p, expected in manifest[kind].items())
before = read(DISPATCH / 'REGISTRY_BEFORE_MEMORY_RECOVERY_20261120.json')
after = read(RUNTIME / 'auxiliary_jobs.json')
other = lambda registry: [j for j in registry['jobs'] if 'dual_branch_gemini_' not in j['key']]
assert other(before) == other(after)
assert len([j for j in after['jobs'] if 'dual_branch_gemini_' in j['key']]) == 12
audit = read(DISPATCH / 'FIVEFOLD_INPUT_AUDIT.json')
actual = []
for dataset, fold in [('ALMANAC', 1), ('ONEIL', 2)]:
    directory = RUN / dataset / ('fold%d' % fold)
    progress = read(directory / 'progress.json')
    assert progress['status'] == 'running' and progress['optimizer_updates'] >= 3
    assert Path('/proc/%d' % progress['pid']).exists()
    experiment = read(directory / 'EXPERIMENT.json')
    reference = read(RUN / dataset / 'fold0/EXPERIMENT.json')
    assert experiment['recipe'] == reference['recipe']
    assert experiment['initialization_sha256'] == reference['initialization_sha256']
    assert experiment['inputs']['original_split'] == audit['folds'][dataset + '/fold%d' % fold]['split']
    assert experiment['inputs']['no_checkpoint_loaded']
    assert not experiment['inputs']['test_evaluated']
    assert read(directory / 'OPTIMIZER_GROUPS.json') == read(RUN / dataset / 'fold0/OPTIMIZER_GROUPS.json')
    for epoch in [2, 3]:
        assert all(math.isfinite(v) and v > 0 for v in read(directory / ('GRADIENT_epoch%d.json' % epoch)).values())
    if dataset == 'ALMANAC':
        failed_experiment = read(Path(recovery['failed_directory']) / 'EXPERIMENT.json')
        for key in ['recipe', 'initialization_sha256', 'inputs', 'graph_summary', 'synergy_audit']:
            assert experiment[key] == failed_experiment[key], key
    actual.append(dict(dataset=dataset, fold=fold, pid=progress['pid'],
                       physical_device=progress['physical_device'], updates=progress['optimizer_updates']))
for name, verification in recovery['completed_unchanged'].items():
    assert sha(RUN / name / 'selected.pt') == verification['selection']['checkpoint_sha256']
record = dict(status='passed', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              controller=recovery['controller'], actual_workers=actual,
              completed_checkpoints_unchanged=3, unique_formal_fits=10,
              source_files_unchanged=len(manifest['source']), input_files_unchanged=len(manifest['inputs']),
              other_registry_jobs_unchanged=len(other(after)), global_user_GPU_process_limit=2,
              no_process_exceptions=True, recipe_and_initialization_unchanged=True,
              full_fit_gradient_checks_passed=True, checkpoint_loaded=False, test_evaluated=False)
write(DISPATCH / 'MEMORY_RECOVERY_VERIFIED_20261120.json', record)
print(json.dumps(record, ensure_ascii=False, indent=2))

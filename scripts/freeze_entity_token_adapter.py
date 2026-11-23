"""Freeze an isolated readout adapter after CPU correctness checks pass."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha
DISPATCH=ROOT/'artifacts/entity_token_adapter_20261122'


def main():
    previous=read(ROOT/'artifacts/hypothesis_evidence_cv5_seed0_20261122/frozen/manifest.json')
    for kind in ['source','inputs']:
        assert all(sha(p)==digest for p,digest in previous[kind].items()),kind
    log=(DISPATCH/'unit_tests.log').read_text()
    assert 'Ran 5 tests' in log and '\nOK\n' in log
    checks={ds:read(DISPATCH/('cpu_'+ds)/'CHECKS.json') for ds in ['ALMANAC','ONEIL']}
    assert all(c['status']=='passed' and c['optimizer_updates']==2 for c in checks.values())
    names=['dual_branch/entity_token_adapter.py','dual_branch/adapted_readout_cli.py',
        'tests/test_entity_token_adapter.py','scripts/check_entity_token_adapter.py',
        'scripts/freeze_entity_token_adapter.py','ENTITY_TOKEN_ADAPTER.md']
    source=dict(previous['source'])
    source.update({str(ROOT/n):sha(ROOT/n) for n in names})
    target=DISPATCH/'frozen'
    assert not target.exists()
    for p in source:
        path=Path(p)
        relative=path.relative_to(ROOT) if str(path).startswith(str(ROOT)+'/') else Path('external')/str(path).lstrip('/')
        destination=target/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    write(target/'manifest.json',dict(source=source,inputs=previous['inputs'],
        version=DISPATCH.name,previous113_training_sources_and62_inputs_unchanged=True))
    write(DISPATCH/'UNIT_CHECKS.json',dict(status='passed',tests=5,log=str(DISPATCH/'unit_tests.log')))
    write(DISPATCH/'VERIFIED.json',dict(status='CPU_verified',unit_tests=5,source_files=len(source),
        input_files=len(previous['inputs']),CPU_checks=checks,source_manifest=str(target/'manifest.json'),
        current_formal_runs_changed=False,new_GPU_jobs=0,no_performance_claim=True))
    write(ROOT/'ENTITY_TOKEN_ADAPTER_VERSION.json',dict(version=DISPATCH.name,status='CPU_verified',
        model_class='dual_branch.entity_token_adapter.EntityAdaptedHypothesisModel',
        training_entry='dual_branch.adapted_readout_cli',code=str(ROOT/'dual_branch/entity_token_adapter.py'),
        source_manifest=str(target/'manifest.json'),current_fivefold_uses_this_model=False,
        formal_GPU_training_started=False))
    print(dict(status='CPU_verified',unit_tests=5,source_files=len(source),input_files=len(previous['inputs'])))

if __name__=='__main__':main()

"""Freeze the already audited model plus its new formal training adapter."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_cell.common import sha, read, write
from dual_branch.hypothesis_common import DISPATCH

def main():
    prior=read(ROOT/'artifacts/hypothesis_evidence_20261122/frozen/manifest.json')
    for kind in ['source','inputs']:
        assert all(sha(p)==s for p,s in prior[kind].items()),kind
    source=dict(prior['source'])
    names=['dual_branch/hypothesis_cli.py','dual_branch/hypothesis_common.py','dual_branch/hypothesis_checks.py',
        'scripts/enqueue_hypothesis_evidence_cv5.py','scripts/report_hypothesis_evidence_cv5.py',
        'scripts/freeze_hypothesis_evidence_cv5.py','HYPOTHESIS_EVIDENCE_CV5.md']
    paths=[ROOT/n for n in names]+[ROOT.parent/'mcdsp_moe/MCDSP-main/tools/paper_routes/train_hermes_hypothesis_evidence_cv5.py']
    source.update({str(p):sha(p) for p in paths})
    target=DISPATCH/'frozen'
    assert not target.exists()
    for p in source:
        path=Path(p)
        relative=path.relative_to(ROOT) if str(path).startswith(str(ROOT)+'/') else Path('external')/str(path).lstrip('/')
        dest=target/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    write(target/'manifest.json',dict(source=source,inputs=prior['inputs'],version=DISPATCH.name,
        previous105_sources_and62_inputs_unchanged=True,from_scratch=True))
    print(dict(source_files=len(source),input_files=len(prior['inputs'])))

if __name__=='__main__':
    main()

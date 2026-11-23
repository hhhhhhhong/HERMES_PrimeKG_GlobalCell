"""Freeze the checked adapter and its isolated fivefold launch support."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from global_cell.common import read,write,sha
from dual_branch.entity_adapter_common import DISPATCH,CPU_CHECKS

def main():
    prior=read(CPU_CHECKS/'frozen/manifest.json')
    assert read(CPU_CHECKS/'VERIFIED.json')['status']=='CPU_verified'
    for kind in ['source','inputs']:
        assert all(sha(p)==digest for p,digest in prior[kind].items()),kind
    names=['dual_branch/entity_adapter_common.py','scripts/enqueue_entity_adapter_cv5.py',
        'scripts/report_entity_adapter_cv5.py','scripts/freeze_entity_adapter_cv5.py','ENTITY_TOKEN_ADAPTER_CV5.md']
    source=dict(prior['source']);paths=[ROOT/n for n in names]
    paths.append(ROOT.parent/'mcdsp_moe/MCDSP-main/tools/paper_routes/train_hermes_entity_adapter_cv5.py')
    source.update({str(p):sha(p) for p in paths})
    target=DISPATCH/'frozen';assert not target.exists()
    for p in source:
        path=Path(p);relative=path.relative_to(ROOT) if str(path).startswith(str(ROOT)+'/') else Path('external')/str(path).lstrip('/')
        dest=target/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(target/'manifest.json',dict(source=source,inputs=prior['inputs'],version=DISPATCH.name,
        previous119_sources_and62_inputs_unchanged=True,from_scratch=True))
    print(dict(source_files=len(source),input_files=len(prior['inputs'])))

if __name__=='__main__':main()

"""Freeze this version's sources/inputs without changing any historical suite."""
from pathlib import Path
import shutil
from .common import ROOT, STRUCTURE, DISPATCH, DATA
from global_cell.common import read, write, sha


def main():
    assert read(DISPATCH / 'PREPARATION.json')['status'] == 'complete'
    assert read(DISPATCH / 'UNIT_CHECKS.json')['status'] == 'passed'
    assert not (DISPATCH / 'frozen/manifest.json').exists()
    # The user's old fits continue to execute exactly their frozen versions.
    for name in ['zero_gated_hgnn3_cv5_seed0_20261118',
                 'knowledge_decay_ALMANAC_fold0_seed0_20261119',
                 'knowledge_decay_ONEIL_cv5_seed0_20261119']:
        old = read(ROOT / 'artifacts' / name / 'frozen/source_manifest.json')
        assert all(sha(ROOT / p) == digest for p, digest in old['package'].items())
        assert all(sha(p) == digest for p, digest in old['extra'].items())
    source = list((ROOT / 'dual_branch').glob('*.py')) + list((ROOT / 'global_cell').glob('*.py'))
    source += [ROOT / 'models/zero_gated_hgnn.py', ROOT / 'tests/test_dual_branch.py']
    source += [ROOT.parent / 'mcdsp_moe/MCDSP-main/tools/paper_routes/train_hermes_dual_branch.py']
    source += [STRUCTURE / name for name in ['drug_init_2d3d/__init__.py', 'drug_init_2d3d/model.py',
        'structure_residual/__init__.py', 'structure_residual/model.py',
        'structure_residual/chemistry.py', 'structure_residual/common.py']]
    source += list((ROOT / 'vendor/hermes/Model/HERMES').glob('*.py'))
    source += [ROOT / 'vendor/hermes/Model/utils.py', ROOT / 'vendor/hermes/Model/drug_util.py']
    inputs = [DISPATCH / 'PREPARATION.json', DATA / 'cell_annotations.json',
        DISPATCH / 'CELL_ANNOTATION_AUDIT.json', ROOT / 'artifacts/CODER_DOWNLOAD.json', ROOT / 'artifacts/ENCODING_AUDIT.json']
    for dataset in ['ALMANAC', 'ONEIL']:
        inputs += [DATA / dataset / 'all_nodes_coder.pt', ROOT / 'data' / dataset / 'graph.pt']
        inputs += [ROOT / 'data' / dataset / ('fold%d' % fold) / 'inputs.pt' for fold in range(5)]
        inputs += [STRUCTURE / 'artifacts' / kind / (dataset + suffix)
            for kind in ['chemistry', 'unimol'] for suffix in ['.pt', '.json']]
        inputs += [ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / name
            for name in ['drug_smiles.csv', 'cell line_gene_expression.csv']]
    manifest = dict(source={str(p): sha(p) for p in sorted(set(source))},
        inputs={str(p): sha(p) for p in sorted(set(inputs))}, version='dual_branch_v1_20261119',
        historical_frozen_sources_unchanged=True)
    write(DISPATCH / 'frozen/manifest.json', manifest)
    for path in source:
        if ROOT in path.parents:
            relative = 'project/' + str(path.relative_to(ROOT))
        elif STRUCTURE in path.parents:
            relative = 'structure_reference/' + str(path.relative_to(STRUCTURE))
        else:
            relative = 'bridge/' + path.name
        destination = DISPATCH / 'frozen/source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(dict(source_files=len(manifest['source']), input_files=len(manifest['inputs']),
        historical_frozen_sources_unchanged=True))


if __name__ == '__main__':
    main()

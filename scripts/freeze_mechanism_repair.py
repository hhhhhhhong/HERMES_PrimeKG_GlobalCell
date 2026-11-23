"""Freeze the new implementation and unchanged dependencies, never checkpoints."""
from pathlib import Path
import json
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / 'artifacts/mechanism_repair_20261122'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    destination = DISPATCH / 'frozen'
    assert not destination.exists(), 'Keep each earlier frozen attempt'
    previous = json.loads((ROOT / 'artifacts/kpaths_cv5_seed0_20261120/frozen/manifest.json').read_text())
    assert all(sha(p) == s for p, s in previous['source'].items())
    paths = set(previous['source'])
    paths.update(str(p) for p in (ROOT / 'dual_branch').glob('mechanism_*.py'))
    paths.update(str(ROOT / p) for p in ['dual_branch/multimodal_fusion.py',
        'scripts/prepare_mechanism_repair.py', 'scripts/freeze_mechanism_repair.py', 'MECHANISM_REPAIR.md'])
    paths.update(str(p) for p in (ROOT / 'tests').glob('test_mechanism_*.py'))
    source = {p: sha(p) for p in sorted(paths)}
    inputs = dict(previous['inputs'])
    for dataset in ['ALMANAC', 'ONEIL']:
        folder = ROOT / 'data/mechanism_repair_20261122' / dataset
        for name in ['graph.pt', 'paths.pt', 'all_nodes_coder.pt', 'paths.jsonl', 'PROVENANCE.json']:
            inputs[str(folder / name)] = sha(folder / name)
    for p in source:
        path = Path(p)
        relative = path.relative_to(ROOT) if str(path).startswith(str(ROOT) + '/') else Path('external') / str(path).lstrip('/')
        target = destination / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (destination / 'manifest.json').write_text(json.dumps(dict(source=source, inputs=inputs,
        version='mechanism_multimodal_repair_20261122', old64_training_sources_unchanged=True), indent=2) + '\n')
    print(dict(source_files=len(source), input_files=len(inputs), frozen=str(destination)), flush=True)


if __name__ == '__main__':
    main()

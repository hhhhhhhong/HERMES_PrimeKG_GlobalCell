"""Freeze gate-only changes and the unchanged mechanism dependencies/inputs."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / 'artifacts/gated_mechanism_cv5_seed0_20261122'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    previous = json.loads((ROOT / 'artifacts/mechanism_repair_20261122/frozen/manifest.json').read_text())
    assert all(sha(p) == s for p, s in previous['source'].items())
    assert all(sha(p) == s for p, s in previous['inputs'].items())
    source = dict(previous['source'])
    names = ['dual_branch/gated_mechanism.py', 'dual_branch/gated_cli.py', 'dual_branch/gated_checks.py',
        'dual_branch/gated_common.py', 'tests/test_gated_mechanism.py',
        'scripts/freeze_gated_mechanism_cv5.py', 'scripts/enqueue_gated_mechanism_cv5.py',
        'scripts/report_gated_mechanism_cv5.py', 'GATED_MECHANISM_CV5.md']
    paths = [ROOT / n for n in names]
    paths.append(ROOT.parent / 'mcdsp_moe/MCDSP-main/tools/paper_routes/train_hermes_gated_mechanism_cv5.py')
    source.update({str(p): sha(p) for p in paths})
    target = DISPATCH / 'frozen'
    assert not target.exists(), 'Preserve earlier frozen attempts'
    for p in source:
        path = Path(p)
        relative = path.relative_to(ROOT) if str(path).startswith(str(ROOT) + '/') else Path('external') / str(path).lstrip('/')
        destination = target / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    manifest = dict(source=source, inputs=previous['inputs'], version='gated_mechanism_cv5_seed0_20261122',
                    previous75_training_sources_unchanged=True, from_scratch=True)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(dict(source_files=len(source), input_files=len(previous['inputs']), manifest=str(target / 'manifest.json')))


if __name__ == '__main__':
    main()

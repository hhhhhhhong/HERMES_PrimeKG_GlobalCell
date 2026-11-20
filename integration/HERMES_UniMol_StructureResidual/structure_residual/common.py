"""Local immutable inputs, provenance checks and atomic output helpers."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/ifs2/home/liushuhong/work2/mcdsp_moe/experiments/hermes_baseline_cv5_20261113')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'structure_residual').glob('*.py')}


def author_imports():
    vendor = ROOT / 'vendor/hermes'
    manifest = read(vendor / 'PROVENANCE.json')
    assert all(sha(vendor / p) == digest for p, digest in manifest['files_sha256'].items())
    sys.path.insert(0, str(vendor / 'Model'))
    result = {name: importlib.import_module(name) for name in ['utils', 'drug_util', 'HERMES.main', 'HERMES.model']}
    assert all(str(Path(m.__file__).resolve()).startswith(str(vendor) + '/') for m in result.values())
    return result

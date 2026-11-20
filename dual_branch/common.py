"""Version-local paths and provenance; historical experiments stay immutable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = ROOT.parent / 'HERMES_UniMol_StructureResidual'
DISPATCH = ROOT / 'artifacts/dual_branch_v1_20261119'
DATA = ROOT / 'data/dual_branch_v1_20261119'
RUNTIME = ROOT.parent / 'runtime/paper_routes_20261009'


def environment():
    """Call before transformers/torch imports, including CPU-only preparation."""
    for name, suffix in dict(TMPDIR='tmp', HF_HOME='cache/huggingface', TORCH_HOME='cache/torch',
        XDG_CACHE_HOME='cache/xdg', WANDB_DIR='wandb', WANDB_CACHE_DIR='wandb_cache',
        WANDB_DATA_DIR='wandb_data', WANDB_CONFIG_DIR='wandb_config').items():
        path = RUNTIME / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def structural_imports():
    import sys
    sys.path.insert(0, str(STRUCTURE))
    from drug_init_2d3d.model import Drug2D3DEncoder
    return Drug2D3DEncoder

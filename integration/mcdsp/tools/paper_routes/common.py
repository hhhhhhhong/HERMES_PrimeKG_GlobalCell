import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
BASELINE = Path('/ifs/data/liushuhong/work2/results/graph_semantic_followup_20261008')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def tensor_hash(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def graph_data(dataset, labelled=False):
    from Utils.graph_constrct import get_graph
    os.chdir(PROJECT)
    seed(0)
    graph, mapping, synergy = get_graph(dataset, node_dim=300, load_synergy=labelled,
        write_mapping=False, deterministic_protein_order=True)
    record = dict(nodes=graph.num_nodes, edges=graph.num_edges,
        x_sha256=tensor_hash(graph.x), edge_index_sha256=tensor_hash(graph.edge_index),
        edge_type_sha256=tensor_hash(graph.edge_type),
        mapping_sha256=hashlib.sha256(json.dumps(sorted((str(k), int(v)) for k, v in mapping.items())).encode()).hexdigest())
    return graph, mapping, synergy, record


def load_task(dataset, device):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    from Utils.feature_utils import drug_cline_feature
    from main_KG_DualGranularity import load_samples, validate_feature_alignment, outer_split, iter_folds
    graph, mapping, synergy, record = graph_data(dataset, labelled=True)
    cells, drugs = drug_cline_feature(mapping, dataset)
    validate_feature_alignment(graph.num_cells, graph.num_drugs, drugs, cells)
    index, strings, labels = load_samples(dataset, mapping)
    index[:, :2].sort(axis=1)
    check = synergy[:, :3].copy()
    check[:, :2].sort(axis=1)
    assert np.array_equal(index, check) and np.array_equal(labels, synergy[:, 3])
    development, test = outer_split(len(labels), 0)
    folds = [dict(fold=i, train=t, validation=v) for i, (t, v) in
             enumerate(iter_folds(development, strings, 1, 5, 0))]
    manifest = dict(folds=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in f.items()} for f in folds],
        excluded_test_rows=test.tolist(), test_policy='sealed; no test predictions')
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    old = json.loads((BASELINE / dataset / 'seed0/paper/config.json').read_text())
    assert old['split_sha256'] == signature and old['graph'] == record, 'unmatched baseline inputs'
    keys = np.asarray(['\t'.join(sorted(q[:2])) + '\t' + q[2] for q in strings])
    return dict(graph=graph.to(device), mapping=mapping, index=torch.as_tensor(index, device=device),
        cells=cells.to(device), drugs=drugs.to(device), y=labels,
        labels=torch.as_tensor(labels, device=device), strings=strings, keys=keys,
        folds=folds, test=test, split_manifest=manifest, split_sha256=signature,
        graph_record=record, baseline_config=old)


def scores(labels, probability, ranking=None):
    from tools.run_fact_pooling_cv1 import metrics
    from sklearn.metrics import f1_score
    # AUC/AP need a continuous decision score. Large class-subgraph dot products
    # can saturate float32 softmax to exactly 0/1 and introduce artificial ties.
    result = metrics(labels, probability if ranking is None else ranking)
    result['acc'] = float(np.mean((np.asarray(probability) >= .5) == np.asarray(labels)))
    result['f1'] = float(f1_score(labels, probability >= .5, zero_division=0))
    return result


def snapshot_sources(output):
    import shutil
    paths = list((PROJECT / 'tools/paper_routes').glob('*.py'))
    paths += [PROJECT / 'Model' / name for name in ['rgcn_model.py', 'Multimodal_OncologyScreen.py',
        'graph_semantic_i2moe.py', 'modality_training_recipe.py']]
    result = {}
    for path in paths:
        rel = path.relative_to(PROJECT)
        dest = Path(output) / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        result[str(rel)] = digest(path)
    return result

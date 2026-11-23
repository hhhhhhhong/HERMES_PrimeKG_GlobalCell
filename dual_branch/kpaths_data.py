"""Fold-isolated supervision on a fixed label-free path-selected global graph."""
from .common import ROOT
from global_cell.common import read, sha

PATH_DATA = ROOT / 'data/kpaths_cv5_20261120'
DISPATCH = ROOT / 'artifacts/kpaths_cv5_seed0_20261120'
RUN = ROOT / 'runs/kpaths_cv5_seed0_20261120'


def load(dataset, fold):
    import torch
    from .data import load as original_load
    from global_cell.synergy import add_training_synergy
    data = original_load(dataset, fold)
    folder = PATH_DATA / dataset
    provenance = read(folder / 'PROVENANCE.json')
    for name, signature in provenance['files_sha256'].items():
        assert sha(folder / name) == signature, name
    graph = torch.load(folder / 'graph.pt', map_location='cpu', weights_only=True)
    features = torch.load(folder / 'all_nodes_coder.pt', map_location='cpu', weights_only=True)
    assert torch.equal(features[:len(data['semantic'])], data['semantic'])
    assert graph['cids'] == data['graph']['cids'] and graph['cell_names'] == data['graph']['cell_names']
    assert graph['groups']['cell_expression'] == data['graph']['groups']['cell_expression']
    graph = add_training_synergy(graph, data['raw']['fit']['rows'], fold=fold,
        expected_train_sha256=data['raw']['metadata']['split']['train_array_sha256'])
    assert graph['groups']['Synergy_Hyperedge'] == data['graph']['groups']['Synergy_Hyperedge']
    data.update(graph=graph, semantic=features,
        paths=torch.load(folder / 'paths.pt', map_location='cpu', weights_only=True))
    data['input_provenance'] = dict(data['input_provenance'], path_cache=provenance,
        graph_sha256=sha(folder / 'graph.pt'), semantic_sha256=sha(folder / 'all_nodes_coder.pt'))
    return data

"""Independent new path/graph cache; original splits and modalities retained."""
from .common import ROOT
from global_cell.common import read, sha

CACHE = ROOT / 'data/cell_conditioned_retrieval_20261122'


def load(dataset, fold=0):
    import torch
    from .mechanism_data import load as original_load
    from global_cell.synergy import add_training_synergy
    data = original_load(dataset, fold)
    folder = CACHE / dataset
    provenance = read(folder / 'PROVENANCE.json')
    assert all(sha(folder / name) == value for name, value in provenance['files_sha256'].items())
    graph = torch.load(folder / 'graph.pt', map_location='cpu', weights_only=True)
    assert graph['external_prime_indices'] == data['graph']['external_prime_indices']
    assert graph['groups']['cell_expression'] == data['graph']['groups']['cell_expression']
    assert graph['cell_names'] == data['graph']['cell_names']
    graph = add_training_synergy(graph, data['raw']['fit']['rows'], fold=fold)
    assert graph['groups']['Synergy_Hyperedge'] == data['graph']['groups']['Synergy_Hyperedge']
    data.update(graph=graph, paths=torch.load(folder / 'paths.pt', map_location='cpu', weights_only=True))
    data['input_provenance'] = dict(data['input_provenance'], Cell_conditioned_retrieval=provenance)
    return data

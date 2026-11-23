"""Verified new mechanism facts with original fold rows and unchanged Cell input."""
import hashlib
from .common import ROOT
from global_cell.common import read, sha

CACHE = ROOT / 'data/mechanism_repair_20261122'


def load(dataset, fold):
    import torch
    from .data import load as original_load
    from global_cell.synergy import add_training_synergy
    data = original_load(dataset, fold)
    folder = CACHE / dataset
    provenance = read(folder / 'PROVENANCE.json')
    assert all(sha(folder / name) == expected for name, expected in provenance['files_sha256'].items())
    graph = torch.load(folder / 'graph.pt', map_location='cpu', weights_only=True)
    semantic = torch.load(folder / 'all_nodes_coder.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(semantic[:len(data['semantic'])], data['semantic'], rtol=0, atol=0)
    assert graph['cids'] == data['graph']['cids'] and graph['cell_names'] == data['graph']['cell_names']
    assert graph['groups']['cell_expression'] == data['graph']['groups']['cell_expression']
    for part, key in [('fit', 'train'), ('valid', 'valid')]:
        actual = hashlib.sha256(data['raw'][part]['rows'].numpy().tobytes()).hexdigest()
        assert actual == data['raw']['metadata']['split'][key + '_array_sha256']
    graph = add_training_synergy(graph, data['raw']['fit']['rows'], fold=fold)
    assert graph['groups']['Synergy_Hyperedge'] == data['graph']['groups']['Synergy_Hyperedge']
    data.update(graph=graph, semantic=semantic,
                paths=torch.load(folder / 'paths.pt', map_location='cpu', weights_only=True))
    data['input_provenance'] = dict(data['input_provenance'], mechanism_cache=provenance,
        mechanism_graph_sha256=sha(folder / 'graph.pt'), mechanism_semantic_sha256=sha(folder / 'all_nodes_coder.pt'))
    return data

"""Original fold rows, typed graph, and independently aligned input modalities."""
import csv
from .common import ROOT, STRUCTURE, DATA
from global_cell.common import read, sha


def entity_records(dataset, graph):
    """Dataset names are evidence; unverified tissue/disease text is not invented."""
    drug_file = ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / 'drug_smiles.csv'
    cell_file = ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / 'cell line_gene_expression.csv'
    with drug_file.open() as stream:
        drugs = list(csv.DictReader(stream))
    with cell_file.open() as stream:
        cells = list(csv.reader(stream))[1:]
    ids = [str(row['pubchemid']) for row in drugs]
    assert ids == [str(cid) for cid in graph['cids']]
    names = [row[0] for row in cells]
    assert names == graph['cell_names']
    records = [dict(node=i, key='pubchem:' + cid, kind='drug', text=row['name'],
        source=str(drug_file), source_sha256=sha(drug_file)) for i, (cid, row) in enumerate(zip(ids, drugs))]
    records += [dict(node=len(drugs) + i, key='cell:' + name, kind='cell', text=name,
        source=str(cell_file), source_sha256=sha(cell_file), name_only=True) for i, name in enumerate(names)]
    annotation_file = DATA / 'cell_annotations.json'
    annotations = read(annotation_file) if annotation_file.exists() else {}
    for record in records[len(drugs):]:
        annotation = annotations.get(record['text'], {})
        if annotation.get('status') == 'unique':
            parts = [record['text'] + ' cell line', 'Species: Homo sapiens']
            for key, title in [('disease_labels', 'Disease'), ('tissue_labels', 'Tissue'), ('cell_type_labels', 'Cell type')]:
                if annotation.get(key):
                    parts.append(title + ': ' + '; '.join(annotation[key][:2]))
            record.update(text='. '.join(parts), name_only=False, annotation=annotation,
                annotation_file_sha256=sha(annotation_file))
        else:
            record['annotation_status'] = annotation.get('status', 'not_available')
    records += [dict(node=graph['n_base'] + i, key='primekg_index:' + str(index),
        kind=node['node_type'], text=node['node_name'], source='audited PrimeKG node_name')
        for i, (index, node) in enumerate(zip(graph['external_prime_indices'], graph['external_nodes']))]
    assert len(records) == len(graph['node_types'])
    assert [row['node'] for row in records] == list(range(len(records)))
    assert all(row['text'].strip() for row in records)
    return records


def load(dataset, fold):
    """Read-only, label-isolated construction. Never open an old neural checkpoint."""
    import torch
    from global_cell.synergy import add_training_synergy
    raw = torch.load(ROOT / 'data' / dataset / ('fold%d' % fold) / 'inputs.pt',
        map_location='cpu', weights_only=True)
    graph_path = ROOT / 'data' / dataset / 'graph.pt'
    graph = torch.load(graph_path, map_location='cpu', weights_only=True)
    semantic = torch.load(DATA / dataset / 'all_nodes_coder.pt', map_location='cpu', weights_only=True)
    assert semantic['graph_sha256'] == sha(graph_path)
    assert semantic['records'] == entity_records(dataset, graph)
    assert semantic['features'].shape == (len(graph['node_types']), 768)
    chemistry_path = STRUCTURE / 'artifacts/chemistry' / (dataset + '.pt')
    unimol_path = STRUCTURE / 'artifacts/unimol' / (dataset + '.pt')
    chemistry_meta = read(chemistry_path.with_suffix('.json'))
    unimol_meta = read(unimol_path.with_suffix('.json'))
    ids = [str(cid) for cid in graph['cids']]
    assert ids == chemistry_meta['drug_ids'] == unimol_meta['drug_ids']
    assert sha(ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / 'drug_smiles.csv') == chemistry_meta['input_sha256']
    assert sha(unimol_path) == unimol_meta['embeddings_sha256']
    assert sha(chemistry_path) == semantic['chemistry_sha256']
    assert sha(unimol_path) == semantic['unimol_sha256']
    chemistry = torch.load(chemistry_path, map_location='cpu', weights_only=True)
    unimol = torch.load(unimol_path, map_location='cpu', weights_only=True)
    assert len(chemistry['drug_heavy']) == raw['num_drug'] == graph['n_drug']
    assert len(chemistry['fragment_drug']) == len(unimol['fragment_3d'])
    assert raw['inputs']['cells'].shape == (raw['num_cell'], raw['cell_input_dim'])
    assert raw['num_drug'] + raw['num_cell'] == graph['n_base']
    graph = add_training_synergy(graph, raw['fit']['rows'], fold=fold,
        expected_train_sha256=raw['metadata']['split']['train_array_sha256'])
    return dict(raw=raw, graph=graph, semantic=semantic['features'], chemistry=chemistry, unimol=unimol,
        input_provenance=dict(semantic_sha256=sha(DATA / dataset / 'all_nodes_coder.pt'),
            graph_sha256=sha(graph_path), chemistry_sha256=sha(chemistry_path), unimol_sha256=sha(unimol_path),
            original_split=raw['metadata']['split'], no_checkpoint_loaded=True, test_evaluated=False))

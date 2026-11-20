"""Offline CODER inputs for every node, with root identity/order provenance."""
import os
from .common import ROOT, STRUCTURE, DISPATCH, DATA, environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def build_tokenizer():
    """Official vocabulary + explicit uncased tokenization, not the old local override.

    The upstream pinned model has vocab.txt but no tokenizer_config.json.
    The old local do_lower_case=False override collapsed many names to UNK.
    Author UMLS preprocessing lowercases terms; retain the original text in
    provenance while the tokenizer performs the matching normalization.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / 'models/coder_eng', local_files_only=True, do_lower_case=True)


def main():
    import hashlib
    import json
    import time
    import torch
    from transformers import AutoModel
    from global_cell.common import read, write, sha
    from .data import entity_records
    torch.set_num_threads(2)
    assert not (DISPATCH / 'PREPARATION.json').exists(), 'Preserve this versioned cache'
    assert read(DISPATCH / 'CELL_ANNOTATION_AUDIT.json')['status'] == 'complete'
    download = read(ROOT / 'artifacts/CODER_DOWNLOAD.json')
    model_path = ROOT / 'models/coder_eng'
    assert all(sha(model_path / name) == digest for name, digest in download['files'].items())
    graphs, records, vectors = {}, {}, {}
    for dataset in ['ALMANAC', 'ONEIL']:
        graph = torch.load(ROOT / 'data' / dataset / 'graph.pt', map_location='cpu', weights_only=True)
        graphs[dataset] = graph
        records[dataset] = entity_records(dataset, graph)
    # Corrected tokenization changes the inputs. Re-encode every unique text;
    # never mix old cased external vectors with newly corrected root vectors.
    needed = sorted({row['text'] for rows in records.values() for row in rows})
    tokenizer = build_tokenizer()
    assert tokenizer.tokenize('Methotrexate') == tokenizer.tokenize('methotrexate')
    assert tokenizer.unk_token not in tokenizer.tokenize('Methotrexate')
    assert tokenizer.encode('A549') != tokenizer.encode('A498')
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    # The published CODER bundle has 30,522 embedding rows and 28,895
    # vocabulary entries. Extra unused rows are valid; no resizing/remapping.
    assert max(tokenizer.get_vocab().values()) < encoder.get_input_embeddings().num_embeddings
    truncated = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(needed), 32):
            texts = needed[start:start + 32]
            for text in texts:
                if len(tokenizer.encode(text, add_special_tokens=True)) > 128:
                    truncated.append(text)
            signature = hashlib.sha256(json.dumps([download['revision'], True, 128, texts]).encode()).hexdigest()
            cached = DATA / 'corrected_batches' / (signature + '.pt')
            if cached.exists():
                features = torch.load(cached, map_location='cpu', weights_only=True)
            else:
                batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                features = torch.nn.functional.normalize(encoder(**batch).pooler_output, dim=-1).cpu()
                cached.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features, cached)
            for text, vector in zip(texts, features):
                vectors[text] = vector
            progress = dict(status='running', encoded=min(start + 32, len(needed)), total_new_texts=len(needed),
                seconds=time.monotonic() - started, do_lower_case=True, max_length=128)
            write(DISPATCH / 'ENCODING_PROGRESS.json', progress)
            if start % 320 == 0 or start + 32 >= len(needed):
                print(progress, flush=True)
    del encoder
    summaries = {}
    for dataset, rows in records.items():
        graph = graphs[dataset]
        features = torch.stack([vectors[row['text']] for row in rows])
        assert torch.isfinite(features).all()
        torch.testing.assert_close(features.norm(dim=-1), torch.ones(len(features)), rtol=0, atol=2e-5)
        chemistry = STRUCTURE / 'artifacts/chemistry' / (dataset + '.pt')
        unimol = STRUCTURE / 'artifacts/unimol' / (dataset + '.pt')
        target = DATA / dataset / 'all_nodes_coder.pt'
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(features=features, records=rows, graph_sha256=sha(ROOT / 'data' / dataset / 'graph.pt'),
            chemistry_sha256=sha(chemistry), unimol_sha256=sha(unimol)), target)
        drug = features[:graph['n_drug']]
        cells = features[graph['n_drug']:graph['n_base']]
        summaries[dataset] = dict(nodes=len(features), drug_nodes=len(drug), cell_nodes=len(cells),
            external_nodes=len(features) - graph['n_base'], all_nodes_CODER=True, file_sha256=sha(target),
            unique_cell_vectors=len({hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in cells}),
            cell_text='dataset name plus exact-matched official Cellosaurus disease/tissue where available',
            Cellosaurus_annotated_cells=sum(r['kind'] == 'cell' and not r.get('name_only', True) for r in rows),
            drug_text='original dataset drug name; PubChem CID used for alignment only',
            unimol_coverage=read(unimol.with_suffix('.json')),
            input_order_exact=True, synergy_labels_used_for_encoding=False)
        write(DATA / dataset / 'ENTITY_TEXTS.json', rows)
    write(DISPATCH / 'PREPARATION.json', dict(status='complete', model=download,
        pooling='pooler_output + L2; unchanged fixed CODER weights, corrected uncased tokenizer', max_tokens=128,
        newly_encoded_unique_texts=len(needed), truncated_texts=truncated, datasets=summaries,
        do_lower_case=True, official_external_name_inputs_reused=False, all_inputs_reencoded=True,
        pretrained_CODER_not_finetuned=True,
        vocabulary_entries=len(tokenizer), embedding_rows=30522,
        new_neural_checkpoint_loaded=False, test_evaluated=False))
    write(DISPATCH / 'ENCODING_PROGRESS.json', dict(progress, status='complete'))


if __name__ == '__main__':
    main()

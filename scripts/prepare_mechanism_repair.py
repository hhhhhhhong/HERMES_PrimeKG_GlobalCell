"""CPU-only, label-free mechanism cache with PPI bridges and single-drug facts."""
import copy
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import DATA, environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha
from dual_branch.kpaths_retrieval import FUNCTIONS
from dual_branch.mechanism_retrieval import MechanismIndex

DESTINATION = ROOT / 'data/mechanism_repair_20261122'
DISPATCH = ROOT / 'artifacts/mechanism_repair_20261122'


def main():
    import torch
    torch.set_num_threads(2)
    assert not (DISPATCH / 'PREPARATION.json').exists(), 'Preserve existing preparation'
    source_dir = ROOT / 'artifacts/kpaths_reference_review_20261120/raw_primekg_audit'
    source = read(source_dir / 'INDEX.json')
    assert sha(source_dir / 'index_facts.jsonl') == source['index_facts_sha256']
    facts = []
    with (source_dir / 'index_facts.jsonl').open() as stream:
        for line in stream:
            fact = json.loads(line)
            if fact['family'] in FUNCTIONS | {'drug_target', 'protein_protein'}:
                facts.append(fact)
    nodes = {int(k): v for k, v in read(source_dir / 'index_nodes.json').items()}
    index = MechanismIndex(facts, {i: n['node_type'] for i, n in nodes.items()})
    mapping_file = ROOT.parent / 'HERMES_PrimeKG_HHNN/data/raw/drug_mapping.json'
    mapping = {x['cid']: x['best']['node'] for x in read(mapping_file) if x['matched']}
    drugbank = {n['node_id']: i for i, n in nodes.items() if n['node_type'] == 'drug'}
    pending, vectors, summaries = {}, {}, {}
    for dataset in ['ALMANAC', 'ONEIL']:
        folder = DESTINATION / dataset
        assert not folder.exists(), 'Use distinct paths for failed attempts'
        folder.mkdir(parents=True)
        graph = copy.deepcopy(torch.load(ROOT / 'data' / dataset / 'graph.pt', map_location='cpu', weights_only=True))
        semantic = torch.load(DATA / dataset / 'all_nodes_coder.pt', map_location='cpu', weights_only=True)
        for record, vector in zip(semantic['records'], semantic['features']):
            vectors.setdefault(record['text'], vector)
        drug_file = ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / 'drug_smiles.csv'
        with drug_file.open() as stream:
            drugs = list(csv.DictReader(stream))
        assert [str(x['pubchemid']) for x in drugs] == [str(x) for x in graph['cids']]
        prime = [drugbank.get(mapping.get(str(x['pubchemid']), '')) for x in drugs]
        drug_local = {p: i for i, p in enumerate(prime) if p is not None}
        assert len(drug_local) == sum(p is not None for p in prime)
        old_folder = ROOT / 'data/kpaths_cv5_20261120' / dataset
        previous = read(old_folder / 'PROVENANCE.json')
        assert sha(old_folder / 'paths.jsonl') == previous['files_sha256']['paths.jsonl']
        records = []
        with (old_folder / 'paths.jsonl').open() as stream:
            for line in stream:
                old = json.loads(line)
                for path in old['paths']:
                    assert [facts[i]['row'] for i in path['fact_ids']] == path['source_rows']
                result = index.extend(*old['prime_indices'], old['paths'])
                records.append(dict(drug_indices=old['drug_indices'], prime_indices=old['prime_indices'],
                    previous_path_count=len(old['paths']), **result))
        with (folder / 'paths.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        needed = {fid for q in records for p in q['paths'] for fid in p['fact_ids']}
        # Direct facts remain accessible when this pair has no connecting
        # mechanism. These are rooted annotations, never invented pair paths.
        targets = set()
        for drug in drug_local:
            for protein, fid in index.targets.get(drug, {}).items():
                needed.add(fid)
                targets.add(protein)
        annotation_context = set()
        for protein in sorted(targets):
            by_family = defaultdict(list)
            for anchor in index.function_neighbors[protein]:
                size = len(index.function_members[anchor])
                if size <= 200:
                    fid = index.function_fact[anchor, protein]
                    by_family[facts[fid]['family']].append((size, anchor, fid))
            for family in by_family:
                annotation_context.update(fid for size, anchor, fid in sorted(by_family[family])[:4])
        needed.update(annotation_context)
        node_local = {int(p): graph['n_base'] + i for i, p in enumerate(graph['external_prime_indices'])}
        requested = {v for fid in needed for v in (facts[fid]['head'], facts[fid]['tail']) if nodes[v]['node_type'] != 'drug'}
        extra = sorted(requested - set(node_local))
        original_size = len(graph['node_types'])
        for p in extra:
            node_local[p] = len(graph['node_types'])
            graph['node_types'] = torch.cat((graph['node_types'], torch.tensor([graph['type_names'].index(nodes[p]['node_type'])])))
            graph['external_prime_indices'].append(p)
            graph['external_nodes'].append(nodes[p])
        groups = defaultdict(list)
        groups['cell_expression'] = graph['groups']['cell_expression']
        routes = {int(k): v for k, v in graph['routes'].items() if int(k) >= graph['n_drug']}
        functions = defaultdict(dict)
        for fid in sorted(needed):
            f = facts[fid]
            family, h, t = f['family'], f['head'], f['tail']
            if family in FUNCTIONS:
                functions[family, h][t] = f
                continue
            if family == 'drug_target':
                if h not in drug_local:
                    continue
                a, b = drug_local[h], node_local[t]
                routes.setdefault(a, []).append(dict(protein=b, relation=family,
                    proof=dict(raw_row=f['row'], raw_primekg_endpoints=[h, t])))
            else:
                a, b = node_local[h], node_local[t]
            groups[family].append(dict(members=[a, b], positions=[0, 1], weight=1.,
                proof=dict(raw_row=f['row'], raw_primekg_endpoints=[h, t])))
        for (family, anchor), members in sorted(functions.items()):
            proteins = sorted(members)
            groups[family].append(dict(members=[node_local[anchor]] + [node_local[p] for p in proteins],
                positions=[0] + [1] * len(proteins), weight=1.,
                proof=dict(raw_rows=[members[p]['row'] for p in proteins], raw_anchor=anchor,
                    raw_members=proteins, subset_of_original_function_members=True,
                    binary_annotation=len(proteins) == 1)))
        graph['groups'], graph['routes'] = dict(sorted(groups.items())), routes
        graph['summary'] = dict(version='mechanism_repair_bridge_and_root_context_20261122',
            node_count=len(graph['node_types']), original_nodes_retained=original_size, new_external_nodes=len(extra),
            family_edges={f: len(es) for f, es in graph['groups'].items()},
            Synergy_added_per_fold_from_fit_only=True, path_union_and_root_context_facts=len(needed))
        relation_names = sorted({r for q in records for p in q['paths'] for r in p['relations']})
        relation_id = {r: i + 1 for i, r in enumerate(relation_names)}
        n = len(drugs)
        path_nodes, path_rel, hops = [[0] * 5], [[0] * 4], [0]
        pair_paths, pair_mask = torch.zeros((n * n, 10), dtype=torch.long), torch.zeros((n * n, 10), dtype=torch.bool)
        for q in records:
            a, b = q['drug_indices']
            pa, pb = q['prime_indices']
            for k, p in enumerate(q['paths']):
                if p['entities'][0] != pa:
                    p = index.path(list(reversed(p['entities'])), list(reversed(p['fact_ids'])))
                assert p['entities'][0] == pa and p['entities'][-1] == pb
                local = [a] + [node_local[v] for v in p['entities'][1:-1]] + [b]
                pid, h = len(hops), p['hops']
                path_nodes.append(local + [0] * (5 - len(local)))
                path_rel.append([relation_id[r] for r in p['relations']] + [0] * (4 - h))
                hops.append(h)
                pair_paths[a * n + b, k] = pair_paths[b * n + a, k] = pid
                pair_mask[a * n + b, k] = pair_mask[b * n + a, k] = True
        cache = dict(nodes=torch.tensor(path_nodes), relations=torch.tensor(path_rel), hops=torch.tensor(hops),
            pair_paths=pair_paths, pair_mask=pair_mask, relation_names=['PAD'] + relation_names,
            number_of_drugs=n, identity='dataset drug index ascending, inverse relation IDs preserved')
        torch.save(graph, folder / 'graph.pt')
        torch.save(cache, folder / 'paths.pt')
        pending[dataset] = (semantic['features'], [nodes[p]['node_name'] for p in extra])
        summaries[dataset] = dict(drugs=n, mapped_drugs=len(drug_local), drugs_with_targets=len({d for d in prime if d in index.targets}),
            possible_pairs=len(records), previous_pairs_with_paths=sum(q['previous_path_count'] > 0 for q in records),
            pairs_with_paths=sum(bool(q['paths']) for q in records),
            truncated_bridge_search_pairs=sum(q['bridge_budget_truncated'] for q in records),
            selected_bridge_paths=sum(p.get('template') == 'target_ppi_bridge_target' for q in records for p in q['paths']),
            paths=sum(len(q['paths']) for q in records), single_drug_annotation_facts=len(annotation_context),
            graph=graph['summary'], drug_file_sha256=sha(drug_file), source=source)
        write(DISPATCH / 'progress.json', dict(status='running', phase='retrieved', dataset=dataset, **summaries[dataset]))
        print(dataset, summaries[dataset]['pairs_with_paths'], 'pairs with paths;', len(extra), 'new nodes', flush=True)
    needed_text = sorted({text for base, extra in pending.values() for text in extra} - set(vectors))
    if needed_text:
        from dual_branch.prepare import build_tokenizer
        from transformers import AutoModel
        download = read(ROOT / 'artifacts/CODER_DOWNLOAD.json')
        bundle = ROOT / 'models/coder_eng'
        assert all(sha(bundle / name) == digest for name, digest in download['files'].items())
        tokenizer = build_tokenizer()
        encoder = AutoModel.from_pretrained(bundle, local_files_only=True).eval()
        with torch.no_grad():
            for start in range(0, len(needed_text), 32):
                texts = needed_text[start:start + 32]
                batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                values = torch.nn.functional.normalize(encoder(**batch).pooler_output, dim=-1)
                vectors.update(zip(texts, values.cpu()))
                write(DISPATCH / 'progress.json', dict(status='running', phase='extra_CODER',
                    completed=min(start + 32, len(needed_text)), total=len(needed_text)))
        del encoder
    for dataset, (base, extra) in pending.items():
        folder = DESTINATION / dataset
        features = torch.cat((base, torch.stack([vectors[t] for t in extra]))) if extra else base
        torch.testing.assert_close(features[:len(base)], base, rtol=0, atol=0)
        assert bool(torch.isfinite(features).all())
        torch.save(features, folder / 'all_nodes_coder.pt')
        summaries[dataset]['files_sha256'] = {name: sha(folder / name) for name in ['graph.pt', 'paths.pt', 'paths.jsonl', 'all_nodes_coder.pt']}
        write(folder / 'PROVENANCE.json', dict(summaries[dataset], labels_used=False, original_CODER_rows_exact=True,
            Cell_expression_unchanged=True, extra_names=extra, mapping_sha256=sha(mapping_file)))
    write(DISPATCH / 'PREPARATION.json', dict(status='complete', datasets=summaries, extra_names_encoded=len(needed_text),
        bridge_max_intermediate_proteins=1, bridge_budget=128, single_target_top_annotations_per_family=4,
        maximum_original_function_members=200, K=10, labels_read=False, GPU_used=False,
        encoder='existing pinned HF CODER, corrected uncased tokenizer, pooler L2'))
    write(DISPATCH / 'progress.json', dict(status='complete', phase='prepared'))


if __name__ == '__main__':
    main()

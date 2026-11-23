"""CPU-only, no-label preparation of Cell-specific path evidence.

Run once per dataset. Refuses overwrites. It never opens fold labels, old
checkpoints, test data or an encoder, and never submits GPU/fit jobs.
"""
import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha
from dual_branch.kpaths_retrieval import FUNCTIONS
from dual_branch.cell_conditioned_retrieval import CellMechanismIndex, CellExpressionSupport

VERSION = 'cell_conditioned_retrieval_20261122'


def extend_graph(graph, facts, needed):
    """Add only witnessed path facts, using already encoded graph nodes."""
    graph = copy.deepcopy(graph)
    local = {p: graph['n_base'] + i for i, p in enumerate(graph['external_prime_indices'])}
    binary = {name: {tuple(sorted(e['members'])) for e in edges}
              for name, edges in graph['groups'].items() if name == 'protein_protein'}
    functional = {}
    for name, edges in graph['groups'].items():
        if name in FUNCTIONS:
            for edge in edges:
                functional[name, edge['proof']['raw_anchor']] = edge
    added = Counter()
    for fid in sorted(needed):
        fact = facts[fid]
        name, h, t = fact['family'], fact['head'], fact['tail']
        if name == 'drug_target':
            # The original graph already retains ALL mapped-drug DTIs.
            continue
        if name == 'protein_protein':
            members = [local[h], local[t]]
            identity = tuple(sorted(members))
            existing = binary.setdefault(name, set())
            if identity not in existing:
                graph['groups'].setdefault(name, []).append(dict(members=members, positions=[0, 1], weight=1.,
                    proof=dict(raw_row=fact['row'], raw_primekg_endpoints=[h, t])))
                existing.add(identity)
                added[name] += 1
        elif name in FUNCTIONS:
            key = (name, h)
            if key not in functional:
                edge = dict(members=[local[h]], positions=[0], weight=1.,
                    proof=dict(raw_rows=[], raw_anchor=h, raw_members=[],
                               subset_of_original_function_members=True, binary_annotation=False))
                graph['groups'].setdefault(name, []).append(edge)
                functional[key] = edge
            edge = functional[key]
            if t not in edge['proof']['raw_members']:
                edge['proof']['raw_members'].append(t)
                edge['proof']['raw_rows'].append(fact['row'])
                edge['members'].append(local[t])
                edge['positions'].append(1)
                added[name] += 1
            edge['proof']['binary_annotation'] = len(edge['members']) == 2
        else:
            raise ValueError('Unexpected fact family')
    graph['groups'] = dict(sorted(graph['groups'].items()))
    graph['summary'] = dict(graph['summary'], version=VERSION,
        family_edges={name: len(edges) for name, edges in graph['groups'].items()},
        added_witnessed_member_facts=dict(added), new_external_nodes=0,
        global_graph_is_candidate_pool_union=True, global_propagation_per_query=False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['ALMANAC', 'ONEIL'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    dataset = args.dataset
    folder = ROOT / 'data' / VERSION / dataset
    if folder.exists():
        raise FileExistsError('Preserve previous preparation attempts')
    folder.mkdir(parents=True)
    dispatch = ROOT / 'artifacts' / VERSION
    started = time.monotonic()
    progress = dispatch / ('PREPARE_' + dataset + '.json')
    write(progress, dict(status='running', phase='loading', pid=os.getpid()))
    source_dir = ROOT / 'artifacts/kpaths_reference_review_20261120/raw_primekg_audit'
    source = read(source_dir / 'INDEX.json')
    assert sha(source_dir / 'index_facts.jsonl') == source['index_facts_sha256']
    facts = []
    with (source_dir / 'index_facts.jsonl').open() as stream:
        for line in stream:
            fact = json.loads(line)
            if fact['family'] in FUNCTIONS | {'drug_target', 'protein_protein'}:
                facts.append(fact)
    node_file = source_dir / 'index_nodes.json'
    nodes = {int(k): v for k, v in read(node_file).items()}
    types = {p: record['node_type'] for p, record in nodes.items()}
    index = CellMechanismIndex(facts, types)
    previous_folder = ROOT / 'data/mechanism_repair_20261122' / dataset
    previous = read(previous_folder / 'PROVENANCE.json')
    assert all(sha(previous_folder / name) == digest for name, digest in previous['files_sha256'].items())
    graph = torch.load(previous_folder / 'graph.pt', map_location='cpu', weights_only=True)
    assert 'Synergy_Hyperedge' not in graph['groups']
    support = CellExpressionSupport.from_graph(graph, types)
    local = {p: graph['n_base'] + i for i, p in enumerate(graph['external_prime_indices'])}
    records = []
    with (previous_folder / 'paths.jsonl').open() as stream:
        for number, line in enumerate(stream):
            old = json.loads(line)
            pool = index.candidates(*old['prime_indices'], old['paths'], allowed_nodes=set(local))
            records.append(dict(drug_indices=old['drug_indices'], prime_indices=old['prime_indices'], **pool))
            if number % 200 == 0:
                write(progress, dict(status='running', phase='retrieving', pairs=len(records), pid=os.getpid()))
    needed = {fid for record in records for path in record['paths'] for fid in path['fact_ids']}
    new_graph = extend_graph(graph, facts, needed)
    assert new_graph['groups']['cell_expression'] == graph['groups']['cell_expression']
    assert new_graph['routes'] == graph['routes']
    torch.save(new_graph, folder / 'graph.pt')
    names = sorted({relation for record in records for path in record['paths'] for relation in path['relations']})
    relation_id = {name: i + 1 for i, name in enumerate(names)}
    n, nc, k = graph['n_drug'], graph['n_cell'], 10
    path_nodes, path_rel, hops = [[0] * 5], [[0] * 4], [0]
    pair_paths = torch.zeros(n * n, k, dtype=torch.long)
    pair_mask = torch.zeros(n * n, k, dtype=torch.bool)
    pair_slot = torch.zeros(n * n, dtype=torch.long)
    cell_paths = torch.zeros(nc, len(records) + 1, k, dtype=torch.long)
    cell_mask = torch.zeros_like(cell_paths, dtype=torch.bool)
    examples, statistics = [], Counter()
    with (folder / 'pools.jsonl').open('w') as pool_stream:
        for slot, record in enumerate(records, start=1):
            a, b = record['drug_indices']
            pa, pb = record['prime_indices']
            pair_slot[a * n + b] = pair_slot[b * n + a] = slot
            paths = record['paths']
            identities = {}
            for path in paths:
                # Independent replay of every path against the immutable,
                # previously raw-CSV-audited source facts, including roles.
                replay = index.path(path['entities'], path['fact_ids'])
                for key in ('relations', 'roles', 'source_rows', 'hops'):
                    assert replay[key] == path[key], key
                entities, fids = path['entities'], path['fact_ids']
                if entities[0] != pa:
                    entities, fids = list(reversed(entities)), list(reversed(fids))
                assert entities[0] == pa and entities[-1] == pb
                oriented = index.path(entities, fids)
                local_nodes = [a] + [local[p] for p in entities[1:-1]] + [b]
                pid, h = len(hops), len(fids)
                path_nodes.append(local_nodes + [0] * (5 - len(local_nodes)))
                path_rel.append([relation_id[r] for r in oriented['relations']] + [0] * (4 - h))
                hops.append(h)
                path['path_id'] = pid
                identities[tuple(path['entities']), tuple(path['fact_ids'])] = pid

            def path_ids(selected):
                return [identities[tuple(p['entities']), tuple(p['fact_ids'])] for p in selected]

            topology = path_ids(support.select(paths, 0, k=k, use_cell=False))
            for pair in (a * n + b, b * n + a):
                pair_paths[pair, :len(topology)] = torch.tensor(topology, dtype=torch.long)
                pair_mask[pair, :len(topology)] = True
            selected_sets = []
            different_set = False
            for cell in range(nc):
                chosen = support.select(paths, cell, k=k)
                ids = path_ids(chosen)
                cell_paths[cell, slot, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                cell_mask[cell, slot, :len(ids)] = True
                selected_sets.append(frozenset(ids))
                different_set |= frozenset(ids) != frozenset(topology)
                statistics['selected_instances'] += len(ids)
                statistics['selected_with_expression_mapping'] += sum(bool(support.support(p, cell)['measured_proteins']) for p in chosen)
            varying = len(set(selected_sets)) > 1
            statistics['pairs_with_paths'] += bool(paths)
            statistics['pairs_with_Cell_dependent_membership'] += varying
            statistics['pairs_different_from_topology'] += different_set
            statistics['pool_paths'] += len(paths)
            statistics['pool_paths_with_expression_mapping'] += sum(bool(support.support(p, 0)['measured_proteins']) for p in paths)
            statistics['pairs_enumeration_truncated'] += any(record['truncated'].values())
            statistics['pairs_pool_truncated'] += record['pool_budget_truncated']
            statistics['excluded_unencoded_candidates'] += record['excluded_unencoded']
            if varying and len(examples) < 6:
                other = next(cell for cell in range(1, nc) if selected_sets[cell] != selected_sets[0])
                chosen = {cell: support.select(paths, cell, k=k) for cell in (0, other)}
                examples.append(dict(drug_indices=[a, b],
                    pubchem_ids=[str(graph['cids'][d]) for d in (a, b)],
                    Cells=[dict(name=graph['cell_names'][cell], path_ids=path_ids(values),
                        evidence=[dict(path_id=p['path_id'], names=[nodes[v]['node_name'] for v in p['entities']],
                            source_rows=p['source_rows'], **support.support(p, cell)) for p in values])
                        for cell, values in chosen.items()]))
            pool_stream.write(json.dumps(record) + '\n')
            if slot % 200 == 0:
                write(progress, dict(status='running', phase='Cell_selection', pairs=slot, pid=os.getpid()))
    cache = dict(nodes=torch.tensor(path_nodes), relations=torch.tensor(path_rel), hops=torch.tensor(hops),
        pair_paths=pair_paths, pair_mask=pair_mask, relation_names=['PAD'] + names,
        number_of_drugs=n, cell_pair_paths=cell_paths, cell_pair_mask=cell_mask, pair_slot=pair_slot,
        identity='dataset Drug ascending; Cell in original expression-file row order',
        selection='relative expression-guided deterministic pre-gamma diverse Top-K')
    torch.save(cache, folder / 'paths.pt')
    for name, value in [('cell_pair_paths', cell_paths), ('pair_paths', pair_paths)]:
        mask = cell_mask if name.startswith('cell') else pair_mask
        assert bool((value[~mask] == 0).all())
        assert bool((value[mask] > 0).all())
        assert int(value.max()) < len(hops)
    statistics = dict(statistics)
    statistics.update(possible_pairs=len(records), Cells=nc, measured_proteins=len(support.ranks),
        fraction_path_pairs_Cell_dependent=statistics.get('pairs_with_Cell_dependent_membership', 0) / max(statistics.get('pairs_with_paths', 0), 1))
    audit = dict(status='passed', dataset=dataset, statistics=statistics, examples=examples,
        every_path_replayed=True, source_fact_index_sha256=source['index_facts_sha256'],
        raw_CSV_rescanned_this_run=False, labels_read=False, no_performance_claim=True)
    write(folder / 'SELECTION_AUDIT.json', audit)
    files = ['graph.pt', 'paths.pt', 'pools.jsonl', 'SELECTION_AUDIT.json']
    provenance = dict(dataset=dataset, version=VERSION, files_sha256={name: sha(folder / name) for name in files},
        previous_cache_sha256=sha(previous_folder / 'PROVENANCE.json'),
        source_fact_index_sha256=source['index_facts_sha256'], nodes_index_sha256=sha(node_file),
        expression_source_sha256=graph['expression_source_sha256'],
        expression_normalization='per-gene average rank across fixed unlabeled transductive Cells -> [-1,1]',
        path_score='-heuristic_cost + mean relative expression over ALL path proteins; missing=0',
        pool_size=64, enumeration_budget_per_family=4096, bridge_budget=128,
        maximum_original_function_members=200, pool_variants_per_relation=8, final_variants_per_relation=2, K=10,
        unknown_expression_not_removed=True, Synergy_added_per_fold_fit_only=True,
        Cell_matrix_changed=False, global_graph_reconstructed_per_query=False,
        learned_selection=False, labels_read=False, fold_files_opened=False, checkpoint_loaded=False,
        new_embeddings_generated=False, new_external_nodes=0, formal_fit_submitted=False,
        statistics=statistics, elapsed_seconds=time.monotonic() - started)
    write(folder / 'PROVENANCE.json', provenance)
    write(progress, dict(status='complete', dataset=dataset, **statistics, elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(dataset=dataset, statistics=statistics), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

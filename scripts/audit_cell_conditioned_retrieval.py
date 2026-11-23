"""Independent endpoint/role/source/graph-membership audit of all new paths."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha
from dual_branch.kpaths_retrieval import FUNCTIONS


def main():
    import torch
    torch.set_num_threads(2)
    source = ROOT / 'artifacts/kpaths_reference_review_20261120/raw_primekg_audit'
    metadata = read(source / 'INDEX.json')
    assert sha(source / 'index_facts.jsonl') == metadata['index_facts_sha256']
    facts = []
    with (source / 'index_facts.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['family'] in FUNCTIONS | {'drug_target', 'protein_protein'}:
                facts.append(record)
    summaries = {}
    for dataset in ['ALMANAC', 'ONEIL']:
        folder = ROOT / 'data/cell_conditioned_retrieval_20261122' / dataset
        provenance = read(folder / 'PROVENANCE.json')
        assert all(sha(folder / name) == value for name, value in provenance['files_sha256'].items())
        graph = torch.load(folder / 'graph.pt', map_location='cpu', weights_only=True)
        packed = torch.load(folder / 'paths.pt', map_location='cpu', weights_only=True)
        local = {p: graph['n_base'] + i for i, p in enumerate(graph['external_prime_indices'])}
        binary = {family: {tuple(sorted(edge['members'])) for edge in graph['groups'][family]}
                  for family in ['drug_target', 'protein_protein']}
        annotations = {(name, edge['proof']['raw_anchor']): set(zip(edge['proof']['raw_members'], edge['proof']['raw_rows']))
                       for name, edges in graph['groups'].items() if name in FUNCTIONS for edge in edges}
        n = graph['n_drug']
        total = 0
        with (folder / 'pools.jsonl').open() as stream:
            for slot, line in enumerate(stream, start=1):
                record = json.loads(line)
                a, b = record['drug_indices']
                pa, pb = record['prime_indices']
                assert int(packed['pair_slot'][a*n+b]) == slot == int(packed['pair_slot'][b*n+a])
                allowed_ids = set()
                prime_to_local = dict(local, **{})
                prime_to_local.update({p: d for p, d in [(pa, a), (pb, b)] if p is not None})
                for path in record['paths']:
                    entities, fids = path['entities'], path['fact_ids']
                    assert len(entities) == len(fids) + 1 and len(set(entities)) == len(entities)
                    assert {entities[0], entities[-1]} == {pa, pb}
                    for u, v, fid, relation, role, row in zip(entities, entities[1:], fids,
                            path['relations'], path['roles'], path['source_rows']):
                        fact = facts[fid]
                        name, h, t = fact['family'], fact['head'], fact['tail']
                        assert {u, v} == {h, t} and row == fact['row']
                        forward = u == h
                        assert role == ([0, 1] if forward else [1, 0])
                        assert relation == name + ('' if forward or name == 'protein_protein' else '^-1')
                        if name in binary:
                            assert tuple(sorted((prime_to_local[h], prime_to_local[t]))) in binary[name]
                        else:
                            assert (t, row) in annotations[name, h]
                    # Verify the actual packed neural encoder sequence is
                    # oriented by DATASET drug IDs, not PrimeKG index order.
                    if entities[0] != pa:
                        entities, fids = entities[::-1], fids[::-1]
                    pid, hops = path['path_id'], path['hops']
                    expected_nodes = [a] + [local[p] for p in entities[1:-1]] + [b]
                    assert packed['nodes'][pid, :hops+1].tolist() == expected_nodes
                    expected_relations = []
                    for u, fid in zip(entities, fids):
                        fact = facts[fid]
                        inverse = u != fact['head'] and fact['family'] != 'protein_protein'
                        expected_relations.append(fact['family'] + ('^-1' if inverse else ''))
                    assert [packed['relation_names'][i] for i in packed['relations'][pid, :hops].tolist()] == expected_relations
                    assert int(packed['hops'][pid]) == hops
                    allowed_ids.add(pid)
                    total += 1
                selected = packed['cell_pair_paths'][:, slot]
                selected_mask = packed['cell_pair_mask'][:, slot]
                assert bool((selected[~selected_mask] == 0).all())
                assert set(selected[selected_mask].tolist()) <= allowed_ids
                assert set(packed['pair_paths'][a*n+b][packed['pair_mask'][a*n+b]].tolist()) <= allowed_ids
                torch.testing.assert_close(packed['pair_paths'][a*n+b], packed['pair_paths'][b*n+a], rtol=0, atol=0)
        assert total == len(packed['hops']) - 1
        summaries[dataset] = dict(paths=total, every_source_row_endpoint_role_verified=True,
            all_path_facts_present_in_global_graph=True, every_selected_path_belongs_to_own_pair=True,
            encoded_sequence_orientation_verified=True, padding_zero=True, label_files_opened=False)
    write(ROOT / 'artifacts/cell_conditioned_retrieval_20261122/PATH_PROOF_AUDIT.json',
        dict(status='passed', datasets=summaries, source_index_sha256=metadata['index_facts_sha256'],
             raw_CSV_rescanned=False, independent_of_retrieval_and_ranking_functions=True))
    print(summaries, flush=True)


if __name__ == '__main__':
    main()

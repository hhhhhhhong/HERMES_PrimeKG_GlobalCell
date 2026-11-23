"""Independent proof replay of prepared paths against the immutable KG fact index."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/kpaths_reference_review_20261120/raw_primekg_audit'
    metadata = json.loads((source / 'INDEX.json').read_text())
    digest = hashlib.sha256((source / 'index_facts.jsonl').read_bytes()).hexdigest()
    assert digest == metadata['index_facts_sha256']
    families = {'drug_target', 'protein_protein', 'bioprocess_protein', 'cellcomp_protein',
                'molfunc_protein', 'pathway_protein'}
    facts = []
    with (source / 'index_facts.jsonl').open() as stream:
        for line in stream:
            f = json.loads(line)
            if f['family'] in families:
                facts.append(f)
    report = {}
    for dataset in ['ALMANAC', 'ONEIL']:
        folder = ROOT / 'data/mechanism_repair_20261122' / dataset
        records = [json.loads(s) for s in (folder / 'paths.jsonl').read_text().splitlines()]
        prime = {p for q in records for p in q['prime_indices'] if p is not None}
        targets = {f['tail'] for f in facts if f['family'] == 'drug_target' and f['head'] in prime}
        checked, bridges, extra_bridges, raw_rows = 0, set(), set(), set()
        for q in records:
            assert q['drug_indices'][0] < q['drug_indices'][1]
            assert len(q['paths']) <= 10
            signatures = Counter()
            for p in q['paths']:
                ids, fids = p['entities'], p['fact_ids']
                assert len(ids) == len(set(ids)) == len(fids) + 1
                assert p['hops'] == len(fids) <= 4
                assert {ids[0], ids[-1]} == set(q['prime_indices'])
                assert facts[fids[0]]['family'] == facts[fids[-1]]['family'] == 'drug_target'
                signatures[tuple(p['relations'])] += 1
                for i, (a, b, fid) in enumerate(zip(ids, ids[1:], fids)):
                    f = facts[fid]
                    assert {a, b} == {f['head'], f['tail']}
                    forward = a == f['head']
                    expected = f['family'] + ('' if forward or f['family'] == 'protein_protein' else '^-1')
                    assert p['relations'][i] == expected
                    assert p['roles'][i] == ([0, 1] if forward else [1, 0])
                    assert p['source_rows'][i] == f['row']
                    raw_rows.add(f['row'])
                if p.get('template') == 'target_ppi_bridge_target':
                    assert len(fids) == 4
                    assert [facts[i]['family'] for i in fids[1:-1]] == ['protein_protein', 'protein_protein']
                    bridges.add(ids[2])
                    if ids[2] not in targets:
                        extra_bridges.add(ids[2])
                checked += 1
            assert all(n <= 2 for n in signatures.values())
        report[dataset] = dict(paths_independently_replayed=checked, unique_raw_source_rows=len(raw_rows),
            unique_bridge_proteins=len(bridges), bridges_not_dataset_direct_targets=len(extra_bridges),
            no_fabricated_pair_paths=True, all_ordered_roles_verified=True,
            proof_source='previously raw-CSV-audited immutable full KG index; SHA verified')
    target = ROOT / 'artifacts/mechanism_repair_20261122/PATH_PROOF_AUDIT.json'
    target.write_text(json.dumps(dict(status='passed', datasets=report, labels_read=False,
        index_sha256=digest, auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

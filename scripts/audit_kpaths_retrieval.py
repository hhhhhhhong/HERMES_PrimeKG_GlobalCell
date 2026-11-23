"""CPU-only, label-free path audit over the full raw PrimeKG fact network.

Only dataset drug identity CSVs are read. No synergy rows, expression matrix,
validation/test samples, embeddings, model checkpoint or GPU are accessed.
"""
import argparse
from collections import Counter
import csv
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dual_branch.kpaths_retrieval import KPathsIndex, ALLOWED, FUNCTIONS
from global_cell.common import read, write, sha

RAW = ROOT.parent / 'HERMES_PrimeKG_HHNN/data/raw'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=16)
    parser.add_argument('--candidate-budget', type=int, default=128)
    parser.add_argument('--seconds-budget', type=float, default=2.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.candidate_budget < 1 or args.seconds_budget <= 0:
        parser.error('Pairs, candidate budget and seconds budget must be positive')
    out = args.out.resolve()
    assert str(out).startswith(str(ROOT) + '/')
    assert not (out / 'progress.json').exists(), 'Retain earlier audits'
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(out/'progress.json', dict(status='running', phase='reading_PrimeKG', pid=os.getpid()))
    nodes = {int(n['node_index']):n for n in csv.DictReader((RAW/'nodes.csv').open())}
    mapping = {x['cid']:x['best']['node'] for x in read(RAW/'drug_mapping.json') if x['matched']}
    drugbank = {n['node_id']:i for i,n in nodes.items() if n['node_type']=='drug'}
    drugs, roots = {}, set()
    for dataset in ['ALMANAC','ONEIL']:
        path = ROOT/'vendor/hermes/data'/(dataset+'-COSMIC')/'drug_smiles.csv'
        records = list(csv.DictReader(path.open()))
        ids = [drugbank.get(mapping.get(str(x['pubchemid']),'')) for x in records]
        roots.update(p for p in ids if p is not None)
        drugs[dataset] = dict(records=records, prime_indices=ids, CSV=str(path), CSV_sha256=sha(path))
    facts, seen, raw_count = [], set(), 0
    raw_allowed = FUNCTIONS | {'drug_protein','protein_protein','disease_protein','indication','contraindication','off-label use'}
    with (RAW/'edges.csv').open() as stream:
        for row_id, row in enumerate(csv.DictReader(stream)):
            raw_count += 1
            rel = row['relation']
            if rel not in raw_allowed:
                continue
            original_x, original_y = int(row['x_index']), int(row['y_index'])
            x,y = original_x, original_y
            if rel=='drug_protein' or rel in {'indication','contraindication','off-label use'}:
                if nodes[x]['node_type']!='drug': x,y=y,x
                if x not in roots: continue
                family='drug_'+row['display_relation'] if rel=='drug_protein' else 'prime_drug_disease_'+rel.replace(' ','_').replace('-','_')
            elif rel=='protein_protein':
                x,y=min(x,y),max(x,y)
                family=rel
            elif rel=='disease_protein':
                if nodes[x]['node_type']!='disease':x,y=y,x
                family='prime_disease_protein'
            else:
                if nodes[x]['node_type']=='gene/protein':x,y=y,x
                family=rel
            if family not in ALLOWED or (family,x,y) in seen:
                continue
            seen.add((family,x,y))
            facts.append(dict(family=family,head=x,tail=y,row=row_id,
                raw_x=original_x,raw_y=original_y,raw_relation=rel))
            if len(facts)%100000==0:
                print('FACTS',len(facts),'RAW_ROWS',raw_count,flush=True)
    print('BUILD_INDEX',len(facts),flush=True)
    index = KPathsIndex(facts,{i:n['node_type'] for i,n in nodes.items()})
    with (out/'index_facts.jsonl').open('w') as stream:
        for f in facts:stream.write(json.dumps(f)+'\n')
    write(out/'index_nodes.json',{str(i):dict(node_type=n['node_type'],node_name=n['node_name'],node_id=n['node_id']) for i,n in nodes.items()})
    raw_hash = {name:sha(RAW/name) for name in ['nodes.csv','edges.csv','drug_mapping.json']}
    reference = read(ROOT.parent/'HERMES_PrimeKG_PathBatch/artifacts/PREPARED.json')
    assert raw_hash['edges.csv']==reference['raw_edges_sha256'] and raw_hash['nodes.csv']==reference['raw_nodes_sha256']
    write(out/'INDEX.json',dict(status='complete',facts=len(facts),raw_rows=raw_count,
        family_counts=dict(Counter(f['family'] for f in facts)),raw_sha256=raw_hash,
        index_facts_sha256=sha(out/'index_facts.jsonl'),source_row_index='zero-based data row, excluding CSV header',
        scope='Full raw PrimeKG for allowed relation families, Drug facts limited to dataset roots',
        no_restriction_to_prior_global_graph_protein_pool=True,labels_read=False))
    summaries = {}
    for dataset, d in drugs.items():
        pairs=list(itertools.combinations(range(len(d['records'])),2))
        selected=sorted(random.Random(0).sample(pairs,min(args.pairs,len(pairs))))
        records=[]
        for count,(a,b) in enumerate(selected):
            pa,pb=d['prime_indices'][a],d['prime_indices'][b]
            kwargs=dict(candidate_budget=args.candidate_budget,seconds_budget=args.seconds_budget)
            ref=index.reference(pa,pb,**kwargs)
            adapted=index.adapted(pa,pb,**kwargs)
            for result in [ref,adapted]:
                for p in result['paths']+result['fallback']:
                    assert len(p['entities'])==len(set(p['entities']))
                    assert all(nodes[v]['node_type']!='drug' for v in p['entities'][1:-1])
                    for u,v,fid,row in zip(p['entities'],p['entities'][1:],p['fact_ids'],p['source_rows']):
                        f=facts[fid]
                        assert {u,v}=={f['head'],f['tail']}=={f['raw_x'],f['raw_y']}
                        assert f['row']==row and f['family'] in ALLOWED
            record=dict(drug_indices=[a,b],drug_names=[d['records'][a]['name'],d['records'][b]['name']],
                prime_indices=[pa,pb],reference=ref,adapted=adapted)
            records.append(record)
            with (out/(dataset+'_paths.jsonl')).open('a') as stream:stream.write(json.dumps(record)+'\n')
            write(out/'progress.json',dict(status='running',phase='retrieving',dataset=dataset,
                completed_pairs=count+1,requested_pairs=len(selected),pid=os.getpid(),seconds=time.monotonic()-started))
            print(dataset,count+1,'of',len(selected),'paths',len(ref['paths']),len(adapted['paths']),
                'censored',ref['censored'],adapted['censored'],flush=True)
        summary=dict(sampled_pairs=len(records),total_possible_unordered_pairs=len(pairs),sample_seed=0,
            coverage_is_sample_only=True,training_seed_not_changed=True,drug_CSV=d['CSV'],drug_CSV_sha256=d['CSV_sha256'])
        summary['drug_mapping'] = dict(total=len(d['records']),
            matched=sum(x is not None for x in d['prime_indices']),
            unique_primekg_drugs=len({x for x in d['prime_indices'] if x is not None}),
            sampled_pairs_with_unmapped_endpoint=sum(None in q['prime_indices'] for q in records),
            sampled_pairs_mapping_to_same_drug=sum(q['prime_indices'][0] is not None and
                q['prime_indices'][0]==q['prime_indices'][1] for q in records))
        for method in ['reference','adapted']:
            values=[q[method] for q in records]
            summary[method]=dict(with_connecting_paths=sum(bool(v['paths']) for v in values),
                fallback_only=sum(not v['paths'] and bool(v['fallback']) for v in values),
                censored_queries=sum(v['censored'] for v in values),
                mean_paths=statistics.mean(len(v['paths']) for v in values),
                mean_seconds=statistics.mean(v['seconds'] for v in values),
                max_seconds=max(v['seconds'] for v in values),
                families=dict(Counter(p['family'] for v in values for p in v['paths'])),
                four_hop_paths=sum(p['hops']==4 for v in values for p in v['paths']))
        summaries[dataset]=summary
        write(out/(dataset+'_SUMMARY.json'),summary)
    write(out/'AUDIT.json',dict(status='passed',datasets=summaries,index=str(out/'INDEX.json'),
        source_sha256={str(ROOT/name):sha(ROOT/name) for name in ['dual_branch/kpaths_retrieval.py','scripts/audit_kpaths_retrieval.py','tests/test_kpaths_retrieval.py']},
        GPU_used=False,model_changed=False,training_started=False,embeddings_generated=False,
        synergy_samples_or_sealed_test_read=False,seconds=time.monotonic()-started))
    write(out/'progress.json',dict(status='complete',phase='audit_complete',pid=os.getpid(),seconds=time.monotonic()-started))
    print('COMPLETE',summaries,flush=True)


if __name__=='__main__':
    main()

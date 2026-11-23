"""CPU-only path preparation. Reads drug identities and KG facts, no labels.

Target/GO/Reactome/PPI candidates are retrieved on the raw KG first. The union
of their witnessed facts joins the original Cell-expression graph; per-query
path IDs remain distinct for gamma. Synergy is added later from EACH fit only.
"""
import copy
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import DATA, environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dual_branch.kpaths_data import PATH_DATA, DISPATCH
from dual_branch.kpaths_retrieval import KPathsIndex, FUNCTIONS
from global_cell.common import read, write, sha


def main():
    import torch
    torch.set_num_threads(2)
    assert not (DISPATCH / 'PREPARATION.json').exists(), 'Retain previous cache'
    DISPATCH.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    audit = ROOT / 'artifacts/kpaths_reference_review_20261120/raw_primekg_audit'
    source = read(audit / 'INDEX.json')
    assert sha(audit / 'index_facts.jsonl') == source['index_facts_sha256']
    allowed = FUNCTIONS | {'drug_target', 'protein_protein'}
    with (audit / 'index_facts.jsonl').open() as stream:
        facts = [json.loads(s) for s in stream if json.loads(s)['family'] in allowed]
    nodes = {int(k): v for k, v in read(audit / 'index_nodes.json').items()}
    index = KPathsIndex(facts, {i:n['node_type'] for i,n in nodes.items()})
    mapping_path = ROOT.parent / 'HERMES_PrimeKG_HHNN/data/raw/drug_mapping.json'
    mapping = {x['cid']:x['best']['node'] for x in read(mapping_path) if x['matched']}
    drugbank = {n['node_id']:i for i,n in nodes.items() if n['node_type']=='drug'}
    prepared = {}
    new_texts, pending = set(), {}
    for dataset in ['ALMANAC', 'ONEIL']:
        folder = PATH_DATA / dataset
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / 'PROVENANCE.json').exists()
        original_graph_path = ROOT / 'data' / dataset / 'graph.pt'
        graph = copy.deepcopy(torch.load(original_graph_path, weights_only=True, map_location='cpu'))
        semantic = torch.load(DATA / dataset / 'all_nodes_coder.pt', weights_only=True, map_location='cpu')
        drug_csv = ROOT / 'vendor/hermes/data' / (dataset+'-COSMIC') / 'drug_smiles.csv'
        with drug_csv.open() as stream:
            drugs = list(csv.DictReader(stream))
        assert [str(x['pubchemid']) for x in drugs] == [str(x) for x in graph['cids']]
        prime = [drugbank.get(mapping.get(str(x['pubchemid']),'')) for x in drugs]
        # Partial path retrieval survives an interrupted preparation without
        # being mistaken for a complete cache. No supervised rows are read.
        records = [json.loads(s) for s in (folder / 'paths.jsonl').read_text().splitlines()] if (folder / 'paths.jsonl').exists() else []
        by_pair = {tuple(q['drug_indices']):q for q in records}
        n = len(drugs)
        for a in range(n):
            for b in range(a+1,n):
                if (a,b) in by_pair:
                    continue
                r = index.adapted(prime[a],prime[b],k=10,candidate_budget=128,seconds_budget=2.)
                # This index contains only target/PPI/function facts. Fallback
                # is logged but never used as a connecting mechanism path.
                for p in r['paths']:
                    assert p['family'] in {'shared_protein','ppi','go','pathway'}
                    assert all(facts[i]['family']=='drug_target' for i in (p['fact_ids'][0],p['fact_ids'][-1]))
                q=dict(drug_indices=[a,b],prime_indices=[prime[a],prime[b]],
                    paths=r['paths'],fallback=r['fallback'],censored=r['censored'],seconds=r['seconds'])
                with (folder/'paths.jsonl').open('a') as stream:stream.write(json.dumps(q)+'\n')
                by_pair[a,b]=q
            progress=dict(status='running',phase='retrieving',dataset=dataset,completed_pairs=len(by_pair),
                total_pairs=n*(n-1)//2,seconds=time.monotonic()-started,pid=os.getpid())
            write(DISPATCH/'progress.json',progress)
            print(progress,flush=True)
        records=[by_pair[k] for k in sorted(by_pair)]
        assert len(records)==n*(n-1)//2
        needed_fids = {fid for q in records for p in q['paths'] for fid in p['fact_ids']}
        # Keep all genuine direct targets of dataset roots even for queries
        # with no connecting route. Gamma still sees only connected paths.
        drug_local={p:i for i,p in enumerate(prime) if p is not None}
        assert len(drug_local)==sum(p is not None for p in prime), 'Resolve alias collisions explicitly'
        needed_fids.update(i for i,f in enumerate(facts) if f['family']=='drug_target' and f['head'] in drug_local)
        node_local={int(p):graph['n_base']+i for i,p in enumerate(graph['external_prime_indices'])}
        requested={v for fid in needed_fids for v in [facts[fid]['head'],facts[fid]['tail']] if nodes[v]['node_type']!='drug'}
        extra=sorted(requested-set(node_local))
        original_size=len(graph['node_types'])
        for p in extra:
            node_local[p]=len(graph['node_types'])
            t=graph['type_names'].index(nodes[p]['node_type'])
            graph['node_types']=torch.cat([graph['node_types'],torch.tensor([t],dtype=torch.long)])
            graph['external_prime_indices'].append(p)
            graph['external_nodes'].append(nodes[p])
            new_texts.add(nodes[p]['node_name'])
        groups=defaultdict(list)
        groups['cell_expression']=graph['groups']['cell_expression']
        routes={int(k):v for k,v in graph['routes'].items() if int(k)>=graph['n_drug']}
        functions=defaultdict(dict)
        for fid in sorted(needed_fids):
            f=facts[fid];family=f['family'];h,t=f['head'],f['tail']
            if family in FUNCTIONS:
                functions[family,h][t]=f
                continue
            if family=='drug_target':
                if h not in drug_local:continue
                a,b=drug_local[h],node_local[t]
                routes.setdefault(a,[]).append(dict(protein=b,relation=family,
                    proof=dict(raw_row=f['row'],raw_primekg_endpoints=[h,t])))
            else:a,b=node_local[h],node_local[t]
            groups[family].append(dict(members=[a,b],positions=[0,1],weight=1.,
                proof=dict(raw_row=f['row'],raw_primekg_endpoints=[h,t])))
        for (family,anchor),members in sorted(functions.items()):
            assert len(members)>=2
            proteins=sorted(members)
            groups[family].append(dict(members=[node_local[anchor]]+[node_local[p] for p in proteins],
                positions=[0]+[1]*len(proteins),weight=1.,
                proof=dict(raw_rows=[members[p]['row'] for p in proteins],raw_anchor=anchor,
                    raw_members=proteins,subset_of_original_function_members=True)))
        graph['groups']=dict(sorted(groups.items()))
        graph['routes']=routes
        graph['summary']=dict(version='target_kpaths_union_with_global_Cell',node_count=len(graph['node_types']),
            original_nodes_retained=original_size,new_external_nodes=len(extra),
            family_edges={f:len(es) for f,es in graph['groups'].items()},
            Synergy_added_per_fold_from_fit_only=True,
            raw_primekg_allowed_facts=len(facts),path_union_facts=len(needed_fids))
        relation_names=sorted({r for q in records for p in q['paths'] for r in p['relations']})
        rel={r:i+1 for i,r in enumerate(relation_names)}
        path_nodes=[];path_rel=[];lengths=[];pair_paths=torch.zeros((n*n,10),dtype=torch.long)
        pair_mask=torch.zeros((n*n,10),dtype=torch.bool)
        # Index0 is an explicit unavailable path, never a real entity route.
        path_nodes.append([0]*5);path_rel.append([0]*4);lengths.append(0)
        for q in records:
            a,b=q['drug_indices'];pa,pb=q['prime_indices']
            for k,p in enumerate(q['paths']):
                if p['entities'][0]!=pa:
                    p=index.path(list(reversed(p['entities'])),list(reversed(p['fact_ids'])))
                assert p['entities'][0]==pa and p['entities'][-1]==pb
                local=[a]+[node_local[v] for v in p['entities'][1:-1]]+[b]
                pid=len(path_nodes);h=p['hops']
                path_nodes.append(local+[0]*(5-len(local)))
                path_rel.append([rel[r] for r in p['relations']]+[0]*(4-h));lengths.append(h)
                pair_paths[a*n+b,k]=pair_paths[b*n+a,k]=pid
                pair_mask[a*n+b,k]=pair_mask[b*n+a,k]=True
        cache=dict(nodes=torch.tensor(path_nodes,dtype=torch.long),relations=torch.tensor(path_rel,dtype=torch.long),
            hops=torch.tensor(lengths,dtype=torch.long),pair_paths=pair_paths,pair_mask=pair_mask,
            relation_names=['PAD']+relation_names,number_of_drugs=n,
            identity='dataset drug index ascending, direction inverted when raw canonical order differs')
        torch.save(graph,folder/'graph.pt');torch.save(cache,folder/'paths.pt')
        pending[dataset]=(semantic['features'],[nodes[p]['node_name'] for p in extra])
        prepared[dataset]=dict(drugs=n,mapped_drugs=len(drug_local),possible_pairs=len(records),
            pairs_with_paths=sum(bool(q['paths']) for q in records),censored_pairs=sum(q['censored'] for q in records),
            path_counts=dict(Counter(p['family'] for q in records for p in q['paths'])),
            graph=graph['summary'],drug_csv_sha256=sha(drug_csv),base_graph_sha256=sha(original_graph_path))
    # Existing corrected CODER vectors are reused exactly. Encode only extra
    # raw KG names, using the same pinned HF model/tokenizer, on CPU.
    vectors={}
    for dataset in prepared:
        old=torch.load(DATA/dataset/'all_nodes_coder.pt',weights_only=True,map_location='cpu')
        for r,v in zip(old['records'],old['features']):vectors.setdefault(r['text'],v)
    needed=sorted(new_texts-set(vectors))
    if needed:
        from dual_branch.prepare import build_tokenizer
        from transformers import AutoModel
        download=read(ROOT/'artifacts/CODER_DOWNLOAD.json')
        bundle=ROOT/'models/coder_eng'
        assert all(sha(bundle/name)==sig for name,sig in download['files'].items())
        tokenizer=build_tokenizer();encoder=AutoModel.from_pretrained(bundle,local_files_only=True).eval()
        with torch.no_grad():
            for start in range(0,len(needed),32):
                texts=needed[start:start+32]
                batch=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
                features=torch.nn.functional.normalize(encoder(**batch).pooler_output,dim=-1)
                vectors.update(zip(texts,features.cpu()))
                write(DISPATCH/'progress.json',dict(status='running',phase='encoding_extra_CODER',
                    completed=min(start+32,len(needed)),total=len(needed),pid=os.getpid()))
        del encoder
    for dataset,(base,extra) in pending.items():
        folder=PATH_DATA/dataset
        features=torch.cat([base,torch.stack([vectors[t] for t in extra])]) if extra else base
        assert torch.isfinite(features).all()
        torch.save(features,folder/'all_nodes_coder.pt')
        prepared[dataset]['files_sha256']={name:sha(folder/name) for name in ['graph.pt','paths.pt','all_nodes_coder.pt','paths.jsonl']}
        write(folder/'PROVENANCE.json',dict(prepared[dataset],raw_source=source,
            mapping_sha256=sha(mapping_path),labels_used=False,Cell_expression_unchanged=True,
            original_semantic_rows_exact=True,extra_names=extra,from_old_checkpoint=False))
    write(DISPATCH/'PREPARATION.json',dict(status='complete',datasets=prepared,
        allowed_families=sorted(allowed),K=10,max_hops=4,four_hops_only_target_function_target=True,
        candidate_budget=128,soft_seconds_budget=2.,templates_max_variants=2,
        extra_names_encoded=len(needed),encoder='existing pinned HF CODER; unchanged uncased recipe',
        synergy_samples_or_labels_read=False,expression_values_modified=False,
        GPU_used=False,seconds=time.monotonic()-started))
    write(DISPATCH/'progress.json',dict(status='complete',phase='prepared',pid=os.getpid()))
    print('PREPARATION_COMPLETE',prepared,flush=True)


if __name__=='__main__':main()

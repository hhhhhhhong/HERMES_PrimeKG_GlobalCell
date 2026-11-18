"""One graph per dataset: all Drug/Cell roots and shared real PrimeKG entities.

The protein pool retains the previous audited DTI/PPI intermediary pool and
adds EVERY uniquely matched measured gene. PrimeKG incidences are rebuilt
from raw rows on this explicit pool. No pair-specific graph or synergy label
is used. Cell expression links are separately attributed measurements.
"""
import sys,csv,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from global_cell.common import read,write,sha
OLD=ROOT.with_name('HERMES_PrimeKG_CODER_SinglePath')
RAW=ROOT.with_name('HERMES_PrimeKG_HHNN')/'data/raw'
FUNCTIONAL={'bioprocess_protein','cellcomp_protein','molfunc_protein','pathway_protein'}

def main():
 started=time.monotonic();nodes={int(n['node_index']):n for n in csv.DictReader((RAW/'nodes.csv').open())}
 names=collections.defaultdict(list)
 for i,n in nodes.items():
  if n['node_type']=='gene/protein':names[n['node_name']].append(i)
 mapping={x['cid']:x['best']['node'] for x in read(RAW/'drug_mapping.json') if x['matched']}
 drug_ids={n['node_id']:i for i,n in nodes.items() if n['node_type']=='drug'}
 datasets={}
 for ds in ['ALMANAC','ONEIL']:
  previous=torch.load(OLD/'data'/ds/'graph.pt',weights_only=True,map_location='cpu')
  raw=torch.load(OLD/'data'/ds/'fold0/inputs.pt',weights_only=True,map_location='cpu')
  expression_file=ROOT/'vendor/hermes/data'/f'{ds}-COSMIC/cell line_gene_expression.csv'
  expression=pd.read_csv(expression_file,index_col=0)
  assert np.array_equal(expression.to_numpy(dtype='float32'),raw['inputs']['cells'].numpy())
  genes=[dict(column=j,symbol=g,prime_index=names[g][0]) for j,g in enumerate(expression.columns) if len(names[g])==1]
  proteins={i for i in previous['external_prime_indices'] if nodes[i]['node_type']=='gene/protein'}|{g['prime_index'] for g in genes}
  diseases={i for i in previous['external_prime_indices'] if nodes[i]['node_type']=='disease'}
  drugs={drug_ids[mapping[cid]]:j for j,cid in enumerate(previous['cids']) if cid in mapping and mapping[cid] in drug_ids}
  datasets[ds]=dict(previous=previous,raw=raw,expression=expression,expression_file=expression_file,
   expression_sha256=sha(expression_file),genes=genes,proteins=proteins,diseases=diseases,drugs=drugs,facts={})
  write(ROOT/'artifacts'/f'GENE_MAPPING_{ds}.json',dict(dataset=ds,source_file=str(expression_file),source_sha256=sha(expression_file),
   shape=list(expression.shape),unique_gene_matches=genes,unmatched_or_ambiguous=[dict(column=j,symbol=g,candidates=names[g]) for j,g in enumerate(expression.columns) if len(names[g])!=1],
   rule='exact official symbol, unique PrimeKG gene/protein; no alias guessing or Excel-date correction',
   cell_names=list(expression.index),row_order_exactly_equals_HERMES=True,synergy_labels_used=False))
 allowed=FUNCTIONAL|{'drug_protein','protein_protein','disease_protein','indication','contraindication','off-label use'}
 for frame in pd.read_csv(RAW/'edges.csv',chunksize=500000):
  for row in frame[frame['relation'].isin(allowed)].itertuples():
   u,v=int(row.x_index),int(row.y_index);rel=row.relation
   if rel=='drug_protein' or rel in {'indication','contraindication','off-label use'}:
    if nodes[u]['node_type']!='drug':u,v=v,u
    family='drug_'+str(row.display_relation) if rel=='drug_protein' else 'prime_drug_disease_'+rel.replace(' ','_').replace('-','_')
   elif rel=='protein_protein':u,v=min(u,v),max(u,v);family=rel
   elif rel=='disease_protein':
    if nodes[u]['node_type']!='disease':u,v=v,u
    family='prime_disease_protein'
   else:
    if nodes[u]['node_type']=='gene/protein':u,v=v,u
    family=rel
   for ds,s in datasets.items():
    if rel=='drug_protein':keep=u in s['drugs'] and v in s['proteins']
    elif rel in {'indication','contraindication','off-label use'}:keep=u in s['drugs'] and v in s['diseases']
    elif rel=='protein_protein':keep=u in s['proteins'] and v in s['proteins']
    elif rel=='disease_protein':keep=u in s['diseases'] and v in s['proteins']
    else:keep=v in s['proteins']
    if keep:s['facts'].setdefault((family,u,v),dict(family=family,head=u,tail=v,row=int(row.Index),raw_relation=rel))
  print('raw rows',int(frame.index[-1])+1,[(d,len(s['facts'])) for d,s in datasets.items()],flush=True)
 all_external={}
 for ds,s in datasets.items():
  raw=s['raw'];nd,nc=raw['num_drug'],raw['num_cell'];nb=nd+nc
  ext=sorted(s['proteins']|s['diseases']|{f['head'] for f in s['facts'].values() if f['family'] in FUNCTIONAL})
  local={v:nb+i for i,v in enumerate(ext)};local.update(s['drugs'])
  groups=collections.defaultdict(list);functions=collections.defaultdict(list);routes=collections.defaultdict(list)
  for fact in s['facts'].values():
   family,u,v,row=fact['family'],fact['head'],fact['tail'],fact['row'];a,b=local[u],local[v]
   if family in FUNCTIONAL:functions[(family,u)].append(fact);continue
   proof=dict(source='PrimeKG',source_rows=[row],prime_head=u,prime_tail=v,raw_relation=fact['raw_relation'])
   groups[family].append(dict(members=[a,b],positions=[0,1],weight=1.,proof=proof))
   if family=='protein_protein':groups[family].append(dict(members=[b,a],positions=[0,1],weight=1.,proof=dict(proof,reverse_traversal=True)))
   if a<nd:routes[str(a)].append(dict(protein=b,relation=family,proof=proof))
  for (family,anchor),facts in functions.items():
   facts=sorted(facts,key=lambda f:f['tail'])
   groups[family].append(dict(members=[local[anchor]]+[local[f['tail']] for f in facts],positions=[0]+[1]*len(facts),weight=1.,
    proof=dict(source='PrimeKG',source_rows=[f['row'] for f in facts],prime_anchor=anchor,prime_members=[f['tail'] for f in facts],retained_members=len(facts),
     meaning='observed functional membership subset on declared protein pool; not a complete pathway/reaction')))
  expr=s['expression'].to_numpy(dtype='float32');gene_map=s['genes']
  for cell in range(nc):
   for g in gene_map:
    value=float(expr[cell,g['column']]);assert np.isfinite(value)
    proof=dict(source='HERMES_expression',file_sha256=s['expression_sha256'],cell_index=cell,gene_column=g['column'],
     gene_symbol=g['symbol'],prime_gene_protein=g['prime_index'],expression_value=value)
    groups['cell_expression'].append(dict(members=[nd+cell,local[g['prime_index']]],positions=[0,1],weight=1.,
     expression_value=value,expression_bounded=value/(1+abs(value)),proof=proof))
    routes[str(nd+cell)].append(dict(protein=local[g['prime_index']],relation='cell_expression',proof=proof))
  types=['hermes_drug']*nd+['hermes_cell']*nc+[nodes[i]['node_type'].replace('/','_') for i in ext];type_names=sorted(set(types))
  graph=dict(dataset=ds,n_base=nb,n_roots=nb,n_drug=nd,n_cell=nc,groups={k:groups[k] for k in sorted(groups)},
   node_types=torch.tensor([type_names.index(t) for t in types]),type_names=type_names,external_prime_indices=ext,external_nodes=[nodes[i] for i in ext],
   routes=dict(routes),cids=s['previous']['cids'],cell_names=list(s['expression'].index),gene_mapping=gene_map,expression_source_sha256=s['expression_sha256'])
  graph['summary']=dict(dataset=ds,n_drug=nd,n_cell=nc,external_nodes=len(ext),external_types=dict(collections.Counter(types[nb:])),
   mapped_genes=len(gene_map),measured_gene_columns=expr.shape[1],cell_expression_edges=len(groups['cell_expression']),
   every_cell_degree=len(gene_map),relation_counts={k:len(v) for k,v in graph['groups'].items()},
   true_functional_hyperedges=sum(len(e['members'])>=3 for k,es in graph['groups'].items() if k in FUNCTIONAL for e in es),
   graph_scope='one shared global graph per dataset, all Drug/Cell roots; no pair/path truncation',
   protein_pool='previous audited DTI/PPI intermediate pool UNION all uniquely mapped expression genes',
   cell_link_meaning='gene expression observation mapped to PrimeKG gene/protein, not measured protein abundance/causality',
   synergy_labels_used_for_graph=False,original_HERMES_refiner=False,original_HERMES_synergy_indication=False)
  out=ROOT/'data'/ds;out.mkdir();torch.save(graph,out/'graph.pt');write(out/'GRAPH_AUDIT.json',graph['summary'])
  for fold in range(5):(out/f'fold{fold}').symlink_to(OLD/'data'/ds/f'fold{fold}',target_is_directory=True)
  all_external.update({i:nodes[i] for i in ext});print(ds,graph['summary'],flush=True)
 texts=[dict(prime_index=i,text=n['node_name'],name=n['node_name'],node_type=n['node_type']) for i,n in sorted(all_external.items())]
 write(ROOT/'data/ENTITY_TEXTS.json',texts)
 # Reuse only audited, matching fixed CODER inputs; never reuse neural model states.
 old_vectors=torch.load(OLD/'data/CODER_ENTITY_FEATURES.pt',weights_only=True,map_location='cpu')
 old_text={x['prime_index']:x['text'] for x in read(OLD/'data/ENTITY_TEXTS.json')};old_lookup={i:j for j,i in enumerate(old_vectors['prime_indices'])}
 audit=read(OLD/'runs/coder_singlepath_cv5_seed0_20261116/encoding/ENCODING_AUDIT.json');assert audit['status']=='passed'
 for ds,h in audit['dataset_feature_sha256'].items():
  p=OLD/'data'/ds/'coder_features.pt';assert sha(p)==h
  oldg=datasets[ds]['previous'];assert torch.equal(old_vectors['features'][[old_lookup[i] for i in oldg['external_prime_indices']]],torch.load(p,weights_only=True,map_location='cpu'))
 reused={t['prime_index']:old_vectors['features'][old_lookup[t['prime_index']]] for t in texts if t['prime_index'] in old_lookup and old_text[t['prime_index']]==t['text']}
 torch.save(reused,ROOT/'data/REUSED_CODER.pt')
 missing=[t for t in texts if t['prime_index'] not in reused];write(ROOT/'data/MISSING_CODER_TEXTS.json',missing)
 write(ROOT/'artifacts/PREPARED.json',dict(status='complete',external_union=len(texts),audited_CODER_reused=len(reused),new_CODER_needed=len(missing),
  raw_nodes_sha256=sha(RAW/'nodes.csv'),raw_edges_sha256=sha(RAW/'edges.csv'),seconds=time.monotonic()-started,labels_used_for_graph=False))
 print('PREPARED',len(texts),'reused',len(reused),'missing',len(missing),flush=True)
if __name__=='__main__':main()

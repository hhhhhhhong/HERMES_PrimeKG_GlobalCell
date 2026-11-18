"""Check every graph incidence against its raw PrimeKG or expression witness."""
import sys,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from global_cell.common import read,write,sha
RAW=ROOT.with_name('HERMES_PrimeKG_HHNN')/'data/raw'
FUNCTIONAL={'bioprocess_protein','cellcomp_protein','molfunc_protein','pathway_protein'}

def main():
 nodes={int(n['node_index']):n for n in csv.DictReader((RAW/'nodes.csv').open())};wanted=collections.defaultdict(set);summary={}
 for ds in ['ALMANAC','ONEIL']:
  graph=torch.load(ROOT/'data'/ds/'graph.pt',weights_only=True,map_location='cpu');nd,nc=graph['n_drug'],graph['n_cell'];nb=nd+nc
  drug_map={x['cid']:x['best']['node'] for x in read(RAW/'drug_mapping.json') if x['matched']}
  prime_drug_ids={n['node_id']:i for i,n in nodes.items() if n['node_type']=='drug'}
  root_to_prime={i:prime_drug_ids[drug_map[cid]] for i,cid in enumerate(graph['cids']) if cid in drug_map and drug_map[cid] in prime_drug_ids}
  ext=graph['external_prime_indices'];expression=pd.read_csv(ROOT/'vendor/hermes/data'/f'{ds}-COSMIC/cell line_gene_expression.csv',index_col=0)
  cell_degree=collections.Counter();cell_links=0;raw_incidence=0
  for family,edges in graph['groups'].items():
   for edge in edges:
    proof=edge['proof'];members=edge['members'];positions=edge['positions'];assert len(members)==len(positions)
    if family=='cell_expression':
     assert proof['source']=='HERMES_expression' and proof['file_sha256']==graph['expression_source_sha256']
     c,g=proof['cell_index'],proof['gene_column'];prime=proof['prime_gene_protein']
     assert members==[nd+c,nb+ext.index(prime)] and positions==[0,1]
     assert expression.columns[g]==proof['gene_symbol']==nodes[prime]['node_name']
     value=float(np.float32(expression.iloc[c,g]));assert value==edge['expression_value']==proof['expression_value']
     assert abs(edge['expression_bounded']-value/(1+abs(value)))<1e-8
     cell_degree[c]+=1;cell_links+=1
    else:
     assert proof['source']=='PrimeKG';assert all(v<nd or v>=nb for v in members)
     if family in FUNCTIONAL:
      assert ext[members[0]-nb]==proof['prime_anchor'] and [ext[v-nb] for v in members[1:]]==proof['prime_members']
      assert positions==[0]+[1]*(len(members)-1)
      for row,protein in zip(proof['source_rows'],proof['prime_members']):wanted[row].add((family,proof['prime_anchor'],protein));raw_incidence+=1
     else:
      assert len(members)==2 and positions==[0,1]
      actual=[root_to_prime[v] if v<nd else ext[v-nb] for v in members]
      expected=[proof['prime_head'],proof['prime_tail']]
      if proof.get('reverse_traversal',False):expected=expected[::-1]
      assert actual==expected
      wanted[proof['source_rows'][0]].add((family,proof['prime_head'],proof['prime_tail']));raw_incidence+=1
  assert set(cell_degree)==set(range(nc)) and set(cell_degree.values())=={642}
  for fold in range(5):
   raw=torch.load(ROOT/'data'/ds/f'fold{fold}/inputs.pt',weights_only=True,map_location='cpu')
   assert np.array_equal(raw['inputs']['cells'].numpy(),expression.to_numpy(dtype='float32'))
  summary[ds]=dict(cell_nodes=nc,cell_expression_edges=cell_links,every_cell_degree=642,raw_fact_incidences=raw_incidence)
 verified=0
 for frame in pd.read_csv(RAW/'edges.csv',chunksize=500000):
  sub=frame.loc[frame.index.intersection(list(wanted))]
  for row in sub.itertuples():
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
   assert wanted[int(row.Index)]=={(family,u,v)},(row.Index,wanted[int(row.Index)],family,u,v)
   verified+=1
 assert verified==len(wanted)
 write(ROOT/'artifacts/FACT_AUDIT.json',dict(status='passed',raw_rows_verified=verified,datasets=summary,
  graph_labels_used=False,cell_relations_are_expression_derived=True,raw_edges_sha256=sha(RAW/'edges.csv')))
 print('AUDIT PASSED',verified,summary,flush=True)
if __name__=='__main__':main()

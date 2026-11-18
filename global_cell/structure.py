"""Typed global incidences and relation combinations with shared-entity proof."""
from collections import defaultdict
import torch
from torch import nn

class Group(nn.Module):
 def __init__(self,edges,types,offset):
  super().__init__()
  vertex=torch.tensor([v for e in edges for v in e['members']],dtype=torch.long)
  pos=torch.tensor([p for e in edges for p in e['positions']],dtype=torch.long)
  edge=torch.repeat_interleave(torch.arange(len(edges)),torch.tensor([len(e['members']) for e in edges]))
  active,inverse=torch.unique(vertex,sorted=True,return_inverse=True)
  keys,slot_inverse=torch.unique(vertex*3+pos,sorted=True,return_inverse=True)
  buffers=dict(vertex=vertex,position=pos,edge=edge,active=active,inverse=inverse,slot_inverse=slot_inverse,
   slot_nodes=keys//3,slot_position=keys%3,active_types=types[active],prior=torch.tensor([e['weight'] for e in edges]),
   expression=torch.tensor([e.get('expression_bounded',0.) for e in edges],dtype=torch.float32).reshape(-1,1))
  for name,value in buffers.items():self.register_buffer(name,value)
  self.num_edges=len(edges);self.offset=offset;self.present_types=tuple(sorted(set(types[active].tolist())))
  for t in self.present_types:
   self.register_buffer('node_positions_%d'%t,torch.where(types[active]==t)[0])
   self.register_buffer('inc_positions_%d'%t,torch.where(types[vertex]==t)[0])

def compile_graph(graph):
 groups=nn.ModuleDict();meta=[];by_entity=defaultdict(list);offset=0
 for family,edges in graph['groups'].items():
  g=Group(edges,graph['node_types'],offset);groups[family]=g
  for v,p in zip(g.slot_nodes.tolist(),g.slot_position.tolist()):
   slot=len(meta);by_entity[v].append(slot);meta.append(dict(node=v,family=family,position=p))
  offset+=len(g.slot_nodes)
 pairs=[];pair_by_entity=defaultdict(list)
 for entity,slots in sorted(by_entity.items()):
  for source in slots:
   for target in slots:
    a,b=meta[source],meta[target]
    if a['family']==b['family']:continue
    pid=len(pairs);pair_by_entity[entity].append(pid)
    pairs.append(dict(entity=entity,source=source,target=target,descriptor=(a['family'],a['position'],b['family'],b['position'])))
 routes={}
 def add(root,key,source,proof):routes.setdefault((root,key,source),proof)
 for i,m in enumerate(meta):
  if m['node']<graph['n_roots']:add(m['node'],('single',m['family'],m['position']),i,dict(kind='root_incidence',slot=i))
 nslots=len(meta)
 for i,p in enumerate(pairs):
  if p['entity']<graph['n_roots']:add(p['entity'],('interaction','at_root')+p['descriptor'],nslots+i,dict(kind='same_root_entity',entity=p['entity'],pair=i))
 # Both Drug and Cell roots read ordered relation combinations at an actual
 # incident neighbor. Cell routes retain the expression-file witness; they
 # never masquerade as PrimeKG raw rows or as inferred synergy connections.
 for root,facts in graph['routes'].items():
  root=int(root)
  for f in facts:
   entity=f['protein'];family=f['relation']
   for i in pair_by_entity[entity]:
    p=pairs[i];a,b=meta[p['source']],meta[p['target']]
    if not ((a['family']==family and a['position']==1) or (b['family']==family and b['position']==1)):continue
    add(root,('interaction','via_'+family)+p['descriptor'],nslots+i,
     dict(kind='root_incident_shared_entity_route',root_type='Drug' if root<graph['n_drug'] else 'Cell',
      source_proof=f['proof'],shared_entity=entity,pair=i))
 token_keys=sorted({k for root,k,source in routes},key=str);token_index={k:i for i,k in enumerate(token_keys)}
 route_list=sorted(routes,key=lambda item:(item[0],token_index[item[1]],item[2]))
 packed=dict(source=torch.tensor([source for root,key,source in route_list]),
  group=torch.tensor([root*len(token_keys)+token_index[key] for root,key,source in route_list]),
  token=torch.tensor([token_index[key] for root,key,source in route_list]),
  pair_source=torch.tensor([p['source'] for p in pairs]),pair_target=torch.tensor([p['target'] for p in pairs]),
  pair_position=torch.tensor([[meta[p['source']]['position'],meta[p['target']]['position']] for p in pairs]),
  pair_entity=torch.tensor([p['entity'] for p in pairs]))
 mask=torch.zeros(graph['n_roots']*len(token_keys),dtype=torch.bool);mask[packed['group']]=True
 packed['mask']=mask.reshape(graph['n_roots'],len(token_keys))
 trace=dict(slots=meta,pairs=pairs,token_keys=token_keys,
  route_proofs=[dict(root=root,key=key,source=source,**routes[(root,key,source)]) for root,key,source in route_list],
  counts=dict(slots=len(meta),local_directed_interactions=len(pairs),token_types=len(token_keys),root_routes=len(route_list)))
 return groups,packed,trace

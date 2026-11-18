"""Cell must be an active global graph root, with witnessed expression links."""
import copy,unittest
import torch
from global_cell.structure import compile_graph
from global_cell.attention import AlphaBeta,RelationInteraction

def fixture():
 # 2 Drug roots, 2 Cell roots, 2 measured genes and a functional anchor.
 types=['hermes_drug','hermes_cell','gene_protein','pathway']
 def edge(m,pos=None,**extra):return dict(members=m,positions=pos or [0,1],weight=1.,proof=dict(source='fixture'),**extra)
 groups={'drug_target':[edge([0,4]),edge([1,5])],
  'cell_expression':[edge([2,4],expression_bounded=.5),edge([2,5],expression_bounded=-.3),edge([3,4],expression_bounded=-.1),edge([3,5],expression_bounded=.7)],
  'protein_protein':[edge([4,5]),edge([5,4])], 'pathway_protein':[edge([6,4,5],[0,1,1])]}
 return dict(groups=groups,node_types=torch.tensor([0,0,1,1,2,2,3]),type_names=types,n_base=4,n_roots=4,n_drug=2,
  routes={'0':[dict(protein=4,relation='drug_target',proof={})],'1':[dict(protein=5,relation='drug_target',proof={})],
   '2':[dict(protein=p,relation='cell_expression',proof={}) for p in [4,5]],'3':[dict(protein=p,relation='cell_expression',proof={}) for p in [4,5]]})

class GlobalCellTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(0);torch.set_num_threads(2)
 def setup_graph(self):
  graph=fixture();groups,packed,trace=compile_graph(graph)
  encoder=AlphaBeta(graph,groups,dim=16,heads=4);interaction=RelationInteraction(packed,4,len(trace['token_keys']),dim=16)
  return graph,groups,trace,encoder,interaction
 def test_global_cell_incidence_and_shared_entity_witness(self):
  graph,groups,trace,encoder,ri=self.setup_graph()
  self.assertTrue(ri.mask[2:].any(-1).all())
  self.assertEqual(len(groups['cell_expression'].vertex),8)
  self.assertTrue(any('cell_expression' in p['descriptor'] and 'drug_target' in p['descriptor'] for p in trace['pairs']))
  for pair in trace['pairs']:
   self.assertEqual(trace['slots'][pair['source']]['node'],pair['entity'])
   self.assertEqual(trace['slots'][pair['target']]['node'],pair['entity'])
 def test_graph_only_cell_gradient_and_global_drug_effect(self):
  graph,groups,trace,encoder,ri=self.setup_graph();x=torch.randn(7,16,requires_grad=True)
  slots,_=encoder(x,groups);table,_=ri(slots)
  table[:2].square().sum().backward()
  self.assertGreater(float(x.grad[2:4].norm()),0.)
  expression_grad=sum(float(p.grad.norm()) for p in encoder.expression_projection.parameters() if p.grad is not None)
  self.assertGreater(expression_grad,0.)
  with torch.no_grad():
   changed=x.detach().clone();changed[2]+=torch.linspace(-1,1,16)
   altered,_=ri(encoder(changed,groups)[0]);self.assertGreater(float((altered[:2]-table[:2]).abs().max()),1e-7)
 def test_expression_attribute_changes_cell_and_drug_graph_outputs(self):
  graph,groups,trace,encoder,ri=self.setup_graph();x=torch.randn(7,16)
  with torch.no_grad():
   table,_=ri(encoder(x,groups)[0]);groups['cell_expression'].expression[0]+=.5
   altered,_=ri(encoder(x,groups)[0])
   self.assertGreater(float((altered[2:]-table[2:]).abs().max()),1e-7)
   self.assertGreater(float((altered[:2]-table[:2]).abs().max()),1e-7)

if __name__=='__main__':unittest.main()

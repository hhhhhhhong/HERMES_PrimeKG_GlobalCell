"""ALMANAC fold0: three real hypergraph rounds with initially zero gates."""
import argparse,os,time,json,hashlib,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--device',required=True)
 p.add_argument('--dataset',choices=['ALMANAC','ONEIL'],required=True);p.add_argument('--fold',type=int,choices=range(5),default=0)
 p.add_argument('--stage',choices=['check','train'],required=True);p.add_argument('--result-directory',type=Path,required=True)
 p.add_argument('--expected-source-manifest',type=Path);p.add_argument('--expected-input-manifest',type=Path)
 a=p.parse_args();assert a.dataset=='ALMANAC' and a.fold==0;os.environ['CUDA_VISIBLE_DEVICES']='' if a.device=='cpu' else a.device.split(':')[-1]
 import torch,numpy as np
 from global_cell.common import ROOT,read,write,sha,sources,author_imports
 torch.set_num_threads(2)
 d=a.result_directory.resolve();assert str(d).startswith(str(ROOT)+'/') and not (d/'progress.json').exists();d.mkdir(parents=True,exist_ok=True)
 source=sources();variant={str(f):sha(f) for f in [ROOT/'models/zero_gated_hgnn.py',Path(__file__).resolve()]}
 if a.expected_source_manifest:
  m=read(a.expected_source_manifest);assert m['package']==source and all(sha(f)==h for f,h in m.get('extra',{}).items())
 if a.expected_input_manifest:assert all(sha(f)==h for f,h in read(a.expected_input_manifest).items())
 device=torch.device('cpu' if a.device=='cpu' else 'cuda:0');started=time.monotonic()
 meta=dict(dataset=a.dataset,fold=a.fold,stage=a.stage,seed=0,pid=os.getpid(),physical_device=a.device)
 write(d/'progress.json',dict(meta,status='running',phase='preparing',epoch=0))
 try:
  assert read(ROOT/'artifacts/FACT_AUDIT.json')['status']=='passed'
  author_imports();from utils import set_seed_all,get_metrics,TruncatedExponentialLR
  from global_cell.network import train_epoch,evaluate
  from models.zero_gated_hgnn import ZeroGatedHypergraphModel as GlobalCellModel
  raw=torch.load(ROOT/'data'/a.dataset/f'fold{a.fold}/inputs.pt',weights_only=True,map_location='cpu')
  graph=torch.load(ROOT/'data'/a.dataset/'graph.pt',weights_only=True,map_location='cpu')
  features=torch.load(ROOT/'data'/a.dataset/'coder_features.pt',weights_only=True,map_location='cpu')
  assert sha(ROOT/'data'/a.dataset/'coder_features.pt')==read(ROOT/'artifacts/ENCODING_AUDIT.json')['feature_sha256'][a.dataset]
  for part,key in [('fit','train'),('valid','valid')]:
   assert hashlib.sha256(raw[part]['rows'].numpy().tobytes()).hexdigest()==raw['metadata']['split'][key+'_array_sha256']
  from global_cell.synergy import add_training_synergy,query_overlap_audit
  graph=add_training_synergy(graph,raw['fit']['rows'],fold=a.fold,
   expected_train_sha256=raw['metadata']['split']['train_array_sha256'])
  # The overlap audit sees endpoint IDs only; it cannot affect edge selection.
  write(d/'SYNERGY_AUDIT.json',dict(graph['synergy_audit'],
   existing_split_overlap=query_overlap_audit(raw['fit']['rows'][:,:3],raw['valid']['rows'][:,:3])))
  set_seed_all(0);load=torch.load;state_load=torch.nn.Module.load_state_dict
  def forbidden(*args,**kwargs):raise AssertionError('Fresh model constructor attempted a checkpoint load')
  torch.load=forbidden;torch.nn.Module.load_state_dict=forbidden
  try:model=GlobalCellModel(raw,graph,features).to(device)
  finally:torch.load=load;torch.nn.Module.load_state_dict=state_load
  assert all(p.requires_grad for p in model.parameters())
  assert model.interaction.mask[model.num_drug:].any(-1).all()
  initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
  recipe=dict(model='15-relation global graph / 3 real alpha-beta rounds with zero-initialized gated residual / shared-entity HYPER / conditional gamma / original-style head',
   seed=0,dataset=a.dataset,fold=a.fold,epochs=3000,select_from_epoch=1000,early_stopping=False,optimizer='Adam',lr=1e-4,
   weight_decay=.01,scheduler_gamma=.9997,min_lr=2e-6,query_chunk=1024,one_full_fit_update_per_epoch=True,loss='BCE only',
   original_checkpoint_loaded=False,all_downstream_trainable=True,fixed_CODER_inputs=True,global_Cell_message_passing=True,
   expression_attribute=dict(input='e/(1+abs(e))',projection=str(model.encoder.expression_projection)),
   original_HERMES_refiner=False,original_synergy_indication_graph=False,test_evaluated=False,
   propagation_layers=3,propagation_gate_initialization=0.,identity_branch_normalization=False,
   node_feedback='mean over actual incident relation-role slots only',original_readout_gate_initialization=.05,
   original_HERMES_refiner_exact_copy=False,similarity_smoothing=False,
   synergy_hyperedges=True,synergy_source='current-fold positive fit rows only',synergy_weight=1.,relation_type_count=15)
  write(d/'EXPERIMENT.json',dict(meta,recipe=recipe,source_sha256=source,variant_sources=variant,initialization_sha256=initial,graph_summary=graph['summary'],
   parameters=sum(p.numel() for p in model.parameters()),constructor_forbids_checkpoint_loading=True,baseline_metadata=raw['metadata']))
  write(d/'SPLIT.json',raw['metadata']['split'])
  model.eval();model.capture_propagation=True
  with torch.no_grad():model.knowledge_table(model.base_features())
  assert all(x['delta_abs_max']==0. and x['identity_max_error']==0. for x in model.propagation_audit['layers'])
  write(d/'INITIAL_ZERO_GATE_AUDIT.json',model.propagation_audit)
  model.capture_propagation=False
  for pair in model.trace['pairs']:
   x,y=model.trace['slots'][pair['source']],model.trace['slots'][pair['target']]
   assert x['node']==y['node']==pair['entity'] and x['family']!=y['family']
  examples=[p for p in model.trace['pairs'] if 'cell_expression' in p['descriptor'] and any(str(v).startswith('drug_') for v in p['descriptor'])][:12]
  assert examples
  write(d/'STRUCTURE.json',dict(status='passed',counts=model.trace['counts'],all_cell_roots_have_graph_tokens=True,
   cell_expression_edges=len(graph['groups']['cell_expression']),Cell_Drug_relation_combination_examples=examples))
  rows=raw['fit']['rows'].to(device);valid=raw['valid']['rows'].to(device)
  if a.stage=='check':
   write(d/'progress.json',dict(meta,status='running',phase='global_graph_interventions',epoch=0))
   # Check reachability with gates temporarily open; zero gates should
   # suppress these effects before learning. Restore exact zeros below.
   with torch.no_grad():
    for gate in model.propagation_gates:gate.fill_(.05)
   # The graph remains FULL. Only supervision/query counts are reduced.
   rows=rows[:128];valid=valid[:256]
   model.eval()
   with torch.no_grad():
    base=model.base_features();table=model.knowledge_table(base);sample=valid[:32]
    pred,weights=model(sample,table,base,True)
    split=torch.cat([model(sample[i:i+7],table,base)[0] for i in range(0,len(sample),7)])
    batch_error=float((pred-split).abs().max());assert batch_error<2e-5
    swap=sample.clone();swap[:,:2]=swap[:,[1,0]];swapped,_=model(swap,table,base)
    swap_error=float((pred-swapped).abs().max());assert swap_error<2e-5
    # Hold the prediction Query fixed; modify ONLY the Cell node fed to graph.
    altered_base=base.clone();altered_base[model.num_drug]+=torch.linspace(-.25,.25,256,device=device)
    altered_table=model.knowledge_table(altered_base)
    graph_drug_change=float((table[:model.num_drug]-altered_table[:model.num_drug]).abs().max())
    graph_cell_change=float((table[model.num_drug:]-altered_table[model.num_drug:]).abs().max())
    assert graph_drug_change>1e-8 and graph_cell_change>1e-8
    altered_prediction,_=model(sample,altered_table,base)
    expression=model.groups['cell_expression'].expression;saved=expression.clone()
    expression.copy_(saved.roll(642,0));attribute_table=model.knowledge_table(base);expression.copy_(saved)
    attribute_change=float((table-attribute_table).abs().max());assert attribute_change>1e-8
   write(d/'INTERVENTIONS.json',dict(temporary_propagation_gate=.05,batch_max_error=batch_error,drug_swap_error=swap_error,
    fixed_Query_Cell_graph_node_to_Drug_table_change=graph_drug_change,Cell_graph_table_change=graph_cell_change,
    fixed_Query_graph_prediction_change=float((pred-altered_prediction).abs().max()),
    fixed_nodes_expression_attribute_permutation_change=attribute_change))
   del table,altered_table,attribute_table,base,altered_base
   with torch.no_grad():
    for gate in model.propagation_gates:gate.zero_()
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=.01);scheduler=TruncatedExponentialLR(optimizer,.9997,min_lr=2e-6)
  before={n:p.detach().cpu().clone() for n,p in model.named_parameters()} if a.stage=='check' else None
  epochs=3 if a.stage=='check' else 3000;best=-1.;selected=None;latest_pred=None
  for epoch in range(1,epochs+1):
   model.capture_propagation=epoch<=2 or epoch%100==0
   tick=time.monotonic();loss,grads=train_epoch(model,optimizer,rows,rows[:,3].float(),chunk=32 if a.stage=='check' else 1024)
   scheduler.step();assert np.isfinite(loss)
   if model.capture_propagation:write(d/f'PROPAGATION_epoch{epoch}.json',model.propagation_audit)
   if epoch<=2 or a.stage=='check':
    assert all(np.isfinite(v) for v in grads.values()),grads
    model.verify_gradients(grads, epoch)
    write(d/f'GRADIENT_epoch{epoch}.json',grads)
   record=dict(epoch=epoch,train_BCE=loss,lr=optimizer.param_groups[0]['lr'],train_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started)
   if epoch<=2 or epoch%20==0 or epoch>=1000 or a.stage=='check':
    latest_pred=evaluate(model,valid,chunk=32 if a.stage=='check' else 1024)
    auc,ap=get_metrics(valid[:,3].cpu().numpy(),latest_pred.numpy());record.update(valid_auc=float(auc),valid_aupr=float(ap))
    if epoch>=1000 or a.stage=='check':
     if auc>best:
      best=float(auc);selected=epoch
      checkpoint=dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},recipe=recipe,epoch=epoch,source_sha256=source,
       initialization_sha256=initial,variant_sources=variant,original_checkpoint_loaded=False)
      temp=d/'selected.pt.tmp';torch.save(checkpoint,temp);temp.replace(d/'selected.pt')
      write(d/'selection.json',dict(epoch=epoch,valid_auc=float(auc),valid_aupr=float(ap),checkpoint_sha256=sha(d/'selected.pt')))
   with (d/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
   write(d/'progress.json',dict(meta,status='running',phase='training',epochs=epochs,optimizer_updates=epoch,selected_epoch=selected,**record))
   if epoch<=2 or epoch%20==0:print(record,flush=True)
  if a.stage=='check':
   changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in model.named_parameters()}
   for group,params in model.components().items():
    ids={id(p) for p in params};assert any(changes[n]>0 for n,p in model.named_parameters() if id(p) in ids),group
   checkpoint=torch.load(d/'selected.pt',weights_only=True,map_location=device);model.load_state_dict(checkpoint['model_state'],strict=True)
   replay=evaluate(model,valid,chunk=32);auc,ap=get_metrics(valid[:,3].cpu().numpy(),replay.numpy());selection=read(d/'selection.json')
   assert abs(float(auc)-selection['valid_auc'])<1e-5 and abs(float(ap)-selection['valid_aupr'])<1e-5
   write(d/'CHECKS.json',dict(status='passed',full_global_graph=True,training_rows=len(rows),optimizer_updates=3,
    all_components_nonzero_gradients_after_gate_open=True,actual_parameter_changes=changes,all_cell_roots_graph_tokens=True,
    disk_reload_verified=True,old_checkpoint_loaded=False,source_sha256=source,no_performance_claim=True,
    peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,seconds=time.monotonic()-started))
  write(d/'metrics.json',dict(read(d/'selection.json'),epochs=epochs,test_evaluated=False,no_performance_claim=a.stage=='check'))
  write(d/'progress.json',dict(meta,status='complete',epoch=epochs,epochs=epochs,optimizer_updates=epochs,selected_epoch=selected))
 except Exception as error:
  write(d/'progress.json',dict(meta,status='failed',error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()

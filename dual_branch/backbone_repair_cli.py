"""ALMANAC seed0/fold0 paired repair pilot. GPU admission belongs to shared queue."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output')
    p.add_argument('--dataset', choices=['ALMANAC'], default='ALMANAC')
    p.add_argument('--fold', type=int, choices=[0], default=0)
    p.add_argument('--knowledge', choices=['real','none'], required=True)
    p.add_argument('--stage', choices=['check','train'], default='check')
    p.add_argument('--check-rows', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--result-directory', type=Path, required=True)
    p.add_argument('--expected-manifest', type=Path)
    a=p.parse_args()
    from .common import ROOT, environment
    environment()
    os.environ['CUDA_VISIBLE_DEVICES']='' if a.device=='cpu' else a.device.split(':')[-1]
    from global_cell.common import read,write,sha,author_imports
    directory=a.result_directory.resolve()
    if ROOT not in directory.parents or not str(directory).startswith('/ifs2/'):
        raise ValueError('Output must be a new directory under IFS2 project')
    if directory.exists(): raise FileExistsError(directory)
    if a.stage=='train' and not a.expected_manifest: raise ValueError('Frozen manifest required')
    directory.mkdir(parents=True)
    meta=dict(dataset=a.dataset,fold=0,seed=0,knowledge=a.knowledge,stage=a.stage,
              pid=os.getpid(),physical_device=a.device,test_evaluated=False)
    write(directory/'progress.json',dict(meta,status='running',phase='loading'))
    try:
        import torch
        torch.set_num_threads(2)
        if a.expected_manifest:
            manifest=read(a.expected_manifest)
            for kind in ['source','inputs']:
                assert all(sha(f)==s for f,s in manifest[kind].items()),kind
        author_imports()
        from utils import set_seed_all,get_metrics,TruncatedExponentialLR
        from .cell_conditioned_data import load
        from .checks import balanced
        from .backbone_repair import BackboneRepairModel
        from .backbone_repair_training import train_epoch,evaluate,AucEarlyStop
        data=load('ALMANAC',0)
        set_seed_all(0)
        old_load,old_state=torch.load,torch.nn.Module.load_state_dict
        def forbidden(*args,**kwargs): raise AssertionError('Constructor attempted checkpoint loading')
        torch.load=torch.nn.Module.load_state_dict=forbidden
        try: model=BackboneRepairModel(data,knowledge=a.knowledge)
        finally: torch.load,torch.nn.Module.load_state_dict=old_load,old_state
        initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        assert all(p.requires_grad for p in model.parameters())
        if a.expected_manifest:
            assert initial == manifest['reference_initialization_sha256'], 'CPU/GPU or arm initialization mismatch'
        model.to('cpu' if a.device=='cpu' else 'cuda:0')
        device=next(model.parameters()).device
        optimizer=torch.optim.AdamW(model.optimizer_param_groups(),eps=1e-6)
        scheduler=TruncatedExponentialLR(optimizer,.9997,min_lr=2e-6)
        stopper=AucEarlyStop(patience=100,minimum_epochs=300)
        recipe=dict(model.prediction_recipe(),knowledge_mode=a.knowledge,epochs_max=3000,
            early_stop_minimum_epochs=300,early_stop_patience_epochs=100,selection='best validation AUROC from first evaluation',
            evaluation='epochs1/2/3 and every5',optimizer='AdamW',lr=1e-4,eps=1e-6,
            weight_decay=1e-4,gate_weight_decay=0.,gradient_clip=1.,one_update_per_epoch=True,
            no_knowledge_control='same installed parameters; KG source masked and computation skipped')
        write(directory/'EXPERIMENT.json',dict(meta,recipe=recipe,initialization_sha256=initial,
            parameters=sum(p.numel() for p in model.parameters()),input_provenance=data['input_provenance'],
            fit_only_graph=True,old_checkpoint_loaded=False,source_manifest=str(a.expected_manifest)))
        rows,valid=(data['raw'][x]['rows'].to(device) for x in ['fit','valid'])
        if a.stage=='check': rows,valid=balanced(rows,a.check_rows),balanced(valid,64)
        chunk=min(1024,max(1,len(rows)//2)) if a.stage=='check' else 1024
        epochs=3 if a.stage=='check' else 3000
        started=time.monotonic(); history=[]
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,epochs+1):
            tick=time.monotonic()
            loss,diag=train_epoch(model,optimizer,rows,rows[:,3].float(),chunk=chunk)
            seconds=time.monotonic()-tick
            if epoch<=3:
                active=['drug_2d_and_fusion','drug_3d_projection','cell_encoder','collaborative_refiner',
                        'prediction_head','fusion_gate','joint_transformer','transformer_drug_adapter','transformer_cell_adapter']
                if a.knowledge=='real': active+=['graph_type_projection','alpha_beta','relation_interaction','gamma','path_encoder']
                assert all(diag['component_gradient_L2'][k]>0 for k in active),diag
                if a.knowledge=='none':
                    assert all(diag['component_gradient_L2'][k]==0 for k in ['graph_type_projection','alpha_beta','relation_interaction','gamma','path_encoder'])
            scheduler.step()
            row=dict(epoch=epoch,train_BCE=loss,train_epoch_seconds=seconds,elapsed_seconds=time.monotonic()-started,**diag)
            stop=False
            if epoch<=3 or epoch%5==0:
                pred=evaluate(model,valid)
                auc,ap=get_metrics(valid[:,3].cpu().numpy(),pred.numpy())
                row.update(valid_auc=float(auc),valid_aupr=float(ap))
                improved,stop=stopper.observe(epoch,float(auc))
                if improved:
                    cp=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),epoch=epoch,recipe=recipe,
                        initialization_sha256=initial,correctness_only=a.stage=='check')
                    tmp=directory/'selected_auc.pt.tmp';torch.save(cp,tmp);tmp.replace(directory/'selected_auc.pt')
                    write(directory/'selected_auc.json',dict(epoch=epoch,valid_auc=float(auc),valid_aupr=float(ap),
                        checkpoint_sha256=sha(directory/'selected_auc.pt'),correctness_only=a.stage=='check'))
            with (directory/'history.jsonl').open('a') as f:
                import json
                f.write(json.dumps(row,allow_nan=False)+'\n')
            if a.stage=='check': history.append(row)
            write(directory/'progress.json',dict(meta,status='running',phase='training',**row))
            if epoch<=3 or epoch%20==0: print(dict(meta,**row),flush=True)
            if stop and a.stage=='train': break
        if a.stage=='check':
            model.eval()
            base=model.base_features()
            # Isolated Refiner-only gradient proves a live differentiable path
            # to BOTH drug and Cell modalities, independent of direct fusion.
            cf=model.collaborative_refiner(base,model.synergy_incidence,torch.ones(model.synergy_edges,device=device))
            grad=torch.autograd.grad(cf.square().mean(),base)[0]
            drug_grad=float(grad[:model.num_drug].norm());cell_grad=float(grad[model.num_drug:].norm())
            assert drug_grad>0 and cell_grad>0
            with torch.no_grad():
                table=model.knowledge_table(base)
                error=float((cf-table[model.collaborative_offset:]).abs().max())
                assert error<1e-5
                pred=model(valid,table,base)[0]
                swap=valid.clone();swap[:,:2]=swap[:,[1,0]]
                swap_error=float((model(swap,table,base)[0]-pred).abs().max())
                chunked=torch.cat([model(valid[i:i+7],table,base)[0] for i in range(0,len(valid),7)])
                chunk_error=float((chunked-pred).abs().max())
                assert max(swap_error,chunk_error)<2e-6
                _,detail=model(valid,table,base,capture=True)
                weights=detail['fusion_weights']
                if a.knowledge=='none': assert float(weights[:,:,2].abs().max())==0
                state_file=directory/'correctness_final.pt';torch.save(model.state_dict(),state_file)
                model.load_state_dict(torch.load(state_file,map_location=device,weights_only=True),strict=True)
                replay=evaluate(model,valid)
                reload_error=float((replay-pred.cpu()).abs().max());assert reload_error<2e-6
            write(directory/'CHECKS.json',dict(status='passed',initialization_sha256=initial,optimizer_updates=3,
                history=history,Refiner_modal_gradient_drug=drug_grad,Refiner_modal_gradient_cell=cell_grad,
                Refiner_author_forward_error=error,drug_exchange_error=swap_error,chunk_error=chunk_error,
                strict_reload_error=reload_error,mean_fusion_weights=weights.mean(0).cpu().tolist(),
                full_global_graph=True,supervised_rows=len(rows),validation_rows=len(valid),
                graph_tables_per_epoch=1,Refiner_BN_updates_per_epoch=1,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
                no_performance_claim=True))
        write(directory/'progress.json',dict(meta,status='complete',epochs=epoch,optimizer_updates=epoch,
            stopped_early=epoch<epochs,best_valid_auc=stopper.best,best_epoch=stopper.best_epoch,
            selected_checkpoint=str(directory/'selected_auc.pt')))
    except Exception as error:
        write(directory/'progress.json',dict(meta,status='failed',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()

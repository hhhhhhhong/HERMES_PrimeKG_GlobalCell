"""CPU full-graph correctness of the entity-adapted Transformer readout.

Two small supervised updates are correctness tests, not new fivefold fits or
performance estimates. The frozen original fivefold and its workers stay as-is.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES']=''
from global_cell.common import read,write,sha,author_imports
DISPATCH=ROOT/'artifacts/entity_token_adapter_20261122'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['ALMANAC','ONEIL'])
    args=parser.parse_args()
    directory=DISPATCH/('cpu_'+args.dataset)
    directory.mkdir(parents=True,exist_ok=False)
    write(directory/'progress.json',dict(status='running',phase='loading',pid=os.getpid()))
    try:
        import torch
        torch.set_num_threads(2)
        author_imports()
        from utils import set_seed_all
        from dual_branch.cell_conditioned_data import load
        from dual_branch.hypothesis_evidence import HypothesisEvidenceModel
        from dual_branch.entity_token_adapter import EntityAdaptedHypothesisModel,audit_entity_tokens
        from dual_branch.hypothesis_training import train_epoch,evaluate
        from dual_branch.checks import balanced
        data=load(args.dataset,0)
        def forbidden(*args,**kwargs):raise AssertionError('Constructor attempted a checkpoint load')
        set_seed_all(0)
        with patch.object(torch,'load',forbidden),patch.object(torch.nn.Module,'load_state_dict',forbidden):
            original=HypothesisEvidenceModel(data).eval()
        original_rng=torch.get_rng_state().clone()
        set_seed_all(0)
        with patch.object(torch,'load',forbidden),patch.object(torch.nn.Module,'load_state_dict',forbidden):
            model=EntityAdaptedHypothesisModel(data).eval()
        assert torch.equal(original_rng,torch.get_rng_state())
        original_state,new_state=original.state_dict(),model.state_dict()
        assert set(new_state)-set(original_state)=={
            'joint_transformer.drug_adapter.1.weight','joint_transformer.cell_adapter.1.weight'}
        for key,value in original_state.items():
            torch.testing.assert_close(new_state[key],value,rtol=0,atol=0)
        preserved=len(original_state)
        del original,original_state,new_state
        assert all(p.requires_grad for p in model.parameters())
        initial_sha=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        train=balanced(data['raw']['fit']['rows'],64)
        valid=balanced(data['raw']['valid']['rows'],64)
        captured=[]
        def capture_input(module,arguments):
            captured.append(arguments[0].detach().clone())
        def audit_current():
            captured.clear()
            handle=model.joint_transformer.register_forward_pre_hook(capture_input)
            try:evaluate(model,valid,chunk=64)
            finally:handle.remove()
            assert len(captured)==1
            return audit_entity_tokens(model.joint_transformer,captured[0])
        before_audit=audit_current()
        adapter_initial={k:p.detach().clone() for k,p in model.named_parameters() if '_adapter.' in k}
        optimizer=torch.optim.AdamW(model.optimizer_param_groups(),eps=1e-6)
        steps=[0];original_step=optimizer.step
        def count_step(*a,**kw):
            steps[0]+=1
            return original_step(*a,**kw)
        optimizer.step=count_step
        history=[]
        for epoch in range(1,3):
            started=time.monotonic()
            loss,diagnostic=train_epoch(model,optimizer,train,train[:,3].float(),chunk=32)
            assert diagnostic['updates_this_epoch']==1 and diagnostic['graph_tables_recomputed']==2
            assert all(value>0 for value in diagnostic['component_gradient_L2'].values())
            assert diagnostic['component_gradient_L2']['transformer_drug_adapter']>0
            assert diagnostic['component_gradient_L2']['transformer_cell_adapter']>0
            history.append(dict(epoch=epoch,BCE=loss,seconds=time.monotonic()-started,**diagnostic))
            write(directory/'progress.json',dict(status='running',phase='correctness_updates',optimizer_updates=epoch))
        assert steps[0]==2
        changes={k:float((p.detach()-adapter_initial[k]).norm()) for k,p in model.named_parameters() if k in adapter_initial}
        assert all(value>0 for value in changes.values())
        a,b=evaluate(model,valid,chunk=64),evaluate(model,valid,chunk=7)
        torch.testing.assert_close(a,b,rtol=0,atol=2e-6)
        swap=valid.clone();swap[:,:2]=swap[:,[1,0]]
        c=evaluate(model,swap,chunk=64)
        torch.testing.assert_close(a,c,rtol=0,atol=2e-6)
        after_audit=audit_current()
        state_file=directory/'correctness_final.pt'
        torch.save(dict(model_state=model.state_dict(),recipe=model.prediction_recipe(),correctness_only=True),state_file)
        restored=EntityAdaptedHypothesisModel(data).eval()
        restored.load_state_dict(torch.load(state_file,map_location='cpu',weights_only=True)['model_state'],strict=True)
        d=evaluate(restored,valid,chunk=64)
        torch.testing.assert_close(a,d,rtol=0,atol=0)
        manifest=read(ROOT/'artifacts/hypothesis_evidence_cv5_seed0_20261122/frozen/manifest.json')
        for kind in ['source','inputs']:
            assert all(sha(p)==digest for p,digest in manifest[kind].items()),kind
        write(directory/'CHECKS.json',dict(status='passed',dataset=args.dataset,seed=0,fold=0,
            full_global_graph=True,supervised_rows=64,validation_rows=64,query_chunk=32,
            optimizer_updates=2,actual_optimizer_step_calls=steps[0],all_downstream_trainable=True,
            old_checkpoint_loaded=False,preserved_initial_state_tensors=preserved,
            parent_initialization_and_rng_preserved=True,initial_parameter_sha256=initial_sha,
            adapter_update_L2=changes,history=history,input_audit_before=before_audit,input_audit_after=after_audit,
            drug_exchange_error=float((a-c).abs().max()),query_chunk_error=float((a-b).abs().max()),
            strict_disk_reload_error=float((a-d).abs().max()),original113_sources_and62_inputs_unchanged=True,
            old_running_folds_not_modified=True,target_masking_policy_changed=False,
            GPU_training_started=False,no_performance_claim=True))
        write(directory/'progress.json',dict(status='complete',optimizer_updates=2,correctness_only=True))
        print(dict(status='passed',dataset=args.dataset,adapter_updates=changes),flush=True)
    except Exception as error:
        write(directory/'progress.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()

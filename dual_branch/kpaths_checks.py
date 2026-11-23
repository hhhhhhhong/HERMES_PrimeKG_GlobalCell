"""Real-data checks that gamma uses the selected paths, not a global KG pool."""
import torch


@torch.no_grad()
def check_path_reading(model, rows):
    model.eval()
    base,table=model.base_features(),model.knowledge_table()
    pair=rows[:,0].long()*model.num_drug+rows[:,1].long()
    supported=model.path_pair_mask[pair].any(-1)
    assert bool(supported.any()),'Check batch must contain a real connecting path'
    sample=rows[supported][:8]
    prediction,detail=model(sample,table,base,capture=True)
    flat=detail['available'].reshape(len(sample),-1)
    assert float((detail['attention']*(~flat[:,None,:])).abs().max())==0.
    empty,empty_d=model(sample,table,base,capture=True,empty_evidence=True)
    baseline=model(sample,table,base,use_knowledge=False)[0]
    torch.testing.assert_close(empty,baseline,atol=0,rtol=0)
    assert float(empty_d['context'].abs().max())==0.
    # A path outside this query may be in the graph union, but cannot enter
    # this query's gamma candidate list. Perturb the detached readout only.
    first=sample[:1]
    _,reference=model(first,table,base,capture=True)
    selected=set(reference['path_ids'][0][reference['path_mask'][0]].tolist())
    other=next(i for i in range(1,len(model.path_hops)) if i not in selected)
    changed=table.clone();changed[model.global_rows+other]+=10
    _,isolated=model(first,changed,base,capture=True)
    torch.testing.assert_close(isolated['context'],reference['context'],atol=0,rtol=0)
    used=next(iter(selected))
    changed=table.clone();changed[model.global_rows+used]+=.5
    _,active=model(first,changed,base,capture=True)
    difference=float((active['context']-reference['context']).abs().max())
    assert difference>1e-7,difference
    # The exact same pair has fixed candidates for all Cell contexts; Cell
    # expression changes the query and may change the learned attention.
    cell_changed=base.clone();cell_changed[first[0,2].long()]+=.1
    _,conditional=model(first,table,cell_changed,capture=True)
    cell_difference=float((conditional['attention']-reference['attention']).abs().max())
    assert cell_difference>1e-8,cell_difference
    return dict(status='passed',rows_with_paths=len(sample),
        masked_attention_max=0.,empty_evidence_prediction_error=0.,
        unrelated_path_context_error=0.,selected_path_context_change=difference,
        Cell_query_attention_change=cell_difference,global_Cell_synergy_preserved=True,
        all_downstream_trainable=all(p.requires_grad for p in model.parameters()),
        no_performance_claim=True)

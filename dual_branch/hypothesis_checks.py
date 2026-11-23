"""Context-safe diagnostics for the exact-exclusion formal training entry."""
import torch
from .hypothesis_training import evaluate


@torch.no_grad()
def fusion_audit(model, rows):
    model.eval()
    with model.exclusion_context(rows):
        base = model.base_features()
        table = model.knowledge_table(base)
        _, detail = model(rows, table, base, capture=True)
        weights = detail['fusion_weights']
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=0, atol=2e-6)
        return dict(source_order=list(model.fusion.source_names),
            mean_weights_by_entity=weights.mean(0).tolist(),
            mean_entropy_by_entity=(-(weights*weights.clamp_min(1e-12).log()).sum(-1)).mean(0).tolist(),
            fraction_max_weight_above_095=float((weights.max(-1).values > .95).float().mean()),
            rows=len(rows), exclusion=dict(model.exclusion_audit), no_performance_claim=True)


@torch.no_grad()
def check_hypothesis_reading(model, rows, positive_rows):
    model.eval()
    query = positive_rows[:8]
    with model.exclusion_context(query):
        assert model.exclusion_audit['removed_edge_instances'] >= len(query)
        assert model.synergy_incidence.shape[1] == 3*model.synergy_edges
        exclusion = dict(model.exclusion_audit)
        base = model.base_features()
        table = model.knowledge_table(base)
        prediction, detail = model(query, table, base, capture=True)
        swapped = query.clone()
        swapped[:, :2] = swapped[:, [1, 0]]
        swap = model(swapped, table, base)[0]
        torch.testing.assert_close(swap, prediction, rtol=0, atol=2e-6)
        _, off = model(query, table, base, capture=True, use_collaborative_query=False)
        cf_effect = float((off['context']-detail['context']).abs().max())
        assert cf_effect > 1e-8
        _, empty = model(query, table, base, capture=True, empty_evidence=True)
        assert float(empty['fusion_weights'][:, :, 2].abs().max()) == 0.
        # Compare the actual author's forward on the reduced graph. No copied
        # or stale knowledge table is admitted to model.forward.
        initial = model.semantic.initial_nodes()[:model.nroots]
        cf = model.collaborative_refiner(initial, model.synergy_incidence,
            torch.ones(model.synergy_edges, device=base.device))
        cf_error = float((cf-table[model.collaborative_offset:]).abs().max())
        torch.testing.assert_close(cf, table[model.collaborative_offset:], rtol=0, atol=1e-5 if base.is_cuda else 0)
    try:
        model(query, table, base)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unscoped evidence table was accepted')
    full, chunked = evaluate(model, rows, chunk=1024), evaluate(model, rows, chunk=7)
    torch.testing.assert_close(full, chunked, rtol=0, atol=2e-6)
    return dict(status='passed', exact_target_exclusion=exclusion,
        CF_Query_off_context_effect=cf_effect, empty_knowledge_weight_max=0.,
        author_Refiner_forward_error=cf_error, stale_table_rejected=True,
        drug_exchange_error=float((swap-prediction).abs().max()),
        inference_query_chunk_error=float((full-chunked).abs().max()),
        fusion_gate=dict(status='passed'), no_performance_claim=True)

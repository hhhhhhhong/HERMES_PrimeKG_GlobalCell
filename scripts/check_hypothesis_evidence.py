"""CPU-only exact exclusion and collaboration-to-evidence gradient audit."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha, author_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['ALMANAC', 'ONEIL'])
    args = parser.parse_args()
    directory = ROOT / 'artifacts/hypothesis_evidence_20261122' / ('cpu_' + args.dataset)
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    write(directory / 'progress.json', dict(status='running', phase='loading', pid=os.getpid()))
    import torch
    torch.set_num_threads(2)
    author_imports()
    from utils import set_seed_all
    from dual_branch.cell_conditioned_data import load
    from dual_branch.hypothesis_evidence import HypothesisEvidenceModel
    from dual_branch.query_exclusion import QueryEdgeIndex
    from dual_branch.hypothesis_training import train_epoch, evaluate
    from dual_branch.checks import balanced
    data = load(args.dataset, 0)
    set_seed_all(0)
    def forbidden(*args, **kwargs):
        raise AssertionError('Constructor attempted a checkpoint load')
    with patch.object(torch, 'load', forbidden), patch.object(torch.nn.Module, 'load_state_dict', forbidden):
        model = HypothesisEvidenceModel(data).eval()
    assert all(p.requires_grad for p in model.parameters())
    train = balanced(data['raw']['fit']['rows'], 64)
    valid = balanced(data['raw']['valid']['rows'], 64)
    # Small but real query batch includes known training positives. No label
    # is passed to the edge-exclusion API, including during counterfactuals.
    query = train[train[:, 3] == 1][:8]
    source, index = model._source_graph, model._query_index
    _, keys, blocked = index.exclude(query)
    assert int(blocked.sum()) > 0
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    old_groups, old_incidence = model.semantic.groups, model.synergy_incidence
    with model.exclusion_context(query) as proof:
        assert proof['removed_edge_instances'] == int(blocked.sum())
        assert model.synergy_edges == len(index.edges)-int(blocked.sum())
        if 'Synergy_Hyperedge' in model.semantic.groups:
            assert model.semantic.groups['Synergy_Hyperedge'].num_edges == model.synergy_edges
        # Compare the physical-deletion oracle in IDENTICAL no-grad mode.
        # CPU Transformer/GRU inference fastpaths can differ from grad-enabled
        # execution by a float32 ULP; do not relax an exact equality assertion.
        with torch.no_grad():
            baseline = model(query, model.knowledge_table(), model.base_features())[0].clone()
        base, table = model.base_features(), model.knowledge_table()
        predictions, info = model(query, table, base, capture=True)
        _, off_info = model(query, table, base, capture=True, use_collaborative_query=False)
        cf_context_effect = float((info['context']-off_info['context']).abs().max())
        assert cf_context_effect > 1e-8
        swapped = query.clone()
        swapped[:, :2] = swapped[:, [1, 0]]
        torch.testing.assert_close(model(swapped, table, base)[0], predictions, rtol=0, atol=2e-6)
        # Loss contains ONLY the actual knowledge context, not the decoder or
        # direct CF fusion. Refiner gradients therefore prove the new route.
        contexts = []
        hook = model.gamma.register_forward_hook(lambda module, inputs, output: contexts.append(output[0]))
        try:
            model(query, table, base)
        finally:
            hook.remove()
        cf_parameters = list(model.collaborative_refiner.parameters())
        grads = torch.autograd.grad((contexts[0]*torch.randn_like(contexts[0])).sum(),
            cf_parameters+[model.gamma.query[0].weight], allow_unused=True)
        refiner_gradient = sum(float(g.square().sum()) for g in grads[:-1] if g is not None)**.5
        query_gradient = float(grads[-1][:, 3*model.semantic.dim:].norm())
        assert refiner_gradient > 0 and query_gradient > 0
        first_scope_audit = dict(model.exclusion_audit)
    assert model.semantic.groups is old_groups and model.synergy_incidence is old_incidence
    assert model._scope is None
    # The oracle literally lacks the target edges in its source graph. Fixed
    # trainable schema/state is held constant, isolating forbidden edge data.
    retained = [edge for edge, removed in zip(index.edges, blocked.tolist()) if not removed]
    altered_graphs = {
        'physically_deleted_targets': dict(source, groups=dict(source['groups'], Synergy_Hyperedge=retained)),
        'target_weights_changed_and_duplicates_added': dict(source, groups=dict(source['groups'],
            Synergy_Hyperedge=[dict(edge, weight=100000.) if removed else edge for edge, removed in zip(index.edges, blocked.tolist())]
                + [dict(index.edges[i]) for i in torch.where(blocked)[0].tolist()]))}
    oracle_errors = {}
    with torch.no_grad():
        try:
            for name, graph in altered_graphs.items():
                model._source_graph, model._query_index = graph, QueryEdgeIndex(graph)
                with model.exclusion_context(query):
                    value = model(query, model.knowledge_table(), model.base_features())[0]
                    torch.testing.assert_close(value, baseline, rtol=0, atol=0)
                    oracle_errors[name] = float((value-baseline).abs().max())
        finally:
            model._source_graph, model._query_index = source, index
        # Stale knowledge tables cannot cross graph contexts, even for the
        # SAME query keys. This guards accidental reuse by the old trainer.
        with model.exclusion_context(query):
            old_table = model.knowledge_table()
        with model.exclusion_context(query):
            new_table = model.knowledge_table()
            try:
                model(query, old_table, model.base_features())
            except RuntimeError as error:
                assert 'Stale' in str(error)
            else:
                raise AssertionError('Stale table was accepted')
        try:
            model.knowledge_table()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unmasked propagation was accepted')
        try:
            with model.exclusion_context(query):
                raise RuntimeError('intentional scope restoration check')
        except RuntimeError as error:
            assert str(error) == 'intentional scope restoration check'
        assert model.semantic.groups is old_groups and model.synergy_incidence is old_incidence
    del table, new_table, old_table, contexts, predictions
    optimizer = torch.optim.AdamW(model.optimizer_param_groups(), eps=1e-6)
    steps = [0]
    original_step = optimizer.step
    def step_once(*args, **kwargs):
        steps[0] += 1
        return original_step(*args, **kwargs)
    optimizer.step = step_once
    history = []
    for epoch in range(1, 3):
        epoch_start = time.monotonic()
        loss, diagnostic = train_epoch(model, optimizer, train, train[:, 3].float(), chunk=32)
        assert diagnostic['query_batches'] == diagnostic['graph_tables_recomputed'] == 2
        assert diagnostic['collaborative_query_weight_gradient_L2'] > 0
        assert all(value > 0 for value in diagnostic['component_gradient_L2'].values()), diagnostic
        assert model._scope is None and model.semantic.groups is old_groups
        history.append(dict(epoch=epoch, BCE=loss, elapsed_seconds=time.monotonic()-epoch_start, **diagnostic))
        write(directory / 'progress.json', dict(status='running', phase='correctness_updates', optimizer_updates=epoch))
    assert steps[0] == 2
    a, b = evaluate(model, valid, chunk=7), evaluate(model, valid, chunk=64)
    torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
    state_file = directory / 'correctness_final.pt'
    torch.save(dict(model_state=model.state_dict(), recipe=model.prediction_recipe(), correctness_only=True,
        input_provenance=data['input_provenance']), state_file)
    restored = HypothesisEvidenceModel(data).eval()
    restored.load_state_dict(torch.load(state_file, map_location='cpu', weights_only=True)['model_state'], strict=True)
    c = evaluate(restored, valid, chunk=64)
    torch.testing.assert_close(b, c, rtol=0, atol=0)
    old_source = read(ROOT / 'artifacts/hypothesis_evidence_20261122/ORIGINAL_SOURCE_SHA.json')
    assert all(sha(p) == digest for p, digest in old_source.items())
    write(directory / 'CHECKS.json', dict(status='passed', dataset=args.dataset, seed=0, fold=0,
        supervised_rows=64, validation_rows=64, optimizer_updates=2, optimizer_step_calls=steps[0],
        query_chunk=32, full_global_graph=True, history=history, initial_parameter_sha256=initial_hash,
        exclusion=first_scope_audit, forbidden_target_edge_oracles=oracle_errors,
        graph_restored_after_error=True, stale_table_rejected=True, unmasked_table_rejected=True,
        CF_Query_off_context_effect=cf_context_effect,
        evidence_context_only_loss_to_Refiner_gradient_L2=refiner_gradient,
        evidence_context_only_loss_to_CF_Query_columns_gradient_L2=query_gradient,
        inference_query_chunk_error=float((a-b).abs().max()), strict_disk_reload_error=float((b-c).abs().max()),
        all_components_finite_nonzero=True, all_downstream_trainable=True, old_checkpoint_loaded=False,
        previous98_sources_unchanged=True, GPU_used=False, formal_training=False,
        no_performance_claim=True, elapsed_seconds=time.monotonic()-started))
    write(directory / 'progress.json', dict(status='complete', optimizer_updates=2, correctness_only=True,
        elapsed_seconds=time.monotonic()-started))
    print(args.dataset, 'exact exclusion + collaborative Query CPU checks passed', flush=True)


if __name__ == '__main__':
    main()

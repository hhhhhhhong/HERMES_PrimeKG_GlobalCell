"""Real-data invariants for the two branches and staged zero-residual gradients."""
import torch


def balanced(rows, maximum=128):
    """Small correctness batch, not a performance estimate or a new data split."""
    negative = torch.where(rows[:, 3] == 0)[0][:maximum // 2]
    positive = torch.where(rows[:, 3] == 1)[0][:maximum // 2]
    assert len(negative) and len(positive)
    return rows[torch.cat((negative, positive))]


def before_updates(model, data, rows):
    """Temporary interventions are restored exactly before optimization."""
    rng = torch.get_rng_state()
    model.eval()
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        base = model.base_features()
        nodes = model.semantic.initial_nodes()
        table = model.knowledge_table(base, capture=True)
        propagation = model.semantic.propagation_audit
        assert all(abs(layer['gate_abs_max'] - .1) < 1e-6 for layer in propagation['layers'])
        assert all(layer['delta_rms'] > 0 for layer in propagation['layers'])
        # Inner propagation intentionally starts open. Check the residual's
        # identity as a separate zero-gate counterfactual, preserving that
        # guarantee without claiming the actual0.1 initialization is identity.
        original_gates = [gate.detach().clone() for gate in model.semantic.propagation_gates]
        for gate in model.semantic.propagation_gates:
            gate.zero_()
        _, zero_gate_nodes = model.semantic.propagate(nodes, capture=True)
        zero_gate_propagation = model.semantic.propagation_audit
        torch.testing.assert_close(zero_gate_nodes, nodes, rtol=0, atol=0)
        assert all(layer['delta_rms'] == layer['identity_max_error'] == 0 for layer in zero_gate_propagation['layers'])
        for gate, original in zip(model.semantic.propagation_gates, original_gates):
            gate.copy_(original)
        model.semantic.propagation_audit = propagation
        pred, details = model(rows, table, base, capture=True)
        modal, _ = model(rows, table, base, use_knowledge=False)
        assert details['delta'].abs().max().item() == 0.
        torch.testing.assert_close(pred, modal, rtol=0, atol=0)

        # Exercise the actual fusion path, not only the initially closed output.
        model.fusion.drug_out.weight.copy_(torch.eye(256, device=base.device) * .1)
        model.fusion.cell_out.weight.copy_(torch.eye(256, device=base.device) * .1)
        active, info = model(rows, table, base, capture=True)
        swapped = rows.clone()
        swapped[:, :2] = rows[:, [1, 0]]
        swap, _ = model(swapped, table, base)
        chunked = torch.cat([model(rows[i:i + 7], table, base)[0] for i in range(0, len(rows), 7)])
        torch.testing.assert_close(swap, active, rtol=0, atol=2e-6)
        torch.testing.assert_close(chunked, active, rtol=0, atol=2e-6)

        # Empty evidence is an exact identity even with nonzero fusion weights.
        ids = rows[:, :3].long()
        own = base[ids]
        empty, _ = model.gamma(own, table, torch.zeros_like(model.semantic.interaction.mask), indices=ids)
        empty_final, empty_delta, _ = model.fusion(own, empty)
        torch.testing.assert_close(empty_final, own, rtol=0, atol=0)
        assert empty_delta.abs().max().item() == 0.

        # Hold graph evidence fixed and change only the Cell modal feature.
        changed_base = base.clone()
        cell_id = int(rows[0, 2])
        changed_base[cell_id] += torch.linspace(-.5, .5, 256, device=base.device)
        _, changed_info = model(rows, table, changed_base, capture=True)
        attention_change = float((info['attention'] - changed_info['attention']).abs().max())
        assert attention_change > 1e-8

        # The semantic initialization has no dependency on the modal encoder.
        expression = model.modal.expression.detach().clone()
        model.modal.expression[0] += torch.linspace(-.5, .5, expression.shape[1], device=base.device)
        altered_modal = model.base_features()
        torch.testing.assert_close(model.semantic.initial_nodes(), nodes, rtol=0, atol=0)
        expression_change = float((altered_modal - base).abs().max())
        assert expression_change > 1e-8
        model.modal.expression.copy_(expression)

        # Root Drug/Cell CODER features actually enter the graph branch.
        root_id = int(rows[0, 0])
        coder_row = model.semantic.coder[root_id].clone()
        model.semantic.coder[root_id].copy_(coder_row.flip(0))
        perturbed_table = model.knowledge_table(base)
        coder_change = float((table - perturbed_table).abs().max())
        assert coder_change > 1e-8
        model.semantic.coder[root_id].copy_(coder_row)

        no_3d = model.base_features(use_3d=False)
        three_d_change = float((no_3d[:model.num_drug] - base[:model.num_drug]).abs().max())
        assert three_d_change > 1e-8
        # Removing 3D preserves the Cell branch exactly in evaluation mode.
        torch.testing.assert_close(no_3d[model.num_drug:], base[model.num_drug:], rtol=0, atol=0)
        missing = model.modal.drug.fragment_3d_coverage == 0
        missing_error = None
        repeat_error = None
        missing_gradient = None
        missing_projection_error = None
        token_tolerance = 1e-5 if base.is_cuda else 0.
        if missing.any():
            original_vectors = model.modal.drug.fragment_3d[missing].clone()
            tokens = model.modal.drug.tokens()
            repeat = model.modal.drug.tokens()
            repeat_error = float((tokens - repeat).abs().max())
            try:
                model.modal.drug.fragment_3d[missing] = torch.linspace(-1, 1, 512, device=base.device)
                projected = model.modal.drug.pretrained_projection(
                    model.modal.drug.pretrained_norm(model.modal.drug.fragment_3d))
                masked = projected * model.modal.drug.fragment_3d_coverage[:, None]
                missing_projection_error = float(masked[missing].abs().max())
                assert missing_projection_error == 0.
                new_tokens = model.modal.drug.tokens()
                missing_error = float((tokens - new_tokens).abs().max())
                # GINE/global_mean_pool uses CUDA scatter reductions. Repeated
                # identical forwards need not be bitwise equal; inspect that
                # numerical floor AND prove the missing 3D path is exactly cut.
                with torch.enable_grad():
                    probe = model.modal.drug.fragment_3d.detach().clone().requires_grad_(True)
                    term = model.modal.drug.pretrained_projection(
                        model.modal.drug.pretrained_norm(probe)) * model.modal.drug.fragment_3d_coverage[:, None]
                    derivative = torch.autograd.grad(term.sum(), probe)[0]
                    missing_gradient = float(derivative[missing].abs().max())
                print(dict(missing_3D_mutation_error=missing_error, identical_forward_error=repeat_error,
                    masked_projection_error=missing_projection_error, missing_input_gradient=missing_gradient,
                    fixed_float32_token_tolerance=token_tolerance), flush=True)
                assert missing_gradient == 0.
                torch.testing.assert_close(repeat, tokens, rtol=0, atol=token_tolerance)
                torch.testing.assert_close(new_tokens, tokens, rtol=0, atol=token_tolerance)
            finally:
                model.modal.drug.fragment_3d[missing] = original_vectors
        for name, p in model.named_parameters():
            p.copy_(saved[name])
        assert model.fusion.drug_out.weight.abs().max().item() == 0.
        assert model.fusion.cell_out.weight.abs().max().item() == 0.
    torch.set_rng_state(rng)
    return dict(status='passed', full_global_graph=True, initial_prediction_matches_modal_only_exactly=True,
        initial_fusion_delta_max=0., initial_propagation=propagation,
        initial_propagation_gate_value=.1, initial_propagation_is_identity=False,
        zero_gate_counterfactual=zero_gate_propagation,
        active_fusion_drug_exchange_max_error=float((swap - active).abs().max()),
        active_fusion_query_chunk_max_error=float((chunked - active).abs().max()),
        empty_evidence_delta_max=0., Cell_change_affects_gamma_attention=attention_change,
        modal_expression_change=expression_change, semantic_initialization_unchanged_by_modal_expression=True,
        Drug_CODER_change_affects_semantic_table=coder_change, actual_3D_intervention_change=three_d_change,
        zero_coverage_fragment_count=int(missing.sum()), zero_coverage_vector_mutation_max_error=missing_error,
        identical_tokens_forward_max_error=repeat_error, float32_tokens_tolerance=token_tolerance,
        zero_coverage_projected_term_max=missing_projection_error, zero_coverage_input_gradient_max=missing_gradient,
        original_state_and_CPU_random_stream_restored=True, no_performance_claim=True)


def verify_gradients(gradients, step):
    """Outer zero output learns first; the open0.1 inner gates unblock step2."""
    assert all(torch.isfinite(torch.tensor(v)) for v in gradients.values())
    assert gradients['drug_2d_and_fusion'] > 0 and gradients['drug_3d_projection'] > 0
    assert gradients['cell_encoder'] > 0 and gradients['prediction_head'] > 0
    assert gradients['fusion_output'] > 0
    if step == 1:
        for name in ['graph_type_projection', 'alpha_beta', 'propagation_gates',
                     'relation_interaction', 'gamma', 'fusion_router']:
            assert gradients[name] == 0., (name, gradients)
    else:
        assert all(v > 0 for v in gradients.values()), gradients

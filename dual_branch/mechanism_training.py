"""Explicit checkpoint policy and real-data interventions for the repaired model."""
import math
import torch


class CheckpointSelection:
    """Select AP from the FIRST evaluation; keep AUROC and legacy diagnostics."""
    def __init__(self):
        self.best_ap = self.best_auc = self.legacy_auc = -math.inf

    def consider(self, epoch, auc, ap):
        if epoch < 1 or not math.isfinite(auc) or not math.isfinite(ap):
            raise ValueError('Finite validation metrics and positive epoch required')
        names = []
        if ap > self.best_ap:
            self.best_ap = ap
            names.append('selected')
        if auc > self.best_auc:
            self.best_auc = auc
            names.append('selected_auc')
        if epoch >= 1000 and auc > self.legacy_auc:
            self.legacy_auc = auc
            names.append('legacy_selected_auc')
        return names


@torch.no_grad()
def check_mechanism_reading(model, rows):
    """Frozen-forward interventions, not separate trained performance ablations."""
    from .kpaths_checks import check_path_reading
    model.eval()
    path_checks = check_path_reading(model, rows)
    base, table = model.base_features(), model.knowledge_table()
    prediction, info = model(rows, table, base, capture=True)
    swapped = rows.clone()
    swapped[:, :2] = swapped[:, [1, 0]]
    swap = model(swapped, table, base)[0]
    chunked = torch.cat([model(rows[i:i+7], table, base)[0] for i in range(0, len(rows), 7)])
    torch.testing.assert_close(swap, prediction, rtol=0, atol=2e-6)
    torch.testing.assert_close(chunked, prediction, rtol=0, atol=2e-6)
    zero_knowledge = model(rows, table, base, use_knowledge=False)[0]
    no_collaboration = model(rows, table, base, use_collaborative=False)[0]
    assert float((zero_knowledge - prediction).abs().max()) > 1e-8
    assert float((no_collaboration - prediction).abs().max()) > 1e-8
    # Read the actual HYPER-style output slots. DTI-PPI/Pathway combinations
    # were formerly calculated and then discarded before gamma.
    composition_effects = {}
    ids = rows[:, :3].long().clone()
    ids[:, :2] = ids[:, :2].sort(dim=1).values
    for partner in ['protein_protein', 'pathway_protein']:
        for token in model.global_token_ids.tolist():
            key = model.semantic.trace['token_keys'][token]
            if key[0] != 'interaction' or {key[2], key[4]} != {'drug_target', partner}:
                continue
            supported = model.semantic.interaction.mask[ids, token].any(-1)
            if not bool(supported.any()):
                continue
            row = rows[supported][:1]
            roots = row[0, :3].long()
            root = next(int(r) for r in roots if model.semantic.interaction.mask[int(r), token])
            _, reference = model(row, table, base, capture=True)
            changed = table.clone()
            changed[root * model.relation_tokens + token] += torch.linspace(-1, 1, model.semantic.dim, device=table.device)
            _, result = model(row, changed, base, capture=True)
            effect = float((result['context'] - reference['context']).abs().max())
            assert effect > 1e-8, (partner, key, effect)
            composition_effects[partner] = dict(token_key=key, root=root, context_change=effect)
            break
        assert partner in composition_effects, 'Check rows must contain real ' + partner + ' composition witnesses'
    # Author module is invoked with unchanged equations. Compare its actual
    # readout to a freshly constructed author module with identical state.
    from HERMES.model import Refiner
    reference_refiner = Refiner(256, 256).to(base.device).eval()
    reference_refiner.load_state_dict(model.collaborative_refiner.state_dict(), strict=True)
    initial = model.semantic.initial_nodes()[:model.nroots]
    weights = torch.ones(model.synergy_edges, device=base.device)
    expected = reference_refiner(initial, model.synergy_incidence, weights)
    torch.testing.assert_close(table[model.collaborative_offset:], expected, rtol=0, atol=0)
    changed_incidence = model.synergy_incidence.clone()
    cell_slots = changed_incidence[0] >= model.num_drug
    changed_incidence[0, cell_slots] = model.num_drug + (changed_incidence[0, cell_slots] - model.num_drug + 1) % model.num_cell
    changed_cf = model.collaborative_refiner(initial, changed_incidence, weights)
    incidence_effect = float((changed_cf - expected).abs().max())
    assert incidence_effect > 1e-8
    # Missing 3D is cut after projection, including after parameter updates.
    encoder = model.modal.drug.encoder
    missing = encoder.fragment_3d_coverage == 0
    before = model.base_features()
    if bool(missing.any()):
        saved = encoder.fragment_3d[missing].clone()
        encoder.fragment_3d[missing] = torch.linspace(-1, 1, 512, device=base.device)
        try:
            after = model.base_features()
            torch.testing.assert_close(after, before, rtol=0, atol=1e-5 if base.is_cuda else 0)
        finally:
            encoder.fragment_3d[missing] = saved
    without_3d = model.base_features(use_3d=False)
    torch.testing.assert_close(before[model.num_drug:], without_3d[model.num_drug:], rtol=0, atol=0)
    assert float((before[:model.num_drug] - without_3d[:model.num_drug]).abs().max()) > 1e-8
    return dict(status='passed', path_reading=path_checks,
        composition_output_interventions=composition_effects, author_Refiner_forward_error=0.,
        real_synergy_incidence_feature_effect=incidence_effect,
        knowledge_prediction_effect=float((zero_knowledge - prediction).abs().max()),
        collaboration_prediction_effect=float((no_collaboration - prediction).abs().max()),
        drug_exchange_error=float((swap - prediction).abs().max()),
        query_chunk_error=float((chunked - prediction).abs().max()),
        missing_3D_vector_intervention_passed=True, Cell_encoder_unchanged=True,
        no_performance_claim=True)

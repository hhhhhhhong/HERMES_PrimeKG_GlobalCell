"""DrugPair x Cell conditioned, masked three-source multimodal fusion.

An adaptation inspired by gated multimodal units, not a literal GMU copy.
All graph/chemistry/Refiner operations reuse MechanismMultimodalModel. Only
the final fusion changes from concatenation to aligned, weighted sources.
"""
import torch
from torch import nn
from .mechanism_model import MechanismMultimodalModel


class QueryGatedFusion(nn.Module):
    source_names = ('modal', 'collaborative', 'knowledge')

    def __init__(self, dim=256, hidden=64, dropout=.2):
        super().__init__()
        self.dim = dim
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.drug_alignment = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU()) for _ in range(3)])
        self.cell_alignment = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU()) for _ in range(3)])
        self.query = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

        def gate():
            return nn.Sequential(nn.Linear(4 * dim + 3, hidden), nn.GELU(), nn.Linear(hidden, 3))

        self.drug_gate, self.cell_gate = gate(), gate()
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, modal, collaborative, knowledge, source_available=None, return_weights=False):
        if modal.ndim != 3 or modal.shape[1:] != (3, self.dim):
            raise ValueError('Expected [B,3,dim] entity features')
        if collaborative.shape != modal.shape or knowledge.shape != modal.shape:
            raise ValueError('All three sources must have identical shapes')
        if source_available is None:
            source_available = torch.ones((*modal.shape[:2], 3), dtype=torch.bool, device=modal.device)
        if source_available.shape != (*modal.shape[:2], 3) or source_available.dtype != torch.bool:
            raise ValueError('Expected explicit boolean [B,3,3] source availability')
        if not bool(source_available.any(-1).all()):
            raise ValueError('At least one source must be available for every entity')
        normalized = [norm(value) * source_available[:, :, i:i+1]
                      for i, (norm, value) in enumerate(zip(self.norms, (modal, collaborative, knowledge)))]
        # A/B symmetric global context; Drug A/B use the SAME gate/alignment.
        a, b, cell = normalized[0].unbind(1)
        query = self.query(torch.cat((a + b, (a - b).abs(), cell), -1))
        gate_input = torch.cat((*normalized, query[:, None].expand(-1, 3, -1), source_available.to(modal.dtype)), -1)
        logits = torch.cat((self.drug_gate(gate_input[:, :2]), self.cell_gate(gate_input[:, 2:])), 1)
        weights = logits.masked_fill(~source_available, -torch.inf).softmax(-1)
        aligned = torch.stack([
            torch.cat((self.drug_alignment[i](value[:, :2]), self.cell_alignment[i](value[:, 2:])), 1)
            for i, value in enumerate(normalized)], 2)
        fused = self.output_norm(self.dropout((weights[..., None] * aligned).sum(2)))
        return (fused, weights) if return_weights else fused


class GatedMechanismModel(MechanismMultimodalModel):
    def __init__(self, data, dim=256):
        super().__init__(data, dim)
        self.fusion = QueryGatedFusion(dim)

    def forward(self, rows, table, base, capture=False, use_knowledge=True,
                empty_evidence=False, use_collaborative=True):
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        pair = ids[:, 0] * self.num_drug + ids[:, 1]
        path_ids, path_mask = self.path_pair_paths[pair], self.path_pair_mask[pair]
        attention = available = None
        source_available = torch.ones((*own.shape[:2], 3), dtype=torch.bool, device=own.device)
        if use_knowledge:
            relations = table[:self.global_rows].reshape(self.nroots, self.relation_tokens, -1)
            root_values = relations[:, self.global_token_ids][ids]
            root_mask = self.semantic.interaction.mask[:, self.global_token_ids][ids]
            path_values = table[self.global_rows:self.collaborative_offset][path_ids]
            evidence = torch.cat((root_values, path_values[:, None].expand(-1, 3, -1, -1)), 2)
            available = torch.cat((root_mask, path_mask[:, None].expand(-1, 3, -1)), 2)
            if empty_evidence:
                available = torch.zeros_like(available)
            context, attention = self.gamma(own, evidence, available, capture)
            # Gamma's three output queries can each read all available roles.
            source_available[:, :, 2] = available.reshape(len(own), -1).any(-1)[:, None]
        else:
            context = torch.zeros_like(own)
            source_available[:, :, 2] = False
        collaborative = table[self.collaborative_offset:][ids] if use_collaborative else torch.zeros_like(own)
        source_available[:, :, 1] = use_collaborative
        fused, weights = self.fusion(own, collaborative, context, source_available, return_weights=True)
        prediction = self.decode(fused)
        detail = dict(own=own.detach(), context=context.detach(), collaborative=collaborative.detach(),
            fused_features=fused.detach(), attention=attention,
            available=available.detach() if available is not None else None,
            path_ids=path_ids.detach(), path_mask=path_mask.detach(), global_tokens=self.global_tokens,
            fusion_weights=weights.detach(), source_available=source_available.detach()) if capture else None
        return prediction, detail

    def components(self):
        result = super().components()
        result.pop('multimodal_fusion')
        result['fusion_alignment'] = list(self.fusion.drug_alignment.parameters()) + list(self.fusion.cell_alignment.parameters())
        result['fusion_gate'] = list(self.fusion.query.parameters()) + list(self.fusion.drug_gate.parameters()) + list(self.fusion.cell_gate.parameters())
        result['fusion_output_norm'] = list(self.fusion.output_norm.parameters())
        return result

    @staticmethod
    def prediction_recipe():
        return dict(MechanismMultimodalModel.prediction_recipe(),
            version='gated_mechanism_cv5_seed0_20261122',
            final_fusion='separate LN/projections -> sample-conditioned masked softmax weights -> weighted sum256',
            fusion_source_order=['modal', 'collaborative', 'knowledge'],
            fusion_query='symmetric Drug A+Drug B, absolute difference, Cell, plus each entity sources/mask',
            shared_drug_gate_and_alignment=True, independent_cell_gate_and_alignment=True,
            missing_knowledge_source_masked=True, fusion_weights_are_calibrated_confidence=False,
            gate_auxiliary_loss=False, concatenation_control_trained=False)


@torch.no_grad()
def fusion_audit(model, rows):
    model.eval()
    base, table = model.base_features(), model.knowledge_table()
    prediction, detail = model(rows, table, base, capture=True)
    weights, available = detail['fusion_weights'], detail['source_available']
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=0, atol=2e-6)
    assert float(weights.masked_select(~available).abs().sum()) == 0.
    return dict(source_order=list(model.fusion.source_names), mean_weights_by_entity=weights.mean(0).tolist(),
        min_weights_by_entity=weights.amin(0).tolist(), max_weights_by_entity=weights.amax(0).tolist(),
        mean_entropy_by_entity=(-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean(0).tolist(),
        fraction_max_weight_above_095=float((weights.max(-1).values > .95).float().mean()),
        unavailable_weight_max=0., rows=len(rows), weight_normalization_error=float((weights.sum(-1)-1).abs().max()),
        no_performance_claim=True)

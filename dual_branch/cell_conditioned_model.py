"""The existing gated model with a DrugPair x Cell path lookup before gamma.

    Global alpha/beta propagation still uses one unified graph. Only the local
    path evidence read by gamma is Cell-specific; this is not a per-query
    reconstruction of the global propagation graph or a learned selector.
"""
import torch
from .gated_mechanism import GatedMechanismModel


class CellConditionedMechanismModel(GatedMechanismModel):
    def __init__(self, data, dim=256, condition_on_cell=True):
        super().__init__(data, dim)
        self.condition_on_cell = condition_on_cell
        for name in ('cell_pair_paths', 'cell_pair_mask', 'pair_slot'):
            self.register_buffer(name, data['paths'][name].clone(), persistent=False)
        self.selection_audit = dict(self.selection_audit,
            Cell_conditioned_candidates=condition_on_cell,
            candidate_policy='expanded real path pool -> relative-expression-guided diverse Top-K',
            global_graph_rebuilt_per_query=False, learned_candidate_selector=False)

    def select_paths(self, ids):
        pair = ids[:, 0] * self.num_drug + ids[:, 1]
        cell = ids[:, 2] - self.num_drug
        if not bool(((cell >= 0) & (cell < self.num_cell)).all()):
            raise ValueError('Cell IDs must be HERMES global root IDs')
        if self.condition_on_cell:
            slot = self.pair_slot[pair]
            return self.cell_pair_paths[cell, slot], self.cell_pair_mask[cell, slot]
        return self.path_pair_paths[pair], self.path_pair_mask[pair]

    def forward(self, rows, table, base, capture=False, use_knowledge=True,
                empty_evidence=False, use_collaborative=True):
        # Equations below deliberately match GatedMechanismModel.forward;
        # only select_paths replaces its drug-pair-only lookup.
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        path_ids, path_mask = self.select_paths(ids)
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
            fusion_weights=weights.detach(), source_available=source_available.detach(),
            Cell_conditioned_candidates=self.condition_on_cell) if capture else None
        return prediction, detail

    @staticmethod
    def prediction_recipe():
        return dict(GatedMechanismModel.prediction_recipe(),
            version='cell_conditioned_retrieval_20261122',
            candidate_selection='deterministic Cell-relative expression support before gamma',
            candidate_pool=64, selected_K=10, ordered_relation_variants=2,
            learned_candidate_selector=False, global_propagation_per_query=False,
            new_auxiliary_objective=False)

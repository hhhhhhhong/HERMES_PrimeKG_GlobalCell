"""Exact query-edge exclusion plus Refiner-conditioned biological evidence readout.

The full source graph is immutable. Each training batch obtains its own
strictly reduced Synergy graph in BOTH global alpha/beta and author Refiner.
All backward passes using that graph must finish INSIDE exclusion_context.
"""
from contextlib import contextmanager
import math
import time
import torch
from torch import nn
from torch.nn import functional as F

from .numerical_safety import SafeCellConditionedMechanismModel, SafeConditionalGamma, finite_masked_softmax
from .query_exclusion import QueryEdgeIndex, query_keys, compile_excluded
from .mechanism_model import kept_synergy_incidence


class HypothesisConditionalGamma(SafeConditionalGamma):
    """Concatenate symmetric modality and normalized collaboration Query terms."""
    def __init__(self, dim=256):
        super().__init__(dim)
        self.query = nn.Sequential(nn.Linear(6*dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, base_features, table, mask, capture=False, indices=None,
                collaborative=None, use_collaborative_query=True):
        if collaborative is None or collaborative.shape != base_features.shape:
            raise ValueError('Explicit [B,3,D] held-out collaborative features required')
        b = len(base_features)
        roles = torch.tensor([0, 0, 2], device=base_features.device)
        role_e, role_o = self.evidence_role(roles), self.output_role(roles)
        if indices is None:
            tokens = table.shape[2]
            evidence = (table + role_e.view(1, 3, 1, self.dim)).reshape(b, 3*tokens, self.dim)
            keys, values = self.key(evidence), self.value(evidence)
            available = mask.reshape(b, 3*tokens)
        else:
            tokens = table.shape[1]
            keys = (self.key(table)[indices] + self.key(role_e)[None, :, None]).reshape(b, 3*tokens, self.dim)
            values = (self.value(table)[indices] + self.value(role_e)[None, :, None]).reshape(b, 3*tokens, self.dim)
            available = mask[indices].reshape(b, 3*tokens)
        a, drug_b, cell = base_features.unbind(1)
        # No detach: the evidence-reading objective also trains Refiner.
        cf = F.layer_norm(collaborative, (self.dim,)) if use_collaborative_query else torch.zeros_like(collaborative)
        ca, cb, cc = cf.unbind(1)
        joint = torch.cat((a+drug_b, (a-drug_b).abs(), cell, ca+cb, (ca-cb).abs(), cc), -1)
        query = self.query(joint)[:, None] + self.self_query(base_features) + role_o[None]
        score = torch.einsum('bod,btd->bot', query, keys) / math.sqrt(self.dim)
        perm = torch.tensor([1, 0, 2], device=base_features.device)
        bias = .5*(self.role_pair_bias + self.role_pair_bias[perm][:, perm])
        score = score + bias[:, :, None].expand(3, 3, tokens).reshape(1, 3, 3*tokens)
        weights = finite_masked_softmax(score, available[:, None])
        context = torch.einsum('bot,btd->bod', weights, values)
        return context, weights.detach() if capture else None


class HypothesisEvidenceModel(SafeCellConditionedMechanismModel):
    def __init__(self, data, dim=256, condition_on_cell=True, collaborative_query=True):
        super().__init__(data, dim, condition_on_cell=condition_on_cell)
        self.gamma = HypothesisConditionalGamma(dim)
        self.collaborative_query = collaborative_query
        self._source_graph = data['graph']
        self._query_index = QueryEdgeIndex(data['graph'])
        self._scope = None
        self._active_table_id = None
        self.exclusion_audit = {}
        self.selection_audit = dict(self.selection_audit,
            target_edges_excluded_in_both_streams=True, label_free_query_exclusion=True,
            collaboration_conditions_gamma=collaborative_query,
            candidate_retrieval_conditioned_on_collaboration=False)

    @contextmanager
    def exclusion_context(self, rows):
        if self._scope is not None:
            raise RuntimeError('Nested query-exclusion contexts are forbidden')
        started = time.monotonic()
        retained, keys, blocked = self._query_index.exclude(rows)
        groups, packed, trace = compile_excluded(self._source_graph, retained, self.semantic.trace['token_keys'])
        device = self.semantic.coder.device
        groups = groups.to(device)
        packed = {name: value.to(device) for name, value in packed.items()}
        semantic, interaction = self.semantic, self.semantic.interaction
        old_groups, old_slots = semantic.groups, semantic.slot_nodes
        old_packed = {name: getattr(interaction, name) for name in packed}
        old_incidence, old_edges = self.synergy_incidence, self.synergy_edges
        try:
            semantic.groups = groups
            semantic.slot_nodes = torch.tensor([s['node'] for s in trace['slots']], dtype=torch.long, device=device)
            for name, value in packed.items():
                setattr(interaction, name, value)
            incidence = [[node, i] for i, edge in enumerate(retained) for node in edge['members']]
            self.synergy_incidence = torch.tensor(incidence, dtype=torch.long, device=device).reshape(-1, 2).t().contiguous()
            self.synergy_edges = len(retained)
            self._scope = dict(keys=set(keys.tolist()), rows=rows[:, :3].detach().clone())
            self._active_table_id = None
            self.exclusion_audit = dict(batch_rows=len(rows), unique_queries=len(keys),
                original_positive_edges=len(blocked), removed_edge_instances=int(blocked.sum()),
                retained_edges=len(retained), both_streams_masked=True, batch_labels_read=False,
                all_duplicate_and_reversed_query_edges_removed=True,
                degrees_softmax_slots_and_routes_recomputed=True,
                compile_seconds=time.monotonic()-started)
            # Independent postcondition on the actual retained edge list.
            retained_keys = QueryEdgeIndex(dict(self._source_graph, groups=dict(self._source_graph['groups'], Synergy_Hyperedge=retained))).edge_keys
            assert not bool(torch.isin(retained_keys, keys).any())
            yield self.exclusion_audit
        finally:
            semantic.groups, semantic.slot_nodes = old_groups, old_slots
            for name, value in old_packed.items():
                setattr(interaction, name, value)
            self.synergy_incidence, self.synergy_edges = old_incidence, old_edges
            self._scope, self._active_table_id = None, None

    def _check_scope(self, rows):
        if self._scope is None:
            raise RuntimeError('Use exclusion_context(rows); unmasked global table reuse is forbidden')
        keys = set(query_keys(rows, self.num_drug, self.num_cell).tolist())
        if not keys <= self._scope['keys']:
            raise RuntimeError('Queries do not belong to this held-out graph context')

    def knowledge_table(self, base=None, capture=False, initial=None):
        if self._scope is None:
            raise RuntimeError('A query-exclusion context is required BEFORE graph propagation')
        initial = self.semantic.initial_nodes() if initial is None else initial
        slots, nodes = self.semantic.propagate(initial, capture=capture or self.semantic.capture_propagation)
        relations, _ = self.semantic.interaction(slots, capture)
        # Encode only paths actually read by this context, with original path
        # IDs/offsets retained. Unselected candidates cannot affect the GRU.
        ids = self._scope['rows'].long().to(initial.device)
        selected, mask = self.select_paths(ids)
        pids = selected[mask].unique(sorted=True)
        paths = initial.new_zeros(len(self.path_hops), self.semantic.dim)
        if len(pids):
            encoded = self.path_encoder(nodes, self.path_nodes[pids], self.path_relations[pids], self.path_hops[pids])
            paths = paths.index_copy(0, pids, encoded)
        incidence, weights = kept_synergy_incidence(self.synergy_incidence, self.synergy_edges, self.training, initial.device)
        collaboration = self.collaborative_refiner(initial[:self.nroots], incidence, weights)
        table = torch.cat((relations.reshape(-1, self.semantic.dim), paths, collaboration))
        self._active_table_id = id(table)
        return table

    def forward(self, rows, table, base, capture=False, use_knowledge=True,
                empty_evidence=False, use_collaborative=True, use_collaborative_query=None):
        self._check_scope(rows)
        if id(table) != self._active_table_id:
            raise RuntimeError('Stale/copied evidence table from another context is forbidden')
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        path_ids, path_mask = self.select_paths(ids)
        collaborative = table[self.collaborative_offset:][ids] if use_collaborative else torch.zeros_like(own)
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
            cf_query = self.collaborative_query if use_collaborative_query is None else use_collaborative_query
            context, attention = self.gamma(own, evidence, available, capture,
                collaborative=collaborative, use_collaborative_query=cf_query)
            source_available[:, :, 2] = available.reshape(len(own), -1).any(-1)[:, None]
        else:
            context = torch.zeros_like(own)
            source_available[:, :, 2] = False
        source_available[:, :, 1] = use_collaborative
        fused, weights = self.fusion(own, collaborative, context, source_available, return_weights=True)
        prediction = self.decode(fused)
        detail = dict(context=context.detach(), attention=attention,
            collaborative=collaborative.detach(), fusion_weights=weights.detach(),
            path_ids=path_ids.detach(), path_mask=path_mask.detach(),
            exclusion=dict(self.exclusion_audit)) if capture else None
        return prediction, detail

    @staticmethod
    def prediction_recipe():
        return dict(SafeCellConditionedMechanismModel.prediction_recipe(),
            version='hypothesis_evidence_exact_exclusion_20261122',
            target_exclusion='all matching canonical Drug/Drug/Cell edges, before BOTH propagation streams',
            target_exclusion_uses_batch_labels=False, Refiner_random_Dropout=.2,
            gamma_query='concat symmetric modal3D and normalized collaboration3D -> MLP',
            query_gradient_reaches_Refiner=True, graph_table_recomputed_per_training_batch=True,
            candidate_retrieval_conditioned_on_collaboration=False)

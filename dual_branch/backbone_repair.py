"""Modal-input Refiner and fixed FIT graph; isolated from active experiments.

Both arms instantiate identical parameters. `none` masks the knowledge source
and skips its computation; this matches installed capacity, not active capacity.
No validation/test labels enter the graph. Original split duplicates remain.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .entity_token_adapter import EntityAdaptedHypothesisModel
from .mechanism_model import kept_synergy_incidence


class BackboneRepairModel(EntityAdaptedHypothesisModel):
    def __init__(self, data, knowledge='real', dim=256):
        if knowledge not in ('real', 'none'):
            raise ValueError('knowledge must be real or none')
        super().__init__(data, dim=dim)
        self.knowledge_mode = knowledge
        self.table_builds = 0
        self.exclusion_audit = {'policy': 'fixed fit-positive graph; no batch target exclusion'}

    def exclusion_context(self, rows):
        raise RuntimeError('Fixed-graph repair requires backbone_repair_training, not exclusion training')

    def _check_scope(self, rows):
        # No query-dependent buffers: one table is deliberately shared by all
        # chunks, including detached gradient-accumulation leaves.
        if self._scope is not None:
            raise RuntimeError('Unexpected mutable graph context')

    def knowledge_table(self, base=None, capture=False):
        self.table_builds += 1
        base = self.base_features() if base is None else base
        if base.shape != (self.nroots, self.semantic.dim):
            raise ValueError('Refiner must receive all Drug/Cell modality roots')
        incidence, weights = kept_synergy_incidence(
            self.synergy_incidence, self.synergy_edges, self.training, base.device)
        # Key repair: chemistry/expression -> author Refiner, with no detach.
        collaboration = self.collaborative_refiner(base, incidence, weights)
        if self.knowledge_mode == 'none':
            evidence = base.new_zeros(self.collaborative_offset, self.semantic.dim)
        else:
            # Knowledge dropout must not consume the modality/readout RNG
            # stream of the matched control. Checkpoint recompute preserves RNG.
            devices = [base.device.index] if base.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                initial = self.semantic.initial_nodes()
                slots, nodes = self.semantic.propagate(initial, capture=capture)
                relations, _ = self.semantic.interaction(slots, capture)
                paths = []
                for start in range(0, len(self.path_hops), 512):
                    ids = self.path_nodes[start:start+512]
                    rel = self.path_relations[start:start+512]
                    hops = self.path_hops[start:start+512]
                    def encode(x, ids=ids, rel=rel, hops=hops):
                        return self.path_encoder(x, ids, rel, hops)
                    paths.append(checkpoint(encode, nodes, use_reentrant=False,
                        preserve_rng_state=True) if self.training and torch.is_grad_enabled() else encode(nodes))
                evidence = torch.cat((relations.reshape(-1, self.semantic.dim), torch.cat(paths)))
        return torch.cat((evidence, collaboration))

    def forward(self, rows, table, base, capture=False, **kwargs):
        # Parent ID guard applies to the table passed to this call; unlike the
        # excluded-graph model, no table is tied to a specific query batch.
        self._active_table_id = id(table)
        kwargs['use_knowledge'] = self.knowledge_mode == 'real' and kwargs.get('use_knowledge', True)
        return super().forward(rows, table, base, capture=capture, **kwargs)

    @staticmethod
    def prediction_recipe():
        return dict(version='backbone_repair_fold0_20261123',
            Refiner_input='current BRICS2D/UniMol3D Drug and original Cell-expression encoder outputs',
            Refiner='unmodified author three-layer gated HypergraphConv',
            graph='fixed current-fit positive Synergy; original split retained',
            batch_target_exclusion=False, whole_synergy_edge_dropout=.2,
            CODER='knowledge branch only', knowledge='real vs masked none, identical installed parameter schema',
            readout='existing three-source gate, entity adapters, one-layer Transformer, symmetric head',
            objective='BCE only', original_HERMES_replication=False,
            duplicate_supervision_not_eliminated=True)

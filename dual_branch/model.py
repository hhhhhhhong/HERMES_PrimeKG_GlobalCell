"""All-node semantic hypergraph plus an independent 2D/3D/expression branch.

The graph is initialized exclusively by fixed CODER inputs. Molecular and
expression representations condition the evidence readout, then receive a
zero-initialized residual. This is a new model, not the complete HERMES Refiner.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from global_cell.attention import AlphaBeta, RelationInteraction, ConditionalGamma
from global_cell.structure import compile_graph
from models.zero_gated_hgnn import ZeroGatedHypergraphModel
from .common import structural_imports


class SemanticHypergraph(nn.Module):
    # Reuse the exact existing three-round incidence/role propagation, with
    # semantic inputs for ALL nodes rather than molecular/expression roots.
    propagate = ZeroGatedHypergraphModel.propagate

    def __init__(self, graph, coder, dim=256):
        super().__init__()
        assert coder.shape == (len(graph['node_types']), 768)
        self.dim = dim
        self.register_buffer('coder', coder.clone(), persistent=False)
        self.register_buffer('node_types', graph['node_types'].clone(), persistent=False)
        self.type_projection = nn.ModuleDict({
            str(t): nn.Sequential(nn.Linear(768, dim), nn.Tanh())
            for t in range(len(graph['type_names']))
        })
        for t in self.type_projection:
            self.register_buffer('type_ids_' + t, torch.where(graph['node_types'] == int(t))[0], persistent=False)
        self.groups, packed, self.trace = compile_graph(graph)
        self.encoder = AlphaBeta(graph, self.groups, dim=dim, heads=4)
        self.extra_encoders = nn.ModuleList([AlphaBeta(graph, self.groups, dim=dim, heads=4) for _ in range(2)])
        # Nonzero inner gates let alpha/beta learn after the outer zero output
        # takes its first update. The first graph-side task gradient is still0.
        self.propagation_gates = nn.ParameterList([nn.Parameter(torch.full((1, dim), 0.1)) for _ in range(3)])
        self.register_buffer('slot_nodes', torch.tensor([s['node'] for s in self.trace['slots']], dtype=torch.long), persistent=False)
        self.interaction = RelationInteraction(packed, graph['n_roots'], len(self.trace['token_keys']), dim=dim)
        self.propagation_audit = {}
        self.capture_propagation = False
        self._checkpoint_messages()

    def _checkpoint_messages(self):
        """Recompute activations while preserving whole-edge Dropout masks."""
        for encoder in [self.encoder] + list(self.extra_encoders):
            original = encoder.forward

            def forward(nodes, groups, capture=False, _original=original, _encoder=encoder):
                if not _encoder.training or not torch.is_grad_enabled() or capture:
                    return _original(nodes, groups, capture)
                result = checkpoint(
                    lambda x: _original(x, groups, False)[0],
                    nodes,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                return result, None

            encoder.forward = forward
        original_interaction = self.interaction.forward

        def interaction(slots, capture=False):
            if not self.interaction.training or not torch.is_grad_enabled() or capture:
                return original_interaction(slots, capture)
            table = checkpoint(
                lambda x: original_interaction(x, False)[0],
                slots,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            return table, None

        self.interaction.forward = interaction

    def initial_nodes(self):
        result = self.coder.new_zeros((len(self.coder), self.dim))
        for t, projection in self.type_projection.items():
            ids = getattr(self, 'type_ids_' + t)
            result = result.index_copy(0, ids, projection(self.coder[ids]))
        return result

    def forward(self, capture=False):
        slots, _ = self.propagate(self.initial_nodes(), capture=capture or self.capture_propagation)
        table, _ = self.interaction(slots, capture)
        return table


class ModalEncoder(nn.Module):
    """Reuse audited BRICS/Uni-Mol code and the original HERMES Cell operations."""

    def __init__(self, raw, chemistry, unimol):
        super().__init__()
        from HERMES.model import Initializer
        original = Initializer(75, raw['cell_input_dim'], 256, use_disease=False)
        # Retain only the used Cell modules; do not leave an unused Drug GTN.
        self.cell_first, self.cell_same = original.cline_first, original.cline_same
        self.drug = structural_imports()(chemistry, unimol)
        self.register_buffer('expression', raw['inputs']['cells'].clone(), persistent=False)

    def forward(self, use_3d=True):
        drug = self.drug(use_3d=use_3d)
        cell = self.cell_first(self.expression)
        for layer in self.cell_same:
            cell = cell + layer(cell)
        return torch.cat((drug, cell))


class ConditionalResidual(nn.Module):
    """Shared Drug parameters preserve A/B symmetry; Cell has its own role.

    Zero the output projections only. Routers and upstream representations
    are nonzero so these projections can learn in the first optimizer step.
    RMS-based scaling attenuates small knowledge vectors before injection.
    The learned norm is not a calibrated confidence or a residual RMS bound.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.drug_router = nn.Linear(3 * dim, 1)
        self.cell_router = nn.Linear(2 * dim, 1)
        self.drug_out = nn.Linear(dim, dim, bias=False)
        self.cell_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.drug_out.weight)
        nn.init.zeros_(self.cell_out.weight)

    def forward(self, own, knowledge):
        cell = own[:, 2:, :]
        drug_gate = self.drug_router(torch.cat((own[:, :2], knowledge[:, :2], cell.expand(-1, 2, -1)), -1)).sigmoid()
        cell_gate = self.cell_router(torch.cat((cell, knowledge[:, 2:]), -1)).sigmoid()

        # Preserve magnitude information; do not interpret it as confidence.
        evidence_scale = torch.tanh(knowledge.norm(dim=-1, keepdim=True) / math.sqrt(self.dim))
        normalized = F.layer_norm(knowledge, (self.dim,)) * evidence_scale

        delta = torch.cat(
            (drug_gate * self.drug_out(normalized[:, :2]), cell_gate * self.cell_out(normalized[:, 2:])),
            dim=1,
        )
        return own + delta, delta, torch.cat((drug_gate, cell_gate), dim=1)


class DualBranchModel(nn.Module):
    def __init__(self, data, dim=256):
        super().__init__()
        if dim != 256:
            raise ValueError('Drug2D3DEncoder and original Cell MLP output256; this model requires dim=256')
        from HERMES.model import Consolidator
        self.modal = ModalEncoder(data['raw'], data['chemistry'], data['unimol'])
        self.semantic = SemanticHypergraph(data['graph'], data['semantic'], dim=dim)
        self.gamma = ConditionalGamma(dim=dim)
        self.fusion = ConditionalResidual(dim=dim)
        self.head = Consolidator(dim)
        self.num_drug = data['raw']['num_drug']
        self.num_cell = data['raw']['num_cell']

    def base_features(self, use_3d=True):
        return self.modal(use_3d=use_3d)

    def knowledge_table(self, base=None, capture=False):
        # The API matches the shared exact-full-fit trainer. `base` is unused
        # deliberately: molecular representations never initialize graph nodes.
        return self.semantic(capture=capture)

    def decode(self, own):
        a, b, cell = own.unbind(1)
        return 0.5 * (self.head.forward_once(a, b, cell) + self.head.forward_once(b, a, cell)).reshape(-1)

    def forward(self, rows, table, base, capture=False, use_knowledge=True):
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        if not use_knowledge:
            return self.decode(own), None
        context, attention = self.gamma(own, table, self.semantic.interaction.mask, capture, indices=ids)
        final, delta, gates = self.fusion(own, context)
        details = (
            dict(
                attention=attention,
                delta=delta.detach(),
                gates=gates.detach(),
                own=own.detach(),
                context=context.detach(),
            )
            if capture
            else None
        )
        return self.decode(final), details

    def components(self):
        drug_3d = list(self.modal.drug.pretrained_projection.parameters())
        ids = {id(p) for p in drug_3d}
        return dict(
            drug_2d_and_fusion=[p for p in self.modal.drug.parameters() if id(p) not in ids],
            drug_3d_projection=drug_3d,
            cell_encoder=list(self.modal.cell_first.parameters()) + list(self.modal.cell_same.parameters()),
            graph_type_projection=list(self.semantic.type_projection.parameters()),
            alpha_beta=[p for e in [self.semantic.encoder] + list(self.semantic.extra_encoders) for p in e.parameters()],
            propagation_gates=list(self.semantic.propagation_gates),
            relation_interaction=list(self.semantic.interaction.parameters()),
            gamma=list(self.gamma.parameters()),
            fusion_output=list(self.fusion.drug_out.parameters()) + list(self.fusion.cell_out.parameters()),
            fusion_router=list(self.fusion.drug_router.parameters()) + list(self.fusion.cell_router.parameters()),
            prediction_head=list(self.head.parameters()),
        )

    def optimizer_param_groups(self, base_lr=1e-3, knowledge_lr_mult=2.0, weight_decay=1e-4):
        """Helper param groups ensuring zero-init/gate layers have weight_decay=0."""
        c = self.components()
        groups = [
            {"name": "modal_and_head", "params": c["drug_2d_and_fusion"] + c["drug_3d_projection"] + c["cell_encoder"] + c["prediction_head"],
             "lr": base_lr, "weight_decay": weight_decay},
            {"name": "knowledge", "params": c["graph_type_projection"] + c["alpha_beta"] + c["relation_interaction"] + c["gamma"] + c["fusion_router"],
             "lr": base_lr * knowledge_lr_mult, "weight_decay": weight_decay},
            {"name": "zero_decay_gate_and_output", "params": c["propagation_gates"] + c["fusion_output"],
             "lr": base_lr * knowledge_lr_mult, "weight_decay": 0.0},
        ]
        flattened = [id(parameter) for group in groups for parameter in group['params']]
        assert len(flattened) == len(set(flattened))
        assert set(flattened) == {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        return groups

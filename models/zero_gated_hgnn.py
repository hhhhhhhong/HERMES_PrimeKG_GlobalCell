"""Three real alpha/beta propagation rounds with zero-initialized gates.

This is a task adaptation of gated hypergraph propagation, not the author's
HypergraphConv implementation. Actual typed incidences and role slots are
preserved; no dense similarity graph is constructed.
"""
import torch
from torch import nn
from torch_scatter import scatter

from global_cell.attention import AlphaBeta
from global_cell.network import GlobalCellModel


class ZeroGatedHypergraphModel(GlobalCellModel):
    def __init__(self, cache, graph, coder):
        super().__init__(cache, graph, coder)
        # Keep every existing parameter's seed0 initialization. Added rounds
        # initialize only after the original constructor has finished.
        self.extra_encoders = nn.ModuleList([
            AlphaBeta(graph, self.groups, dim=256, heads=4) for _ in range(2)
        ])
        self.propagation_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 256)) for _ in range(3)
        ])
        self.register_buffer('slot_nodes', torch.tensor([
            slot['node'] for slot in self.trace['slots']
        ], dtype=torch.long), persistent=False)
        self.capture_propagation = False
        self.propagation_audit = {}

    def propagate(self, x, capture=False):
        """Iterate full Drug/Cell/external node state, then retain role slots.

        F_l = alpha/beta over real incidences at the current node state.
        Slot residuals retain relation/role-specific information for HYPER.
        Mean reduction is only the node-state feedback between rounds; it is
        over that node's actual incident channels, never over all graph nodes.
        The mean of final role slots equals the final updated state at each
        active node. Every layer is exactly identity when its gate is zero.
        """
        h = x
        rounds = [self.encoder] + list(self.extra_encoders)
        audit = []
        for layer, (encoder, gate) in enumerate(zip(rounds, self.propagation_gates)):
            messages, _ = encoder(h, self.groups, capture=False)
            gated_messages = torch.tanh(gate) * messages
            # Feedback includes intermediate proteins and functional anchors,
            # enabling genuine multi-hop propagation in the following round.
            delta = scatter(gated_messages, self.slot_nodes, dim=0,
                dim_size=len(h), reduce='mean')
            slots = h[self.slot_nodes] + gated_messages
            next_h = h + delta
            if capture:
                audit.append(dict(layer=layer+1,
                    gate_abs_max=float(gate.detach().abs().max()),
                    delta_rms=float(delta.detach().square().mean().sqrt()),
                    delta_abs_max=float(delta.detach().abs().max()),
                    input_rms=float(h.detach().square().mean().sqrt()),
                    identity_max_error=float((next_h.detach()-h.detach()).abs().max())))
            h = next_h
        if capture:
            self.propagation_audit = dict(layers=audit,
                layer_count=3, real_incidence_only=True,
                external_intermediate_nodes_updated=True,
                normalization_on_identity_branch=False)
        return slots, h

    def knowledge_table(self, base, capture=False):
        external = base.new_zeros((self.external_count, 256))
        for t, projection in self.external_projection.items():
            ids = getattr(self, 'external_ids_' + t)
            external = external.index_copy(0, ids, projection(self.coder_features[ids]))
        slots, _ = self.propagate(torch.cat((base, external)),
            capture=capture or self.capture_propagation)
        table, _ = self.interaction(slots, capture)
        return table

    def components(self):
        return dict(super().components(),
            extra_alpha_beta=list(self.extra_encoders.parameters()),
            propagation_gates=list(self.propagation_gates))

    def verify_gradients(self, gradients, epoch):
        """First-step zero task gradients are expected behind zero gates.

        A nonzero message branch lets the gates learn first. Never zero both
        gate and message projection, or fabricate positive gradient reports.
        Adam's existing weight decay can update branch weights even when
        their supervised gradient is initially zero.
        """
        propagation = {'alpha', 'beta', 'node_to_edge', 'edge_to_node',
                       'expression_edge', 'extra_alpha_beta'}
        if epoch == 1:
            assert all(gradients[name] == 0 for name in propagation), gradients
            assert gradients['propagation_gates'] > 0, gradients
            assert all(v > 0 for name, v in gradients.items() if name not in propagation), gradients
        else:
            assert all(v > 0 for v in gradients.values()), gradients

"""Bounded residuals with the original dual-branch parameters and Cell inputs."""
import math
import torch

from .model import DualBranchModel
from .transformer_prediction import TransformerDualBranchModel
from .knowledge_selection import select_graph
from global_cell.structure import compile_graph


def bounded_delta(delta, reference, fraction):
    """Per-token RMS cap; reference scale has no gradient and zero stays zero.

    torch.linalg.vector_norm has a defined zero subgradient, unlike explicitly
    taking sqrt(mean(x*x)) at zero. Clamping the denominator preserves a finite
    first-step gradient through zero-initialized output projections.
    """
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    cap = fraction * torch.linalg.vector_norm(reference.detach(), dim=-1, keepdim=True)
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return delta * (cap / length.clamp_min(1e-12)).clamp(max=1.)


class StableTransformerModel(TransformerDualBranchModel):
    """No new learned modules. All old parameters stay trainable from scratch.

    Both residual caps use the SAME pre-knowledge modal token as their scale:
    ||delta_KG|| <= .25||modal|| and ||delta_TF|| <= .10||modal||. Consequently
    their combined norm is at most .35||modal||, subject to floating error.
    This limits residual amplitude, not the movement of trainable modal weights
    or the validation risk. It does not restore the original HERMES Refiner.
    """
    knowledge_fraction = .25
    transformer_fraction = .10

    def __init__(self, data, knowledge_mode='full', dim=256):
        if knowledge_mode not in ('full', 'selected'):
            raise ValueError('Unknown knowledge mode')
        super().__init__(data, dim=dim)
        self.knowledge_mode = knowledge_mode
        self.selection_audit = dict(policy='unchanged original graph')
        if knowledge_mode == 'selected':
            graph, self.selection_audit = select_graph(data['graph'])
            self._bind_selected_graph(graph)

    def _bind_selected_graph(self, graph):
        """Replace factual buffers, preserving the full parameter/token universe.

        Initializing the same original modules first gives full/selected pilots
        identical named parameter tensors, initialization and capacity. Missing
        evidence channels remain masked; no dummy biological edges are added.
        """
        semantic = self.semantic
        original_tokens = semantic.trace['token_keys']
        index = {key: i for i, key in enumerate(original_tokens)}
        groups, packed, trace = compile_graph(graph)
        remap = torch.tensor([index[key] for key in trace['token_keys']], dtype=torch.long)
        small_count, full_count = len(remap), len(original_tokens)
        packed['group'] = (packed['group'] // small_count) * full_count + remap[packed['group'] % small_count]
        packed['token'] = remap[packed['token']]
        mask = torch.zeros((graph['n_roots'], full_count), dtype=torch.bool)
        mask.reshape(-1)[packed['group']] = True
        packed['mask'] = mask
        semantic.groups = groups
        semantic.slot_nodes = torch.tensor([s['node'] for s in trace['slots']], dtype=torch.long)
        for name, value in packed.items():
            assert name in semantic.interaction._buffers
            setattr(semantic.interaction, name, value)
        trace['selected_token_keys'] = trace['token_keys']
        trace['token_keys'] = original_tokens
        trace['counts']['selected_token_types'] = small_count
        trace['counts']['token_types'] = full_count
        semantic.trace = trace

    def _decode_bounded(self, features, modal_reference):
        _, raw_delta = self.prediction_transformer(features, return_delta=True)
        delta = bounded_delta(raw_delta, modal_reference, self.transformer_fraction)
        return DualBranchModel.decode(self, features + delta), delta, raw_delta

    def decode(self, own):
        return self._decode_bounded(own, own)[0]

    def forward(self, rows, table, base, capture=False, use_knowledge=True):
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        if use_knowledge:
            context, attention = self.gamma(own, table, self.semantic.interaction.mask, capture, indices=ids)
            _, raw_delta, gates = self.fusion(own, context)
            delta = bounded_delta(raw_delta, own, self.knowledge_fraction)
        else:
            context, attention, gates = torch.zeros_like(own), None, None
            delta = raw_delta = torch.zeros_like(own)
        prediction, tf_delta, tf_raw = self._decode_bounded(own + delta, own)
        detail = dict(own=own.detach(), delta=delta.detach(), raw_delta=raw_delta.detach(),
            transformer_delta=tf_delta.detach(), raw_transformer_delta=tf_raw.detach(),
            context=context.detach(), gates=gates.detach() if gates is not None else None,
            attention=attention) if capture else None
        return prediction, detail

    def optimizer_param_groups(self, base_lr=1e-4, knowledge_lr_mult=1., weight_decay=1e-4):
        return super().optimizer_param_groups(base_lr=base_lr,
            knowledge_lr_mult=knowledge_lr_mult, weight_decay=weight_decay)

"""Finite masked softmax for future gated/Cell-conditioned model versions.

The old training modules remain immutable. This module replaces the fusion
and gamma implementations without changing parameters, state keys or the
constructor's random stream. Synergy Dropout/Refiner equations are unchanged.
"""
import math
import torch

from global_cell.attention import ConditionalGamma
from .gated_mechanism import QueryGatedFusion, GatedMechanismModel
from .cell_conditioned_model import CellConditionedMechanismModel


def finite_masked_softmax(logits, available, dim=-1):
    """FP32 normalization for fp16/bf16; fully masked rows return exact zero.

    Masked positions get a finite dtype-aware floor. Masking after softmax
    followed by renormalization gives zero unavailable weights/gradients.
    Truly nonfinite AVAILABLE logits fail explicitly instead of being hidden.
    Masked NaN/Inf logits are ignored, as those positions are unavailable.
    """
    if not logits.is_floating_point() or available.dtype != torch.bool:
        raise TypeError('Floating logits and a boolean availability mask required')
    available = torch.broadcast_to(available, logits.shape)
    if not bool(torch.isfinite(logits.masked_fill(~available, 0.)).all()):
        raise FloatingPointError('Nonfinite available attention logits')
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    masked = work.masked_fill(~available, torch.finfo(work.dtype).min)
    weights = masked.softmax(dim) * available.to(work.dtype)
    denominator = weights.sum(dim, keepdim=True).clamp_min(torch.finfo(work.dtype).tiny)
    return (weights / denominator).to(logits.dtype)


class SafeQueryGatedFusion(QueryGatedFusion):
    """Same gates and alignments; finite masked normalization is the change."""
    def forward(self, modal, collaborative, knowledge, source_available=None, return_weights=False):
        if modal.ndim != 3 or modal.shape[1:] != (3, self.dim):
            raise ValueError('Expected [B,3,dim] entity features')
        if collaborative.shape != modal.shape or knowledge.shape != modal.shape:
            raise ValueError('All three sources must have identical shapes')
        if source_available is None:
            source_available = torch.ones((*modal.shape[:2], 3), dtype=torch.bool, device=modal.device)
        if source_available.shape != (*modal.shape[:2], 3) or source_available.dtype != torch.bool:
            raise ValueError('Expected explicit boolean [B,3,3] source availability')
        # Modal is always available in the actual model. Keep the contract
        # check: helper-level zero handling must not silently legitimize a
        # malformed entity with no observed or inferred input source.
        if not bool(source_available.any(-1).all()):
            raise ValueError('At least one source must be available for every entity')
        normalized = [norm(value) * source_available[:, :, i:i+1]
                      for i, (norm, value) in enumerate(zip(self.norms, (modal, collaborative, knowledge)))]
        a, b, cell = normalized[0].unbind(1)
        query = self.query(torch.cat((a + b, (a - b).abs(), cell), -1))
        gate_input = torch.cat((*normalized, query[:, None].expand(-1, 3, -1), source_available.to(modal.dtype)), -1)
        logits = torch.cat((self.drug_gate(gate_input[:, :2]), self.cell_gate(gate_input[:, 2:])), 1)
        weights = finite_masked_softmax(logits, source_available)
        aligned = torch.stack([
            torch.cat((self.drug_alignment[i](value[:, :2]), self.cell_alignment[i](value[:, 2:])), 1)
            for i, value in enumerate(normalized)], 2)
        fused = self.output_norm(self.dropout((weights[..., None] * aligned).sum(2)))
        return (fused, weights) if return_weights else fused


class SafeConditionalGamma(ConditionalGamma):
    """Preserve gamma equations; avoid its literal -1e9 overflowing in fp16."""
    def forward(self, base_features, table, mask, capture=False, indices=None):
        b = len(base_features)
        sym_roles = torch.tensor([0, 0, 2], device=base_features.device)
        role_e, role_o = self.evidence_role(sym_roles), self.output_role(sym_roles)
        if indices is None:
            tokens = table.shape[2]
            evidence = (table + role_e.view(1, 3, 1, self.dim)).reshape(b, 3*tokens, self.dim)
            keys, values = self.key(evidence), self.value(evidence)
            available = mask.reshape(b, 3*tokens)
        else:
            tokens = table.shape[1]
            keys = (self.key(table)[indices] + self.key(role_e)[None, :, None, :]).reshape(b, 3*tokens, self.dim)
            values = (self.value(table)[indices] + self.value(role_e)[None, :, None, :]).reshape(b, 3*tokens, self.dim)
            available = mask[indices].reshape(b, 3*tokens)
        a, drug_b, cell = base_features.unbind(1)
        sym_triad = torch.cat((a + drug_b, (a - drug_b).abs(), cell), -1)
        query = self.query(sym_triad)[:, None] + self.self_query(base_features) + role_o[None]
        score = torch.einsum('bod,btd->bot', query, keys) / math.sqrt(self.dim)
        perm = torch.tensor([1, 0, 2], device=base_features.device)
        bias = .5 * (self.role_pair_bias + self.role_pair_bias[perm][:, perm])
        score = score + bias[:, :, None].expand(3, 3, tokens).reshape(1, 3, 3*tokens)
        weights = finite_masked_softmax(score, available[:, None])
        knowledge = torch.einsum('bot,btd->bod', weights, values)
        return knowledge, weights.detach() if capture else None


def _same_state_replacement(original, cls, **kwargs):
    """Copy the just-created module's state, with no prior checkpoint access.

    fork_rng preserves initialization/dropout RNG of the parent model. Direct
    tensor copies also work under the existing constructor's checkpoint-load
    prohibition. No optimizer has been created when replacements are installed.
    """
    reference = next(original.parameters())
    with torch.random.fork_rng(devices=[]):
        replacement = cls(**kwargs).to(device=reference.device, dtype=reference.dtype)
    before, after = original.state_dict(), replacement.state_dict()
    if before.keys() != after.keys():
        raise ValueError('Numerical-only replacement changed state keys')
    with torch.no_grad():
        for name, value in before.items():
            after[name].copy_(value)
    replacement.train(original.training)
    return replacement


def install_safe_attention(model):
    model.fusion = _same_state_replacement(model.fusion, SafeQueryGatedFusion,
        dim=model.fusion.dim, hidden=model.fusion.drug_gate[0].out_features, dropout=model.fusion.dropout.p)
    model.gamma = _same_state_replacement(model.gamma, SafeConditionalGamma, dim=model.gamma.dim)


class SafeGatedMechanismModel(GatedMechanismModel):
    def __init__(self, data, dim=256):
        super().__init__(data, dim)
        install_safe_attention(self)

    @staticmethod
    def prediction_recipe():
        return dict(GatedMechanismModel.prediction_recipe(), version='numerical_safety_gated_20261122',
            masked_softmax='finite floor; FP32 normalization for fp16/bf16; mask and renormalize',
            Refiner_weights_rescaled=False, Synergy_Dropout_policy_changed=False)


class SafeCellConditionedMechanismModel(CellConditionedMechanismModel):
    def __init__(self, data, dim=256, condition_on_cell=True):
        super().__init__(data, dim, condition_on_cell=condition_on_cell)
        install_safe_attention(self)

    @staticmethod
    def prediction_recipe():
        return dict(CellConditionedMechanismModel.prediction_recipe(), version='numerical_safety_Cell_20261122',
            masked_softmax='finite floor; FP32 normalization for fp16/bf16; mask and renormalize',
            Refiner_weights_rescaled=False, Synergy_Dropout_policy_changed=False)

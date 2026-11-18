"""Activation recomputation only: identical model parameters and equations."""
import torch
from torch.utils.checkpoint import checkpoint

from models.zero_gated_hgnn import ZeroGatedHypergraphModel


class CheckpointedZeroGatedModel(ZeroGatedHypergraphModel):
    def __init__(self, cache, graph, coder):
        super().__init__(cache, graph, coder)
        for encoder in [self.encoder] + list(self.extra_encoders):
            original = encoder.forward

            def forward(x, groups, capture=False, _original=original, _encoder=encoder):
                if not _encoder.training or not torch.is_grad_enabled() or capture:
                    return _original(x, groups, capture)

                def compute(nodes):
                    return _original(nodes, groups, False)[0]

                # AlphaBeta has no running-statistics layers. Preserve both
                # CPU/CUDA RNG so synergy Dropout masks match on recomputation.
                slots = checkpoint(compute, x, use_reentrant=False, preserve_rng_state=True)
                return slots, None

            encoder.forward = forward

        original_interaction = self.interaction.forward

        def interaction_forward(slots, capture=False):
            if not self.interaction.training or not torch.is_grad_enabled() or capture:
                return original_interaction(slots, capture)

            def compute(current_slots):
                return original_interaction(current_slots, False)[0]

            return checkpoint(compute, slots, use_reentrant=False, preserve_rng_state=True), None

        self.interaction.forward = interaction_forward

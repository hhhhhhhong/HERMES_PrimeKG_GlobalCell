"""Optional Drug/Drug/Cell Transformer immediately before the original head.

This module does not change Cell inputs, PrimeKG retrieval, or the frozen V2
training source. Import TransformerDualBranchModel explicitly to enable it.
"""
import torch
from torch import nn

from .model import DualBranchModel


class PredictionTokenInteraction(nn.Module):
    """Exchange-equivariant interaction of three tokens with a zero residual.

    Drug A and Drug B share a role embedding. There are no sequence-position
    embeddings: swapping the drugs must not change the meaning of the sample.
    batch_first=True keeps attention within a sample, never across samples.
    """

    def __init__(self, dim=256, heads=4, feedforward_dim=512, dropout=0.2):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('dim must be positive and divisible by heads')
        if feedforward_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError('invalid feedforward dimension or dropout')
        self.dim = dim
        self.role = nn.Embedding(2, dim)
        self.register_buffer('role_ids', torch.tensor([0, 0, 1]), persistent=False)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.role.weight, std=0.02)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, return_delta=False):
        if features.ndim != 3 or features.shape[1:] != (3, self.dim):
            raise ValueError('expected [batch, 3, %d] Drug/Drug/Cell tokens' % self.dim)
        tokens = features + self.role(self.role_ids).unsqueeze(0)
        # Subtract the Transformer input to isolate the learned interaction.
        # Crucially, no LayerNorm or role embedding is applied to the identity
        # path. W_output=0 therefore preserves the input exactly initially.
        interaction = self.encoder(tokens) - tokens
        delta = self.output(interaction)
        refined = features + delta
        return (refined, delta) if return_delta else refined


class TransformerDualBranchModel(DualBranchModel):
    """Current dual branches + token interaction + unchanged HERMES readout.

    All downstream parameters are trainable from scratch. Zero initialization
    preserves the CURRENT dual-branch prediction initially; it does not restore
    the original HERMES Refiner or guarantee a bound on the trained residual.
    """

    def __init__(self, data, dim=256):
        super().__init__(data, dim=dim)
        # Preserve the base model's initialization and subsequent CPU random
        # stream. Construction is on CPU, before the caller moves the model.
        with torch.random.fork_rng(devices=[]):
            self.prediction_transformer = PredictionTokenInteraction(dim=dim)

    def decode(self, own):
        refined = self.prediction_transformer(own)
        # The inherited readout averages both Drug orders. It receives the
        # same three 256-dimensional vectors as before, after interaction.
        return super().decode(refined)

    def components(self):
        components = super().components()
        block = self.prediction_transformer
        components['prediction_transformer'] = (
            list(block.role.parameters()) + list(block.encoder.parameters())
        )
        components['prediction_transformer_output'] = list(block.output.parameters())
        return components

    def optimizer_param_groups(self, base_lr=1e-3, knowledge_lr_mult=2.0,
                               weight_decay=1e-4):
        """Keep all original groups, with separate new body/output groups.

        The Transformer uses the prediction-head learning rate. Its zero output
        has no weight decay, matching the existing zero-residual convention.
        """
        c = self.components()
        groups = [
            dict(name='modal_and_head',
                 params=c['drug_2d_and_fusion'] + c['drug_3d_projection']
                 + c['cell_encoder'] + c['prediction_head'],
                 lr=base_lr, weight_decay=weight_decay),
            dict(name='knowledge',
                 params=c['graph_type_projection'] + c['alpha_beta']
                 + c['relation_interaction'] + c['gamma'] + c['fusion_router'],
                 lr=base_lr * knowledge_lr_mult, weight_decay=weight_decay),
            dict(name='zero_decay_gate_and_output',
                 params=c['propagation_gates'] + c['fusion_output'],
                 lr=base_lr * knowledge_lr_mult, weight_decay=0.0),
            dict(name='prediction_transformer', params=c['prediction_transformer'],
                 lr=base_lr, weight_decay=weight_decay),
            dict(name='prediction_transformer_output',
                 params=c['prediction_transformer_output'],
                 lr=base_lr, weight_decay=0.0),
        ]
        actual = [id(p) for group in groups for p in group['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        assert len(actual) == len(set(actual)) and set(actual) == expected
        return groups

    @staticmethod
    def prediction_recipe():
        return dict(
            version='dual_branch_prediction_transformer_20261120',
            input='knowledge-fused Drug A, Drug B, Cell; [B,3,256]',
            layers=1, heads=4, dim=256, feedforward_dim=512, dropout=0.2,
            norm_first=True, shared_drug_role=True, positional_encoding=False,
            residual='features + zero_initialized_output(Transformer(tokens)-tokens)',
            original_Consolidator_preserved=True, symmetric_readout=True,
            Cell_matrix_unchanged=True, PrimeKG_selection_unchanged=True,
            old_checkpoint_loaded=False,
        )

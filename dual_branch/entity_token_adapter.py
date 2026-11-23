"""Entity-specific adapters between three-source fusion and the Transformer.

The inputs are Drug A / Drug B / Cell ENTITY tokens, not the three source
branches. Upstream modal, collaborative and knowledge fusion stays unchanged.
Drug A/B share one map; Cell has its own map. All normalization operates on
features of one token, never across samples or across the three entities.
"""
import math
import torch
from torch import nn
from .multimodal_fusion import JointTokenInteraction
from .hypothesis_evidence import HypothesisEvidenceModel


class EntityAdaptedTokenInteraction(JointTokenInteraction):
    """Learn per-entity maps, then use the existing one-layer Transformer.

    Non-affine LayerNorm gives each projected token a comparable scale while
    leaving channel semantics to the learned projections. No extra dropout,
    residual fusion, CLS token or attention layer is introduced. Equal drug
    roles and a shared drug adapter preserve exchange equivariance in eval.
    """
    def __init__(self, dim=256, heads=4, dropout=.2):
        super().__init__(dim=dim, heads=heads, dropout=dropout)

        def adapter():
            return nn.Sequential(
                nn.LayerNorm(dim, elementwise_affine=False),
                nn.Linear(dim, dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(dim, elementwise_affine=False),
            )

        self.drug_adapter = adapter()
        self.cell_adapter = adapter()

    def adapt(self, features):
        if features.ndim != 3 or features.shape[1:] != (3, self.dim):
            raise ValueError('Expected [B,3,D] fused Drug A / Drug B / Cell tokens')
        return torch.cat((self.drug_adapter(features[:, :2]),
                          self.cell_adapter(features[:, 2:])), dim=1)

    def forward(self, features):
        return super().forward(self.adapt(features))


def install_entity_token_adapter(model):
    """Preserve freshly initialized core weights and the parent RNG stream.

    Copies refer ONLY to this just-created model, never an old checkpoint.
    Forking RNG prevents the adapter constructor from changing upstream
    dropout/sampling streams. The new parameters have a fixed seed0-derived
    initialization; they remain fully trainable with the rest of the model.
    """
    original = model.joint_transformer
    if isinstance(original, EntityAdaptedTokenInteraction):
        raise ValueError('Entity adapters are already installed')
    reference = next(original.parameters())
    if reference.device.type != 'cpu':
        raise ValueError('Install adapters during fresh CPU construction, before .to(device)')
    with torch.random.fork_rng(devices=[]):
        replacement = EntityAdaptedTokenInteraction(
            dim=original.dim, heads=original.encoder.self_attn.num_heads,
            dropout=original.encoder.dropout.p,
        ).to(dtype=reference.dtype)
    before, after = original.state_dict(), replacement.state_dict()
    if not set(before) <= set(after):
        raise ValueError('Transformer state schema unexpectedly changed')
    with torch.no_grad():
        for key, value in before.items():
            after[key].copy_(value)
    replacement.train(original.training)
    model.joint_transformer = replacement


class EntityAdaptedHypothesisModel(HypothesisEvidenceModel):
    """Current evidence model with only the Transformer input adapter changed.

    Target-edge masking, graph construction, gamma, source fusion and training
    objective are inherited verbatim. An unmasked experiment is a separate
    protocol decision; adding adapters must not silently change that factor.
    """
    def __init__(self, data, dim=256, condition_on_cell=True, collaborative_query=True):
        super().__init__(data, dim, condition_on_cell=condition_on_cell,
                         collaborative_query=collaborative_query)
        install_entity_token_adapter(self)

    def components(self):
        # Separate named groups make it possible to prove that BOTH adapters
        # receive nonzero gradients. Hyperparameters remain the parent's ones.
        groups = super().components()
        drug = list(self.joint_transformer.drug_adapter.parameters())
        cell = list(self.joint_transformer.cell_adapter.parameters())
        ids = {id(p) for p in drug+cell}
        groups['joint_transformer'] = [p for p in groups['joint_transformer'] if id(p) not in ids]
        groups['transformer_drug_adapter'] = drug
        groups['transformer_cell_adapter'] = cell
        return groups

    @staticmethod
    def prediction_recipe():
        return dict(HypothesisEvidenceModel.prediction_recipe(),
            version='entity_token_adapter_20261122',
            transformer_input='fused entity tokens [Drug A, Drug B, Cell]',
            transformer_input_adapter='nonaffine LN -> bias-free Linear256 -> GELU -> nonaffine LN',
            transformer_shared_drug_adapter=True, transformer_independent_cell_adapter=True,
            transformer_input_adapter_parameters=2*256*256,
            extra_adapter_dropout=False, existing_Transformer_and_prediction_head_preserved=True,
            only_readout_input_changed=True, adapter_requires_new_training=True)


@torch.no_grad()
def audit_entity_tokens(readout, features):
    """Actual eval-mode Q/K attention and scale diagnostics, not quality scores."""
    if readout.training:
        raise ValueError('Input audit requires eval mode')
    adapted = readout.adapt(features)
    tokens = adapted + readout.role(readout.role_ids)[None]
    normalized = readout.encoder.norm1(tokens)
    _, weights = readout.encoder.self_attn(normalized, normalized, normalized,
        need_weights=True, average_attn_weights=False)
    output = readout(features)
    all_vectors = (features, adapted, output, weights)
    if not all(bool(torch.isfinite(x).all()) for x in all_vectors):
        raise FloatingPointError('Nonfinite adapted tokens or attention')
    rms = lambda x: x.float().square().mean(-1).sqrt().mean(0).cpu().tolist()
    return dict(entity_order=['Drug A','Drug B','Cell'], rows=len(features),
        input_RMS_by_entity=rms(features), adapted_RMS_by_entity=rms(adapted),
        output_RMS_by_entity=rms(output), role_RMS=rms(readout.role(readout.role_ids)[None]),
        attention_mean_by_head=weights.mean(0).cpu().tolist(),
        attention_entropy_over_log3=float((-(weights*weights.clamp_min(1e-12).log()).sum(-1)/math.log(3)).mean()),
        attention_max_mean=float(weights.max(-1).values.mean()),
        diagnostics_are_not_performance_or_calibrated_confidence=True)

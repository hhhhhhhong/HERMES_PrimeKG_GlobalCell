"""Joint multimodal prediction, reusing the audited K-Paths graph and encoders.

The knowledge stream is an input to a learned concatenation projection. It is
not an additive correction to the molecular/expression stream. In particular,
there is no zero-initialized branch output or 25%/10% residual clipping here.
The standard Transformer still has its own internal attention/FFN skips.
"""
import torch
from torch import nn

from global_cell.attention import ConditionalGamma
from .model import ModalEncoder, SemanticHypergraph, DualBranchModel
from .kpaths_model import KPathsModel, OrderedPathEncoder


class ConcatenationFusion(nn.Module):
    """Normalize two sources independently, then learn a joint projection.

    Drug A/B share the SAME projection; Cell has its own projection because
    its observed modality is expression rather than molecular structure.
    A missing knowledge source is represented by exact zeros. Independent,
    non-affine input normalization keeps these zeros zero; no fake evidence
    vector or modality confidence is inferred from an embedding's magnitude.
    """

    def __init__(self, dim=256, dropout=.2):
        super().__init__()
        self.dim = dim
        self.modal_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.knowledge_norm = nn.LayerNorm(dim, elementwise_affine=False)

        def projection():
            return nn.Sequential(
                nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(dim),
            )

        self.drug = projection()
        self.cell = projection()

    def forward(self, modal, knowledge):
        if modal.ndim != 3 or modal.shape[1:] != (3, self.dim):
            raise ValueError('Expected [batch, 3, dim] Drug/Drug/Cell tokens')
        if knowledge.shape != modal.shape:
            raise ValueError('Knowledge and modal token shapes must agree')
        joined = torch.cat((self.modal_norm(modal), self.knowledge_norm(knowledge)), -1)
        return torch.cat((self.drug(joined[:, :2]), self.cell(joined[:, 2:])), 1)


class JointTokenInteraction(nn.Module):
    """Ordinary 3-token Transformer, without the old capped output correction.

    Equal Drug role embeddings and no positions preserve drug exchange
    equivariance in evaluation. Attention is across the three entities of
    ONE sample; batch entries never attend to other samples.
    """

    def __init__(self, dim=256, heads=4, dropout=.2):
        super().__init__()
        self.dim = dim
        self.role = nn.Embedding(2, dim)
        self.register_buffer('role_ids', torch.tensor([0, 0, 1]), persistent=False)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.role.weight, std=.02)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (3, self.dim):
            raise ValueError('Expected [batch, 3, dim] fused entity tokens')
        tokens = features + self.role(self.role_ids)[None]
        return self.output_norm(self.encoder(tokens))


class MultimodalKPathsModel(nn.Module):
    """Same evidence construction; joint modal/knowledge fusion before readout.

    Fixed CODER and Uni-Mol inputs remain frozen caches. All downstream
    parameters, including molecular/Cell encoders, learn from scratch. This
    model does not load a previous synergy checkpoint or restore the original
    HERMES Refiner. Old run classes and their source files are left intact.
    """

    base_features = DualBranchModel.base_features
    knowledge_table = KPathsModel.knowledge_table

    def __init__(self, data, dim=256):
        super().__init__()
        if dim != 256:
            raise ValueError('Existing Drug and Cell encoders require dim=256')
        from HERMES.model import Consolidator
        self.modal = ModalEncoder(data['raw'], data['chemistry'], data['unimol'])
        self.semantic = SemanticHypergraph(data['graph'], data['semantic'], dim)
        self.gamma = ConditionalGamma(dim)
        self.path_encoder = OrderedPathEncoder(len(data['paths']['relation_names']), dim)
        self.fusion = ConcatenationFusion(dim)
        self.joint_transformer = JointTokenInteraction(dim)
        self.head = Consolidator(dim)
        self.num_drug = data['raw']['num_drug']
        self.num_cell = data['raw']['num_cell']
        for name in ('nodes', 'relations', 'hops', 'pair_paths', 'pair_mask'):
            self.register_buffer('path_' + name, data['paths'][name].clone(), persistent=False)
        families = {'Synergy_Hyperedge', 'cell_expression'}
        keep = []
        for i, key in enumerate(self.semantic.trace['token_keys']):
            if key[0] == 'single' and key[1] in families:
                keep.append(i)
            elif key[0] == 'interaction' and key[2] in families and key[4] in families:
                keep.append(i)
        assert keep, 'Global Cell/Synergy channels must be present'
        self.register_buffer('global_token_ids', torch.tensor(keep, dtype=torch.long), persistent=False)
        self.global_tokens = len(keep)
        self.global_rows = data['graph']['n_roots'] * self.global_tokens
        self.selection_audit = dict(
            policy='same target-anchored K-Paths and global Cell/Synergy as previous run',
            graph=data['graph']['summary'], global_readout_families=sorted(families),
            global_readout_channels=self.global_tokens, paths=len(data['paths']['hops']) - 1,
            includes_Cell_and_fit_synergy=True, evidence_selection_changed=False,
        )

    def decode(self, fused):
        return DualBranchModel.decode(self, self.joint_transformer(fused))

    def forward(self, rows, table, base, capture=False, use_knowledge=True, empty_evidence=False):
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        pair = ids[:, 0] * self.num_drug + ids[:, 1]
        path_ids, path_mask = self.path_pair_paths[pair], self.path_pair_mask[pair]
        attention = available = None
        if use_knowledge:
            global_table = table[:self.global_rows].reshape(-1, self.global_tokens, self.semantic.dim)
            global_available = self.semantic.interaction.mask[:, self.global_token_ids]
            path_values = table[self.global_rows:][path_ids]
            evidence = torch.cat((global_table[ids], path_values[:, None].expand(-1, 3, -1, -1)), 2)
            available = torch.cat((global_available[ids], path_mask[:, None].expand(-1, 3, -1)), 2)
            if empty_evidence:
                available = torch.zeros_like(available)
            context, attention = self.gamma(own, evidence, available, capture)
        else:
            context = torch.zeros_like(own)
        fused = self.fusion(own, context)
        prediction = self.decode(fused)
        detail = dict(
            own=own.detach(), context=context.detach(), fused_features=fused.detach(),
            attention=attention, available=available.detach() if available is not None else None,
            path_ids=path_ids.detach(), path_mask=path_mask.detach(),
            global_tokens=self.global_tokens,
        ) if capture else None
        return prediction, detail

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
            gamma=list(self.gamma.parameters()), path_encoder=list(self.path_encoder.parameters()),
            multimodal_fusion=list(self.fusion.parameters()),
            joint_transformer=list(self.joint_transformer.parameters()),
            prediction_head=list(self.head.parameters()),
        )

    def optimizer_param_groups(self, base_lr=1e-4, weight_decay=1e-4):
        groups = [dict(name=name, params=ps, lr=base_lr,
                       weight_decay=0. if name == 'propagation_gates' else weight_decay)
                  for name, ps in self.components().items()]
        actual = [id(p) for group in groups for p in group['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        assert len(actual) == len(set(actual)) and set(actual) == expected
        return groups

    @staticmethod
    def prediction_recipe():
        return dict(
            version='kpaths_joint_multimodal_20261122',
            fusion='separate source LayerNorm -> concat512 -> Linear256/GELU/Dropout/LayerNorm',
            shared_drug_projection=True, separate_cell_projection=True,
            knowledge_added_to_modal=False, zero_initialized_output=False,
            knowledge_residual_cap=None, transformer_residual_cap=None,
            transformer=dict(tokens=3, dim=256, heads=4, layers=1, feedforward=512,
                             dropout=.2, standard_internal_residuals=True, positional_encoding=False),
            symmetric_original_Consolidator=True, original_HERMES_Refiner=False,
            original_drug_2d_3d_encoder_unchanged=True, Cell_expression_encoder_unchanged=True,
            all_downstream_trainable=True, old_checkpoint_loaded=False,
            graph_and_candidate_selection_unchanged=True,
        )

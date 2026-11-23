"""Independent collaboration and biomedical streams, jointly fused with modalities."""
import torch
from torch import nn
from .multimodal_fusion import MultimodalKPathsModel


class ConcatenatedDrugEncoder(nn.Module):
    """Reuse BRICS and Uni-Mol projections; concatenate their fragment features.

    The original BRICS encoder's internal graph skips are unchanged. Only the
    cross-modality 2D-plus-3D addition is replaced by a learned joint projection.
    Coverage is applied AFTER the 3D projection: missing 3D stays exactly zero.
    """
    def __init__(self, original):
        super().__init__()
        self.encoder = original
        hidden = original.hidden
        self.fragment_fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

    @property
    def pretrained_projection(self):
        return self.encoder.pretrained_projection

    def tokens(self, use_3d=True):
        e = self.encoder
        two_d = e.tokens(use_3d=False)
        if use_3d:
            projected = e.pretrained_projection(e.pretrained_norm(e.fragment_3d))
            projected = projected * e.fragment_3d_coverage[:, None]
            three_d = projected[e.fragment_indices]
        else:
            three_d = torch.zeros_like(two_d)
        return self.fragment_fusion(torch.cat((two_d, three_d), -1))

    def forward(self, use_3d=True):
        e = self.encoder
        tokens = self.tokens(use_3d)
        mask = e.fragment_mask.unsqueeze(-1)
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        counts = torch.stack((e.drug_fragment_counts.log1p(), e.drug_heavy.log1p()), -1)
        return e.drug_projection(torch.cat((mean, counts), -1))


class ThreeSourceFusion(nn.Module):
    """Modality, author-Refiner collaboration and KG are independent inputs.

    Collaboration does not compete with biomedical paths in gamma. No source
    is a zero-output additive correction; each has trainable projection columns.
    """
    def __init__(self, dim=256, dropout=.2):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])

        def projection():
            return nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim))

        self.drug, self.cell = projection(), projection()

    def forward(self, modal, collaborative, knowledge):
        joined = torch.cat([norm(value) for norm, value in zip(self.norms, (modal, collaborative, knowledge))], -1)
        return torch.cat((self.drug(joined[:, :2]), self.cell(joined[:, 2:])), 1)


def kept_synergy_incidence(incidence, edges, training, device):
    """True whole-edge Dropout for the author Refiner; compress remaining IDs."""
    keep = torch.rand(edges, device=device) >= .2 if training else torch.ones(edges, dtype=torch.bool, device=device)
    selected = incidence[:, keep[incidence[1]]].clone()
    selected[1] = (keep.long().cumsum(0) - 1)[selected[1]]
    return selected, torch.ones(int(keep.sum()), device=device)


class MechanismMultimodalModel(MultimodalKPathsModel):
    """Same unified graph, separate readouts of collaboration and real biology.

    The author's actual three-layer Refiner reads the FIT Synergy subrelation
    of that graph, initialized with root CODER projections. Its equations are
    unchanged; CODER inputs, synergy-only graph and edge Dropout are declared
    adaptations. This is not a complete reproduction of HERMES objectives.
    """
    def __init__(self, data, dim=256):
        super().__init__(data, dim)
        from HERMES.model import Refiner
        self.modal.drug = ConcatenatedDrugEncoder(self.modal.drug)
        self.fusion = ThreeSourceFusion(dim)
        self.collaborative_refiner = Refiner(dim, dim)
        groups = data['graph']['groups']['Synergy_Hyperedge']
        incidence = torch.tensor([[node, i] for i, edge in enumerate(groups) for node in edge['members']], dtype=torch.long).t().contiguous()
        self.register_buffer('synergy_incidence', incidence, persistent=False)
        self.synergy_edges = len(groups)
        self.nroots = data['graph']['n_roots']
        self.relation_tokens = len(self.semantic.trace['token_keys'])
        self.global_rows = self.nroots * self.relation_tokens
        self.collaborative_offset = self.global_rows + len(self.path_hops)
        keep = []
        for i, key in enumerate(self.semantic.trace['token_keys']):
            # Keep ALL genuine rooted biomedical combinations, with ordered
            # relation/position descriptors and the compiler's shared-entity
            # witnesses. Synergy has its own direct readout below.
            if key[0] == 'single' and key[1] != 'Synergy_Hyperedge':
                keep.append(i)
            elif key[0] == 'interaction' and key[2] != 'Synergy_Hyperedge' and key[4] != 'Synergy_Hyperedge':
                keep.append(i)
        self.global_token_ids = torch.tensor(keep, dtype=torch.long)
        self.global_tokens = len(keep)
        keys = [self.semantic.trace['token_keys'][i] for i in keep]
        self.selection_audit = dict(self.selection_audit,
            policy='rooted biomedical singles/compositions + exact query paths; independent author Refiner',
            readout_token_keys=keys, all_biomedical_composition_channels_read=True,
            collaborative_readout='author Refiner on fit Synergy incidences, outside gamma',
            PPI_bridge_candidates=True, candidate_selection_changed=True)

    def knowledge_table(self, base=None, capture=False):
        initial = self.semantic.initial_nodes()
        slots, nodes = self.semantic.propagate(initial, capture=capture or self.semantic.capture_propagation)
        relations, _ = self.semantic.interaction(slots, capture)
        paths = self.path_encoder(nodes, self.path_nodes, self.path_relations, self.path_hops)
        incidence, weights = kept_synergy_incidence(self.synergy_incidence, self.synergy_edges,
                                                    self.training, initial.device)
        collaboration = self.collaborative_refiner(initial[:self.nroots], incidence, weights)
        return torch.cat((relations.reshape(-1, self.semantic.dim), paths, collaboration))

    def forward(self, rows, table, base, capture=False, use_knowledge=True,
                empty_evidence=False, use_collaborative=True):
        ids = rows[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        pair = ids[:, 0] * self.num_drug + ids[:, 1]
        path_ids, path_mask = self.path_pair_paths[pair], self.path_pair_mask[pair]
        attention = available = None
        if use_knowledge:
            relations = table[:self.global_rows].reshape(self.nroots, self.relation_tokens, -1)
            root_values = relations[:, self.global_token_ids][ids]
            root_mask = self.semantic.interaction.mask[:, self.global_token_ids][ids]
            path_values = table[self.global_rows:self.collaborative_offset][path_ids]
            evidence = torch.cat((root_values, path_values[:, None].expand(-1, 3, -1, -1)), 2)
            available = torch.cat((root_mask, path_mask[:, None].expand(-1, 3, -1)), 2)
            if empty_evidence:
                available = torch.zeros_like(available)
            context, attention = self.gamma(own, evidence, available, capture)
        else:
            context = torch.zeros_like(own)
        collaborative = table[self.collaborative_offset:][ids] if use_collaborative else torch.zeros_like(own)
        fused = self.fusion(own, collaborative, context)
        prediction = self.decode(fused)
        detail = dict(own=own.detach(), context=context.detach(), collaborative=collaborative.detach(),
            fused_features=fused.detach(), attention=attention,
            available=available.detach() if available is not None else None,
            path_ids=path_ids.detach(), path_mask=path_mask.detach(), global_tokens=self.global_tokens) if capture else None
        return prediction, detail

    def components(self):
        return dict(super().components(), collaborative_refiner=list(self.collaborative_refiner.parameters()))

    @staticmethod
    def prediction_recipe():
        return dict(version='mechanism_multimodal_repair_20261122',
            drug_2d_3d_fusion='BRICS128 || coverage*projected_UniMol128 -> learned128 -> Drug256',
            final_fusion='LN(modal256) || LN(author_Refiner256) || LN(biomedical_gamma256) -> learned256',
            original_HERMES_Refiner_module=True, Refiner_inputs='root CODER projections',
            Refiner_edges='same unified graph: current fit positive Synergy only, whole-edge Dropout.2',
            original_HERMES_full_objectives=False, all_biomedical_compositions_read=True,
            single_drug_context_available_without_pair_path=True, PPI_bridge_max_intermediate_proteins=1,
            all_downstream_trainable=True, old_checkpoint_loaded=False, Cell_expression_unchanged=True,
            knowledge_additive_residual=False, zero_initialized_output=False, external_residual_caps=False,
            Transformer_layers=1, Transformer_heads=4, Transformer_tokens=3,
            primary_selection='internal Val AUPRC from first evaluation; legacy epoch1000 AUROC separate diagnostic',
            test_evaluated=False)

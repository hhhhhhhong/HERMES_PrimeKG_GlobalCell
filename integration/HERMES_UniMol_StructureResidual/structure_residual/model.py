"""Cell-conditioned, exchange-symmetric BRICS/Uni-Mol logit correction.

Only this module is optimised. Original HERMES representations/logits and
Uni-Mol vectors enter as detached buffers; neither pretrained model is part of
the optimiser. BRICS atom/bond encoders adapt the existing two-layer GINE
implementation, retaining real cut bonds and fragment multiplicity.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from .chemistry import ATOM_CARDINALITIES, BOND_CARDINALITIES


class CategoricalChemistry(nn.Module):
    def __init__(self, cardinalities, hidden):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(n, hidden) for n in cardinalities)

    def forward(self, x):
        return sum(t(x[:, i]) for i, t in enumerate(self.tables)) / math.sqrt(len(self.tables))


class StructureResidual(nn.Module):
    def __init__(self, chemistry, unimol, hidden=128):
        super().__init__()
        for name, tensor in dict(chemistry, **unimol).items():
            self.register_buffer(name, tensor.detach().clone())
        self.hidden = hidden
        self.num_fragments = len(self.fragment_drug)
        self.atom_embedding = CategoricalChemistry(ATOM_CARDINALITIES, hidden)
        self.bond_embedding = CategoricalChemistry(BOND_CARDINALITIES, hidden)
        def layer():
            return GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                         nn.Linear(hidden, hidden)), train_eps=True)
        self.atom_layers = nn.ModuleList([layer(), layer()])
        self.atom_norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.fragment_projection = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU())
        self.fragment_layer = layer()
        self.fragment_norm = nn.LayerNorm(hidden)
        self.pretrained_norm = nn.LayerNorm(512, elementwise_affine=False)
        self.pretrained_projection = nn.Linear(512, hidden, bias=False)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.query = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden))
        self.keys = nn.Linear(hidden, hidden, bias=False)
        self.values = nn.Linear(hidden, hidden, bias=False)
        self.partner_query = nn.Linear(hidden, hidden, bias=False)
        self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1, bias=False))
        nn.init.zeros_(self.pair[-1].weight)
        # Lossless padded fragment occurrences. A repeated fragment has its own
        # position; no top-K cropping, learned ID vocabulary or test statistics.
        counts = torch.bincount(self.fragment_drug, minlength=len(self.drug_heavy))
        indices = torch.zeros(len(counts), int(counts.max()), dtype=torch.long)
        mask = torch.zeros_like(indices, dtype=torch.bool)
        for drug in range(len(counts)):
            found = (self.fragment_drug == drug).nonzero().flatten()
            indices[drug, :len(found)] = found
            mask[drug, :len(found)] = True
        self.register_buffer('fragment_indices', indices)
        self.register_buffer('fragment_mask', mask)

    def tokens(self, use_3d=True):
        atoms = self.atom_embedding(self.atoms)
        bonds = self.bond_embedding(self.atom_bonds)
        for layer, norm in zip(self.atom_layers, self.atom_norms):
            atoms = F.relu(norm(layer(atoms, self.atom_edges, bonds)))
        fragments = global_mean_pool(atoms, self.atom_fragment, size=self.num_fragments)
        fragments = self.fragment_projection(torch.cat((fragments, self.fragment_heavy.log1p()[:, None]), -1))
        fragments = fragments + F.relu(self.fragment_norm(self.fragment_layer(
            fragments, self.fragment_edges, self.bond_embedding(self.fragment_bonds))))
        if use_3d:
            # Missing representations are exactly zero, including after training;
            # fraction explicitly downweights a fragment spanning missing atoms.
            fragments = fragments + self.pretrained_projection(self.pretrained_norm(self.fragment_3d)) * self.fragment_3d_coverage[:, None]
        fragments = self.fusion_norm(fragments)
        return fragments[self.fragment_indices]

    def attend(self, query, tokens, mask):
        score = (query[:, None] * self.keys(tokens)).sum(-1) / math.sqrt(self.hidden)
        weights = score.masked_fill(~mask, -torch.inf).softmax(-1)
        return (weights[:, :, None] * self.values(tokens)).sum(1), weights

    def delta(self, table, rows, tokens=None, use_3d=True, return_attention=False):
        if tokens is None:
            tokens = self.tokens(use_3d)
        a, b, c = rows[:, :3].long().unbind(1)
        cell = self.query(table[c].detach())
        ta, tb = tokens[a], tokens[b]
        ma, mb = self.fragment_mask[a], self.fragment_mask[b]
        pa, first_a = self.attend(cell, ta, ma)
        pb, first_b = self.attend(cell, tb, mb)
        # Shared projections in both directions preserve exact A/B symmetry.
        ia, wa = self.attend(cell + self.partner_query(pb), ta, ma)
        ib, wb = self.attend(cell + self.partner_query(pa), tb, mb)
        features = torch.cat((ia + ib, (ia - ib).abs(), ia * ib, cell), -1)
        delta = self.pair(features).squeeze(-1)
        if return_attention:
            return delta, dict(cell_a=first_a, cell_b=first_b, pair_a=wa, pair_b=wb)
        return delta

    def forward(self, table, rows, baseline_logit, tokens=None, use_3d=True):
        delta = self.delta(table, rows, tokens=tokens, use_3d=use_3d)
        return torch.sigmoid(baseline_logit.detach() + delta), delta

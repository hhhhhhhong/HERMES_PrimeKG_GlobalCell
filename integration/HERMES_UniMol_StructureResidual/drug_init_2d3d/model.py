"""Replace only the Drug initializer; keep the author's HGNN and readout.

The frozen Uni-Mol *atom* representations were mapped to BRICS occurrences
offline. Here their projection, the 2D encoder and the whole HERMES are learned
jointly. No cached post-HGNN representation or delta-logit head is used.
"""
import torch
from torch import nn
from structure_residual.model import StructureResidual


class Drug2D3DEncoder(StructureResidual):
    def __init__(self, chemistry, unimol, hidden=128):
        # Reuse exactly the previously audited BRICS atom/cut-bond encoder.
        super().__init__(chemistry, unimol, hidden)
        # These belong to the old post-HGNN correction and must not be optimized
        # as unused parameters in the new initializer.
        for name in ('query', 'keys', 'values', 'partner_query', 'pair'):
            delattr(self, name)
        self.drug_projection = nn.Sequential(nn.Linear(hidden + 2, 256), nn.LayerNorm(256))

    def forward(self, use_3d=True):
        tokens = self.tokens(use_3d=use_3d)
        mask = self.fragment_mask.unsqueeze(-1)
        # Every fragment occurrence participates, including repeated fragments.
        # Explicit counts retain size/multiplicity information lost by a mean.
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        counts = torch.stack((self.drug_fragment_counts.log1p(), self.drug_heavy.log1p()), -1)
        return self.drug_projection(torch.cat((mean, counts), -1))


class Initializer2D3D(nn.Module):
    """Reuse the original Cell/Disease modules without reinitializing them.

    Install AFTER the author's HERMERS constructor, so the new Drug encoder
    cannot alter the seeded initial states of the retained author modules.
    The Cell/Disease operations below are the author's Initializer.forward.
    Original GTN modules are removed rather than kept as unused parameters.
    """
    def __init__(self, original, chemistry, unimol):
        super().__init__()
        self.out_dim = original.out_dim
        self.use_disease = original.use_disease
        self.cline_first, self.cline_same = original.cline_first, original.cline_same
        if self.use_disease:
            self.disease_first, self.disease_same = original.disease_first, original.disease_same
        self.drug_encoder = Drug2D3DEncoder(chemistry, unimol)

    def forward(self, drug_x, drug_adj, ibatch, cline_x, *args):
        # ibatch still identifies the original ordered whole-drug graph batch.
        # Actual Drug features now come from the aligned chemistry buffers.
        drug = self.drug_encoder()
        if int(ibatch.max()) + 1 != len(drug):
            raise ValueError('Expected the complete, original ordered Drug batch')
        cell = self.cline_first(cline_x)
        for layer in self.cline_same:
            cell = cell + layer(cell)
        if self.use_disease:
            disease = self.disease_first(args[0])
            for layer in self.disease_same:
                disease = disease + layer(disease)
            return drug, cell, disease
        return drug, cell


def install(model, chemistry, unimol):
    """A fresh author model, never a trained checkpoint, is the input."""
    device = next(model.parameters()).device
    model.initializer = Initializer2D3D(model.initializer, chemistry, unimol).to(device)
    assert all(p.requires_grad for p in model.parameters())
    assert not hasattr(model.initializer, 'drug_first')
    return model

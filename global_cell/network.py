"""Unified Drug / Cell / PrimeKG graph with observed expression incidences."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .attention import AlphaBeta, RelationInteraction, ConditionalGamma
from .structure import compile_graph


class GatedResidualReadout(nn.Module):
    """
    角色感知的门控残差读取模块：
    final = own + tanh(gate) * Linear(LayerNorm(context))
    - gate 初始化为 0.05，proj 使用标准 Xavier 初始化：
      既保证初始时刻 own 占 >95% 主导权直通预测头，又保证第 1 轮反向传播时所有图谱组件立刻获得健康梯度。
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.drug_norm = nn.LayerNorm(dim)
        self.cell_norm = nn.LayerNorm(dim)

        self.drug_proj = nn.Linear(dim, dim)
        self.cell_proj = nn.Linear(dim, dim)

        # 初始化为 0.05 而非绝对 0，避免 Full-Batch 第 0 步阻断上游图谱梯度
        self.drug_gate = nn.Parameter(torch.full((1, 1, dim), 0.05))
        self.cell_gate = nn.Parameter(torch.full((1, 1, dim), 0.05))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.drug_proj.weight)
        nn.init.zeros_(self.drug_proj.bias)
        nn.init.xavier_uniform_(self.cell_proj.weight)
        nn.init.zeros_(self.cell_proj.bias)

    def forward(self, own: torch.Tensor, context: torch.Tensor, has_evidence: torch.Tensor = None) -> torch.Tensor:
        own_drug, own_cell = own[:, :2, :], own[:, 2:, :]
        ctx_drug, ctx_cell = context[:, :2, :], context[:, 2:, :]

        delta_drug = self.drug_proj(self.drug_norm(ctx_drug))
        delta_cell = self.cell_proj(self.cell_norm(ctx_cell))

        if has_evidence is not None:
            mask_drug = has_evidence[:, :2, :].to(own.dtype)
            mask_cell = has_evidence[:, 2:, :].to(own.dtype)
            delta_drug = delta_drug * mask_drug
            delta_cell = delta_cell * mask_cell

        final_drug = own_drug + torch.tanh(self.drug_gate) * delta_drug
        final_cell = own_cell + torch.tanh(self.cell_gate) * delta_cell

        return torch.cat((final_drug, final_cell), dim=1)


class GlobalCellModel(nn.Module):
    def __init__(self, cache, graph, coder):
        super().__init__()
        from HERMES.model import Initializer, Consolidator
        self.initializer = Initializer(75, cache['cell_input_dim'], 256, use_disease=False)
        self.groups, packed, self.trace = compile_graph(graph)
        self.encoder = AlphaBeta(graph, self.groups, dim=256, heads=4)
        self.interaction = RelationInteraction(packed, graph['n_roots'], len(self.trace['token_keys']), dim=256)
        self.gamma = ConditionalGamma(dim=256)
        self.readout = GatedResidualReadout(dim=256)
        self.head = Consolidator(256)
        self.external_projection = nn.ModuleDict()
        self.register_buffer('coder_features', coder.clone(), persistent=False)
        for t, name in enumerate(graph['type_names']):
            if name.startswith('hermes_'):
                continue
            ids = torch.where(graph['node_types'][graph['n_base']:] == t)[0]
            self.register_buffer('external_ids_' + str(t), ids, persistent=False)
            self.external_projection[str(t)] = nn.Sequential(nn.Linear(768, 256), nn.Tanh())
        for name, value in cache['inputs'].items():
            self.register_buffer('input_' + name, value.clone(), persistent=False)
        self.num_drug = cache['num_drug']
        self.num_cell = cache['num_cell']
        self.external_count = len(coder)

    def base_features(self):
        drug, cell = self.initializer(self.input_drug_x, self.input_drug_adj, self.input_ibatch, self.input_cells)
        return torch.cat((drug, cell))

    def knowledge_table(self, base, capture=False):
        external = base.new_zeros((self.external_count, 256))
        for t, projection in self.external_projection.items():
            ids = getattr(self, 'external_ids_' + t)
            external = external.index_copy(0, ids, projection(self.coder_features[ids]))
        # This is the ONLY hypergraph encoder. Both real DTI/PPI and multi-member
        # GO/pathway hyperedges coexist in the same typed graph.
        slots, _ = self.encoder(torch.cat((base, external)), self.groups, capture)
        table, _ = self.interaction(slots, capture)
        return table

    def forward(self, rows, table, base, capture=False):
        idx = rows[:, :3].long().clone()
        idx[:, :2] = idx[:, :2].sort(dim=1).values
        own = base[idx]
        context, attention = self.gamma(own, table, self.interaction.mask, capture, indices=idx)
        has_evidence = self.interaction.mask[idx].any(dim=-1, keepdim=True)
        final = self.readout(own, context, has_evidence=has_evidence)

        # 双向对称调用 Consolidator.forward_once，消除按节点整数 ID 排序带来的槽位偏置
        d_a, d_b, cell = final.unbind(1)
        pred = 0.5 * (
            self.head.forward_once(d_a, d_b, cell) +
            self.head.forward_once(d_b, d_a, cell)
        ).reshape(-1)
        return pred, attention

    def components(self):
        return dict(
            drug_encoder=list(self.initializer.drug_first.parameters()) + list(self.initializer.drug_conv_same.parameters()),
            cell_encoder=list(self.initializer.cline_first.parameters()) + list(self.initializer.cline_same.parameters()),
            coder_projection=list(self.external_projection.parameters()),
            alpha=list(self.encoder.alpha.parameters()),
            beta=[self.encoder.beta],
            node_to_edge=list(self.encoder.n2e.parameters()),
            edge_to_node=list(self.encoder.e2n.parameters()),
            expression_edge=list(self.encoder.expression_projection.parameters()),
            relation_interaction=list(self.interaction.parameters()),
            gamma=list(self.gamma.parameters()),
            readout=list(self.readout.parameters()),
            prediction_head=list(self.head.parameters())
        )


def gradients(model):
    return {k: sum(float(p.grad.detach().norm()) for p in ps if p.grad is not None) for k, ps in model.components().items()}


def train_epoch(model, optimizer, rows, labels, chunk=1024):
    """Exact full-fit BCE, one step/epoch; chunks only bound query memory."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    base = model.base_features()
    table = model.knowledge_table(base)
    leaf_base = base.detach().requires_grad_(True)
    leaf_table = table.detach().requires_grad_(True)
    total = 0.
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        p, _ = model(rows[start:stop], leaf_table, leaf_base)
        loss = torch.nn.functional.binary_cross_entropy(p, labels[start:stop], reduction='sum') / len(rows)
        loss.backward()
        total += float(loss.detach())
    # The table depends on the same live Drug/Cell features as the query. Both
    # accumulated gradients must traverse the shared upstream graph together.
    torch.autograd.backward((table, base), (leaf_table.grad, leaf_base.grad))
    grad = gradients(model)
    optimizer.step()
    return total, grad


def evaluate(model, rows, chunk=1024):
    model.eval()
    pred = []
    with torch.no_grad():
        base = model.base_features()
        table = model.knowledge_table(base)
        for start in range(0, len(rows), chunk):
            pred.append(model(rows[start:start + chunk], table, base)[0].cpu())
    return torch.cat(pred)
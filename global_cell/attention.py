"""HHNN-style independent alpha/beta channels, witnessed HYPER interaction,
then query-conditioned gamma. Task adaptation; not a full author-model copy.
"""
import math
import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import softmax
from .common import ROOT, read
from .structure import compile_graph
from .hyper_helpers import sinusoidal_positional_encoding


class AlphaBeta(nn.Module):
    def __init__(self, graph, groups, dim=64, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.width = dim // heads
        ntypes = len(graph['type_names'])
        self.n2e = nn.ModuleDict()
        self.e2n = nn.ModuleDict()
        self.alpha = nn.ParameterDict()
        for name, g in groups.items():
            self.n2e[name] = nn.ModuleDict({str(t): nn.Linear(dim, dim, bias=False) for t in g.present_types})
            self.e2n[name] = nn.ModuleDict({str(t): nn.Linear(dim, dim, bias=False) for t in g.present_types})
            self.alpha[name] = nn.Parameter(torch.empty(heads, self.width))
            nn.init.xavier_uniform_(self.alpha[name])
        self.beta = nn.Parameter(torch.empty(ntypes, 3, heads, self.width))
        nn.init.xavier_uniform_(self.beta)
        self.role = nn.Embedding(3, dim)

        self.expression_projection = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )
        nn.init.normal_(self.expression_projection[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.expression_projection[-1].bias)

        self.register_buffer('types', graph['node_types'])
        self.nbase = graph['n_base']

    def forward(self, x, groups, capture=False):
        slots = []
        audit = {}
        edges_out = {}
        for name, g in groups.items():
            projected = x.new_zeros((len(g.active), self.dim))
            for t in g.present_types:
                pos = getattr(g, 'node_positions_%d' % t)
                projected = projected.index_copy(0, pos, self.n2e[name][str(t)](x[g.active[pos]]))
            member = (projected[g.inverse] + self.role(g.position)).view(-1, self.heads, self.width)
            a_score = torch.nn.functional.leaky_relu((member * self.alpha[name]).sum(-1), .2)
            alpha = softmax(a_score, g.edge, num_nodes=g.num_edges)
            edge = scatter(member * alpha[:, :, None], g.edge, dim=0, dim_size=g.num_edges, reduce='sum').reshape(-1, self.dim)

            if name == 'cell_expression':
                expr = g.expression
                expr_z = (expr - expr.mean(dim=0, keepdim=True)) / expr.std(dim=0, keepdim=True).clamp_min(1e-6)
                scale, shift = self.expression_projection(torch.cat([expr, expr_z], dim=-1)).chunk(2, dim=-1)
                edge = edge * (1.0 + torch.tanh(scale)) + shift

            message = x.new_zeros((len(g.vertex), self.dim))
            for t in g.present_types:
                inc = getattr(g, 'inc_positions_%d' % t)
                projected_edges = self.e2n[name][str(t)](edge)
                message = message.index_copy(0, inc, projected_edges[g.edge[inc]])
            message = (message + self.role(g.position)).view(-1, self.heads, self.width)
            b_score = torch.nn.functional.leaky_relu((message * self.beta[self.types[g.vertex], g.position]).sum(-1), .2)
            beta = softmax(b_score, g.slot_inverse, num_nodes=len(g.slot_nodes))

            edge_prior = g.prior[g.edge, None, None]
            if name == 'Synergy_Hyperedge' and self.training:
                # 训练期以 20% 概率随机丢弃协同超边，防止模型直接记忆当前正训练样本自环
                keep = (torch.rand((g.num_edges, 1, 1), device=x.device) > 0.2).to(x.dtype) / 0.8
                edge_prior = edge_prior * keep[g.edge]

            aggregate = scatter(message * beta[:, :, None] * edge_prior, g.slot_inverse,
                                dim=0, dim_size=len(g.slot_nodes), reduce='sum').reshape(-1, self.dim)
            slots.append(aggregate)
            if capture:
                audit[name] = dict(
                    alpha=alpha.detach(), beta=beta.detach(),
                    alpha_sum_error=float((scatter(alpha, g.edge, dim=0, dim_size=g.num_edges, reduce='sum') - 1).abs().max()),
                    beta_sum_error=float((scatter(beta, g.slot_inverse, dim=0, dim_size=len(g.slot_nodes), reduce='sum') - 1).abs().max())
                )
                edges_out[name] = edge.detach()
        return torch.cat(slots), dict(normalization=audit, edge_embeddings=edges_out) if capture else None


class RelationInteraction(nn.Module):
    def __init__(self, packed, num_roots, num_tokens, dim=64):
        super().__init__()
        self.nroots = num_roots
        self.ntokens = num_tokens
        self.dim = dim
        for name, value in packed.items():
            self.register_buffer(name, value)
        self.register_buffer('sin_position', sinusoidal_positional_encoding(4, dim // 2))
        self.position_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.source_projection = nn.Linear(dim, dim, bias=False)
        self.target_projection = nn.Linear(dim, dim, bias=False)
        self.pool_key = nn.Linear(dim, dim, bias=False)
        self.pool_query = nn.Parameter(torch.empty(num_tokens, dim))
        nn.init.xavier_uniform_(self.pool_query)

    def forward(self, slots, capture=False):
        position = self.position_mlp(torch.cat((self.sin_position[self.pair_position[:, 0] + 1],
                                                self.sin_position[self.pair_position[:, 1] + 1]), dim=-1))
        compositions = torch.nn.functional.gelu(self.source_projection(slots[self.pair_source]) * position +
                                                self.target_projection(slots[self.pair_target]))
        values = torch.cat((slots, compositions))[self.source]
        score = (torch.tanh(self.pool_key(values)) * self.pool_query[self.token]).sum(-1) / math.sqrt(self.dim)
        weights = softmax(score, self.group, num_nodes=self.nroots * self.ntokens)
        table = scatter(values * weights[:, None], self.group, dim=0, dim_size=self.nroots * self.ntokens, reduce='sum')
        table = table.reshape(self.nroots, self.ntokens, self.dim)
        return table, dict(routing_attention=weights.detach(), compositions=compositions.detach()) if capture else None


class ConditionalGamma(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.query = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.self_query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output_role = nn.Embedding(3, dim)
        self.evidence_role = nn.Embedding(3, dim)

        bias_init = torch.full((3, 3), -2.5)
        bias_init.fill_diagonal_(2.5)
        self.role_pair_bias = nn.Parameter(bias_init)

    def forward(self, base_features, table, mask, capture=False, indices=None):
        b = len(base_features)
        sym_roles = torch.tensor([0, 0, 2], device=base_features.device)
        role_e = self.evidence_role(sym_roles)
        role_o = self.output_role(sym_roles)

        if indices is None:
            tokens = table.shape[2]
            evidence = (table + role_e.view(1, 3, 1, self.dim)).reshape(b, 3 * tokens, self.dim)
            keys = self.key(evidence)
            values = self.value(evidence)
            available = mask.reshape(b, 3 * tokens)
        else:
            tokens = table.shape[1]
            keys = (self.key(table)[indices] + self.key(role_e)[None, :, None, :]).reshape(b, 3 * tokens, self.dim)
            values = (self.value(table)[indices] + self.value(role_e)[None, :, None, :]).reshape(b, 3 * tokens, self.dim)
            available = mask[indices].reshape(b, 3 * tokens)

        drug_a, drug_b, cell = base_features[:, 0], base_features[:, 1], base_features[:, 2]
        sym_triad = torch.cat([drug_a + drug_b, (drug_a - drug_b).abs(), cell], dim=-1)
        global_q = self.query(sym_triad)[:, None, :]

        query = global_q + self.self_query(base_features) + role_o[None, :, :]
        score = torch.einsum('bod,btd->bot', query, keys) / math.sqrt(self.dim)

        perm = torch.tensor([1, 0, 2], device=base_features.device)
        sym_bias = 0.5 * (self.role_pair_bias + self.role_pair_bias[perm][:, perm])
        routing_bias = sym_bias[:, :, None].expand(3, 3, tokens).reshape(1, 3, 3 * tokens)
        score = score + routing_bias

        score = score.masked_fill(~available[:, None, :], -1e9)
        weights = score.softmax(-1) * available[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        knowledge = torch.einsum('bot,btd->bod', weights, values)
        return knowledge, weights.detach() if capture else None
"""Label-free canonical query matching and exact held-out Synergy topology.

All duplicate edge instances matching a Drug–Drug–Cell query are removed.
Zeroing edge weights AFTER aggregation is deliberately not used: it would
leave target-dependent softmax denominators, degree counts and graph routes.
"""
import torch
from global_cell.structure import compile_graph


def query_keys(rows, num_drug, num_cell):
    """Use ONLY columns 0:3; the current batch label is never inspected."""
    values = rows[:, :3].detach().cpu()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected Drug/Drug/Cell columns')
    if not bool(torch.isfinite(values).all()) or not torch.equal(values, values.long().to(values.dtype)):
        raise ValueError('Entity IDs must be finite integers')
    ids = values.long().clone()
    ids[:, :2] = ids[:, :2].sort(dim=1).values
    if not bool(((ids[:, :2] >= 0) & (ids[:, :2] < num_drug)).all()):
        raise ValueError('Drug ID outside this dataset')
    cell = ids[:, 2] - num_drug
    if not bool(((cell >= 0) & (cell < num_cell)).all()):
        raise ValueError('Cell must use the original global root ID')
    return (ids[:, 0] * num_drug + ids[:, 1]) * num_cell + cell


class QueryEdgeIndex:
    def __init__(self, graph):
        self.num_drug, self.num_cell = graph['n_drug'], graph['n_cell']
        self.edges = graph['groups'].get('Synergy_Hyperedge', [])
        incidence = torch.tensor([e['members'] for e in self.edges], dtype=torch.long).reshape(-1, 3)
        self.edge_keys = query_keys(incidence, self.num_drug, self.num_cell)

    def exclude(self, rows):
        keys = query_keys(rows, self.num_drug, self.num_cell).unique(sorted=True)
        blocked = torch.isin(self.edge_keys, keys)
        retained = [edge for edge, excluded in zip(self.edges, blocked.tolist()) if not excluded]
        return retained, keys, blocked


def compile_excluded(graph, retained, full_token_keys):
    """Recompile TRUE remaining incidences; remap onto a fixed parameter schema.

    RelationInteraction.pool_query keeps its existing rows. Dynamic routes and
    masks are remapped by relation/role descriptors, so a removed edge cannot
    survive via stale role slots, routes or normalization denominators.
    """
    filtered = dict(graph, groups=dict(graph['groups']))
    if retained:
        filtered['groups']['Synergy_Hyperedge'] = retained
    else:
        # The original Group constructor assumes nonempty relation families.
        # No fake placeholder edge or zero-weight incidence is introduced.
        filtered['groups'].pop('Synergy_Hyperedge', None)
    groups, packed, trace = compile_graph(filtered)
    full = {tuple(key): i for i, key in enumerate(full_token_keys)}
    mapping = torch.tensor([full[tuple(key)] for key in trace['token_keys']], dtype=torch.long)
    dynamic_count, fixed_count = len(trace['token_keys']), len(full_token_keys)
    roots = packed['group'] // dynamic_count
    packed['token'] = mapping[packed['token']]
    packed['group'] = roots * fixed_count + packed['token']
    mask = torch.zeros(graph['n_roots'] * fixed_count, dtype=torch.bool)
    mask[packed['group']] = True
    packed['mask'] = mask.reshape(graph['n_roots'], fixed_count)
    return groups, packed, trace

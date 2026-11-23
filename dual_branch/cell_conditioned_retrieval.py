"""Label-free Cell-conditioned path selection, before neural gamma attention.

Expression is a relative transcriptomic prior, not protein abundance, pathway
activation, a DTI confidence score, or proof of synergy. Missing expression is
unknown (zero support), never evidence that a protein/interaction is absent.
"""
from collections import Counter, defaultdict
import itertools
import math

from .mechanism_retrieval import MechanismIndex

FAMILIES = ('shared_protein', 'ppi', 'ppi_bridge', 'pathway', 'go')


def family(path):
    return 'ppi_bridge' if path.get('template') == 'target_ppi_bridge_target' else path['family']


def diverse_select(paths, score, k=10, variants=2):
    """Deterministic round-robin families; limit ordered relation duplicates."""
    if k < 1 or variants < 1:
        raise ValueError('Positive selection budgets required')
    pools = defaultdict(list)
    for path in paths:
        pools[family(path)].append(path)
    for values in pools.values():
        values.sort(key=lambda p: (-score(p), p['heuristic_cost'], tuple(p['entities']), tuple(p['fact_ids'])))
    if set(pools) - set(FAMILIES):
        raise ValueError('Unsupported mechanism family')
    result, seen, counts = [], set(), Counter()
    while len(result) < k and any(pools.values()):
        for name in FAMILIES:
            while pools[name]:
                path = pools[name].pop(0)
                identity = (tuple(path['entities']), tuple(path['fact_ids']))
                signature = tuple(path['relations'])
                if identity in seen or counts[signature] >= variants:
                    continue
                seen.add(identity)
                counts[signature] += 1
                result.append(path)
                break
            if len(result) == k:
                break
    return result


class CellExpressionSupport:
    """Per-gene midranks across the fixed, unlabeled transductive Cell panel.

    Each measured gene is ranked across Cells separately: 2*(midrank/(N-1))−1.
    Ties use their average rank; constant genes give zero support. Raw signs
    and cross-gene magnitudes are not interpreted as activity. The model's
    expression matrix and existing expression-edge values are not modified.
    """
    def __init__(self, expression, node_types):
        self.types = node_types
        self.ranks = {}
        lengths = {len(values) for values in expression.values()}
        if len(lengths) > 1:
            raise ValueError('All measured proteins must use the same Cell order')
        self.num_cells = next(iter(lengths), 0)
        for protein, values in expression.items():
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite expression')
            order = sorted(range(len(values)), key=lambda i: (values[i], i))
            ranks, start = [0.] * len(values), 0
            while start < len(order):
                stop = start + 1
                while stop < len(order) and values[order[stop]] == values[order[start]]:
                    stop += 1
                value = 2. * ((start + stop - 1) / 2.) / (len(values) - 1) - 1. if len(values) > 1 else 0.
                for index in order[start:stop]:
                    ranks[index] = value
                start = stop
            self.ranks[protein] = ranks

    @classmethod
    def from_graph(cls, graph, node_types):
        expression = {}
        local_ids = {p: graph['n_base'] + i for i, p in enumerate(graph['external_prime_indices'])}
        for edge in graph['groups']['cell_expression']:
            proof = edge['proof']
            cell, protein = int(proof['cell_index']), int(proof['prime_gene_protein'])
            values = expression.setdefault(protein, [None] * graph['n_cell'])
            if values[cell] is not None:
                raise ValueError('Duplicate Cell/protein expression witness')
            local = local_ids[protein]
            if edge['members'] != [graph['n_drug'] + cell, local]:
                raise ValueError('Expression endpoint/order mismatch')
            if edge['expression_value'] != proof['expression_value']:
                raise ValueError('Expression source value mismatch')
            values[cell] = float(proof['expression_value'])
        if any(v is None for values in expression.values() for v in values):
            raise ValueError('Incomplete expression panel: do not impute silently')
        return cls(expression, node_types)

    def support(self, path, cell):
        if not 0 <= cell < self.num_cells:
            raise ValueError('Unknown Cell index')
        proteins = [p for p in path['entities'][1:-1] if self.types[p] == 'gene/protein']
        measured = [p for p in proteins if p in self.ranks]
        value = sum(self.ranks[p][cell] for p in measured) / max(len(proteins), 1)
        return dict(relative_expression_support=value, measured_proteins=measured,
                    protein_count=len(proteins), mapping_coverage=len(measured) / max(len(proteins), 1))

    def select(self, paths, cell, k=10, use_cell=True):
        # Fixed coefficient 1, chosen before outcomes. Family balancing keeps
        # a shorter common-target path from consuming all longer path slots.
        def score(path):
            support = self.support(path, cell)['relative_expression_support'] if use_cell else 0.
            return -path['heuristic_cost'] + support
        return diverse_select(paths, score, k=k, variants=2)


class CellMechanismIndex(MechanismIndex):
    """Expand the candidate POOL before Cell-specific Top-K, with real facts.

    Bounded enumeration of shared targets, target PPI, a shared GO/Reactome
    annotation and a one-protein PPI bridge. Only already encoded nodes are
    admitted; missing facts/nodes are not synthesized. Budgets are explicit.
    """
    def candidates(self, a, b, previous, allowed_nodes, pool_size=64,
                   enumeration_budget=4096, bridge_budget=128):
        if min(pool_size, enumeration_budget, bridge_budget) < 1:
            raise ValueError('Positive candidate budgets required')
        if a is None or b is None or a == b:
            return dict(paths=[], truncated={}, enumerated={}, excluded_unencoded=0,
                        eligible_unique_paths=0, pool_budget_truncated=False, labels_used=False)
        a, b = sorted((a, b))
        ta, tb = self.targets.get(a, {}), self.targets.get(b, {})
        pools, seen = defaultdict(list), set()
        excluded = 0

        def keep(path):
            nonlocal excluded
            identity = (tuple(path['entities']), tuple(path['fact_ids']))
            if identity in seen:
                return
            seen.add(identity)
            if any(v not in allowed_nodes for v in path['entities'][1:-1]):
                excluded += 1
                return
            pools[family(path)].append(path)

        for path in previous:
            keep(path)

        def common():
            for target in sorted(set(ta) & set(tb)):
                yield self.path([a, target, b], [ta[target], tb[target]])

        def direct_ppi():
            for x in sorted(ta):
                for y in sorted(set(self.ppi[x]) & set(tb)):
                    if x != y:
                        yield self.path([a, x, y, b], [ta[x], self.ppi[x][y], tb[y]])

        def annotations():
            left = set().union(*(self.function_neighbors[p] for p in ta)) if ta else set()
            right = set().union(*(self.function_neighbors[p] for p in tb)) if tb else set()
            for anchor in sorted(left & right, key=lambda v: (len(self.function_members[v]), v)):
                members = self.function_members[anchor]
                if len(members) > 200:
                    continue
                for x, y in itertools.product(sorted(set(ta) & members), sorted(set(tb) & members)):
                    if x != y:
                        yield self.path([a, x, anchor, y, b],
                            [ta[x], self.function_fact[anchor, x], self.function_fact[anchor, y], tb[y]])

        left, right = self.bridges.get(a, {}), self.bridges.get(b, {})
        bridges = sorted(set(left) & set(right), key=lambda p: (len(self.ppi[p]), p))

        def bridge_paths():
            for middle in bridges[:bridge_budget]:
                for x, y in itertools.product(left[middle], right[middle]):
                    if len({a, x, middle, y, b}) == 5:
                        path = self.path([a, x, middle, y, b],
                            [ta[x], self.ppi[x][middle], self.ppi[middle][y], tb[y]])
                        path['template'] = 'target_ppi_bridge_target'
                        yield path

        truncated = dict(bridge_entities=len(bridges) > bridge_budget)
        counts = {}
        for name, iterator in [('shared_protein', common()), ('ppi', direct_ppi()),
                               ('function', annotations()), ('ppi_bridge', bridge_paths())]:
            count = 0
            truncated[name] = False
            for path in iterator:
                if count == enumeration_budget:
                    truncated[name] = True
                    break
                count += 1
                keep(path)
            counts[name] = count
        paths = [path for values in pools.values() for path in values]
        selected = diverse_select(paths, lambda p: -p['heuristic_cost'], k=pool_size, variants=8)
        return dict(paths=selected, truncated=truncated, enumerated=counts,
                    excluded_unencoded=excluded, eligible_unique_paths=len(paths),
                    pool_budget_truncated=len(selected) < len(paths), labels_used=False)

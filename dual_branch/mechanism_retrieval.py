"""Extend witnessed K-Paths with one intermediate PPI protein, without labels."""
from collections import Counter, defaultdict
from .kpaths_retrieval import KPathsIndex


class MechanismIndex(KPathsIndex):
    def __init__(self, facts, node_types):
        super().__init__(facts, node_types)
        self.ppi = defaultdict(dict)
        for fid, fact in enumerate(facts):
            if fact['family'] == 'protein_protein':
                a, b = fact['head'], fact['tail']
                self.ppi[a].setdefault(b, fid)
                self.ppi[b].setdefault(a, fid)
        self.bridges = {}
        for drug, targets in self.targets.items():
            by_bridge = defaultdict(list)
            for target in sorted(targets):
                for bridge in self.ppi[target]:
                    by_bridge[bridge].append(target)
            self.bridges[drug] = by_bridge

    def extend(self, a, b, previous, k=10, bridge_budget=128):
        """Retain relation diversity, adding D-T-PPI-M-PPI-T-D candidates.

        The bridge need not be a drug target or a previously retained node.
        The explicit budget is a degree-ranked deterministic truncation, not
        an exhaustive top-K claim. One-sided evidence stays separate.
        """
        if k <= 0 or bridge_budget <= 0:
            raise ValueError('Positive retrieval budgets required')
        paths = list(previous)
        candidates = []
        bridges = []
        if a is not None and b is not None and a != b:
            a, b = sorted((a, b))
            left, right = self.bridges.get(a, {}), self.bridges.get(b, {})
            bridges = sorted(set(left) & set(right), key=lambda p: (len(self.ppi[p]), p))
            for middle in bridges[:bridge_budget]:
                for x in left[middle]:
                    for y in right[middle]:
                        if len({a, x, middle, y, b}) != 5:
                            continue
                        path = self.path([a, x, middle, y, b], [
                            self.targets[a][x], self.ppi[x][middle],
                            self.ppi[middle][y], self.targets[b][y]])
                        path['template'] = 'target_ppi_bridge_target'
                        candidates.append(path)
        paths.extend(candidates)
        pools = defaultdict(list)
        for path in paths:
            category = ('ppi_bridge' if path.get('template') == 'target_ppi_bridge_target'
                        else path['family'])
            pools[category].append(path)
        for pool in pools.values():
            pool.sort(key=lambda p: (p['heuristic_cost'], p['entities'], p['fact_ids']))
        found, seen, signatures = [], set(), Counter()
        families = ['shared_protein', 'ppi', 'ppi_bridge', 'pathway', 'go']
        while len(found) < k and any(pools.values()):
            for family in families:
                pool = pools[family]
                while pool:
                    path = pool.pop(0)
                    identity = (tuple(path['entities']), tuple(path['fact_ids']))
                    signature = tuple(path['relations'])
                    if identity in seen or signatures[signature] >= 2:
                        continue
                    seen.add(identity)
                    signatures[signature] += 1
                    found.append(path)
                    break
                if len(found) == k:
                    break
        return dict(paths=found, bridge_candidates=len(bridges),
                    bridge_budget=bridge_budget, bridges_examined=min(len(bridges), bridge_budget),
                    bridge_budget_truncated=len(bridges) > bridge_budget,
                    generated_bridge_paths=len(candidates), labels_used=False)

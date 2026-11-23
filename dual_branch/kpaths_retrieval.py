"""Offline K-Paths-style retrieval, without changing a training model.

Reference: K-Paths KDD2025, Algorithm1 and the audited author's
two_nodes_in_graph: Yen shortest_simple_paths followed by ordered relation
sequence filtering. This adapter preserves parallel PrimeKG facts outside the
simple connectivity graph. A declared enumeration/time budget can censor a
query; censored output must not be called exact global top-K.

The adapted mode supplements constrained four-edge target/function routes and
keeps two entity variants per relation template. These are task adaptations,
not the paper's default3-hop, one-path-per-template recipe.
"""
from collections import Counter, defaultdict
import heapq
import itertools
import math
import time
import networkx as nx

FUNCTIONS = {'bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 'pathway_protein'}
ALLOWED = FUNCTIONS | {'drug_target', 'drug_enzyme', 'drug_transporter', 'drug_carrier',
    'protein_protein', 'prime_disease_protein', 'prime_drug_disease_indication',
    'prime_drug_disease_contraindication', 'prime_drug_disease_off_label_use'}


class KPathsIndex:
    def __init__(self, facts, node_types):
        self.facts = facts
        self.types = node_types
        self.graph = nx.DiGraph()
        self.edges = defaultdict(list)
        self.targets = defaultdict(dict)
        self.function_neighbors = defaultdict(set)
        self.function_members = defaultdict(set)
        self.function_fact = {}
        for fid, f in enumerate(facts):
            family, h, t = f['family'], f['head'], f['tail']
            if family not in ALLOWED:
                raise ValueError('Unsupported/label-derived fact: ' + family)
            if node_types[h] == 'drug' and node_types[t] == 'drug':
                raise ValueError('Direct DDI shortcuts are forbidden')
            # Connectivity alone uses DiGraph. The full set of parallel typed
            # facts is kept in self.edges, never overwritten in edge attrs.
            self.graph.add_edge(h, t)
            self.graph.add_edge(t, h)
            self.edges[h, t].append(fid)
            self.edges[t, h].append(fid)
            if family == 'drug_target':
                self.targets[h][t] = fid
            if family in FUNCTIONS:
                self.function_neighbors[t].add(h)
                self.function_members[h].add(t)
                self.function_fact[h, t] = fid
        for ids in self.edges.values():
            ids.sort(key=lambda i: (facts[i]['family'], facts[i]['head'], facts[i]['tail'], i))
        self.degree = dict(self.graph.degree())

    def path(self, entities, fact_ids):
        assert len(set(entities)) == len(entities), 'A path must be loopless'
        assert len(entities) == len(fact_ids) + 1
        roles, relations, rows = [], [], []
        for u, v, fid in zip(entities, entities[1:], fact_ids):
            f = self.facts[fid]
            assert {u, v} == {f['head'], f['tail']}
            forward = u == f['head']
            roles.append([0, 1] if forward else [1, 0])
            # PPI is treated as undirected, not a causal regulatory relation.
            relations.append(f['family'] + ('' if forward or f['family'] == 'protein_protein' else '^-1'))
            rows.append(f['row'])
        kinds = {self.types[v] for v in entities[1:-1]}
        family = ('pathway' if 'pathway' in kinds else
            'go' if kinds & {'biological_process', 'molecular_function', 'cellular_component'} else
            'disease' if 'disease' in kinds else 'shared_protein' if len(entities) == 3 else 'ppi')
        cost = len(fact_ids) + .1 * sum(math.log1p(self.degree.get(v, 0)) for v in entities[1:-1])
        return dict(entities=list(entities), fact_ids=list(fact_ids), relations=relations,
            roles=roles, source_rows=rows, family=family, hops=len(fact_ids),
            heuristic_cost=cost, biological_causality_claim=False)

    def _fallback(self, a, b, k):
        paths, seen = [], set()
        for root in (a, b):
            if root not in self.graph:
                continue
            for neighbor in self.graph.neighbors(root):
                if self.types[neighbor] == 'drug':
                    continue
                for fid in self.edges[root, neighbor]:
                    f = self.facts[fid]
                    key = (root, f['family'])
                    if key not in seen:
                        seen.add(key)
                        paths.append(self.path([root, neighbor], [fid]))
                    if len(paths) >= k:
                        return paths
        return paths

    def reference(self, a, b, k=10, max_hops=3, variants=1,
                  candidate_budget=128, seconds_budget=2.):
        if not k > 0 or not variants > 0 or candidate_budget < 1 or seconds_budget <= 0:
            raise ValueError('Invalid retrieval budget')
        if a is None or b is None:
            return dict(paths=[], fallback=self._fallback(a, b, k), censored=False,
                entity_paths_examined=0, seconds=0., reason='unmapped query endpoint')
        a, b = sorted((a, b))
        if a == b:
            return dict(paths=[], fallback=[], censored=False, entity_paths_examined=0,
                seconds=0., reason='same drug; no invented connecting path')
        started = time.perf_counter()
        count, signature_count, found, censored = 0, Counter(), [], False
        if a in self.graph and b in self.graph:
            # Keep the two endpoints; exclude other drugs as intermediates.
            # The index already excludes synergy/Cell labels and direct DDI.
            view = nx.subgraph_view(self.graph,
                filter_node=lambda v: v in (a, b) or self.types[v] != 'drug')
            try:
                generator = nx.shortest_simple_paths(view, a, b)
                for entities in generator:
                    if len(entities) - 1 > max_hops:
                        break
                    if count >= candidate_budget or time.perf_counter() - started > seconds_budget:
                        censored = True
                        break
                    count += 1
                    variants_at_edges = [self.edges[u, v] for u, v in zip(entities, entities[1:])]
                    for fids in itertools.product(*variants_at_edges):
                        p = self.path(entities, fids)
                        signature = tuple(p['relations'])
                        if signature_count[signature] < variants:
                            found.append(p)
                            signature_count[signature] += 1
                        if len(found) == k:
                            break
                    if len(found) == k:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        return dict(paths=found, fallback=[] if found else self._fallback(a, b, k),
            censored=censored, entity_paths_examined=count, seconds=time.perf_counter()-started,
            max_hops=max_hops, variants_per_relation_sequence=variants,
            candidate_budget=candidate_budget, soft_seconds_budget=seconds_budget,
            limitation='Time is checked between Yen yields, not a hard per-call timeout')

    def adapted(self, a, b, k=10, candidate_budget=128, seconds_budget=2., max_members=200):
        if a is None or b is None:
            return self.reference(a, b, k=k)
        a, b = sorted((a, b))
        if a == b:
            return self.reference(a, b, k=k)
        started = time.perf_counter()
        base = self.reference(a, b, k=40, max_hops=3, variants=2,
            candidate_budget=candidate_budget, seconds_budget=seconds_budget)
        ta, tb = self.targets.get(a, {}), self.targets.get(b, {})
        left = set().union(*(self.function_neighbors[p] for p in ta)) if ta else set()
        right = set().union(*(self.function_neighbors[p] for p in tb)) if tb else set()
        anchors = sorted(left & right, key=lambda v: (len(self.function_members[v]), v))
        heaps = {'go': [], 'pathway': []}
        enumerated, broad, censored = 0, 0, False
        deadline = time.perf_counter() + seconds_budget
        for anchor in anchors:
            if time.perf_counter() > deadline:
                censored = True
                break
            members = self.function_members[anchor]
            if len(members) > max_members:
                broad += 1
                continue
            for x in sorted(set(ta) & members):
                for y in sorted(set(tb) & members):
                    if x == y:
                        continue
                    p = self.path([a, x, anchor, y, b],
                        [ta[x], self.function_fact[anchor, x], self.function_fact[anchor, y], tb[y]])
                    enumerated += 1
                    h = heaps[p['family']]
                    key = (-p['heuristic_cost'], tuple(-v for v in p['entities']))
                    if len(h) < 64:
                        heapq.heappush(h, (key, p))
                    elif key > h[0][0]:
                        heapq.heapreplace(h, (key, p))
        pools = defaultdict(list)
        for p in base['paths'] + [item[1] for h in heaps.values() for item in h]:
            pools[p['family']].append(p)
        for pool in pools.values():
            pool.sort(key=lambda p: (p['heuristic_cost'], p['entities'], p['fact_ids']))
        result, signatures, seen = [], Counter(), set()
        families = ['shared_protein', 'ppi', 'pathway', 'go', 'disease']
        while len(result) < k and any(pools.values()):
            for family in families:
                pool = pools[family]
                while pool:
                    p = pool.pop(0)
                    identity = (tuple(p['entities']), tuple(p['fact_ids']))
                    signature = tuple(p['relations'])
                    if identity in seen or signatures[signature] >= 2:
                        continue
                    seen.add(identity)
                    signatures[signature] += 1
                    result.append(p)
                    break
                if len(result) == k:
                    break
        return dict(paths=result, fallback=[] if result else self._fallback(a, b, k),
            censored=base['censored'] or censored, base_search=base,
            four_hop_candidates_enumerated=enumerated, broad_function_sets_skipped=broad,
            seconds=time.perf_counter()-started, variants_per_relation_sequence=2,
            four_hop_scope='Drug-target-function-target-Drug only',
            max_original_function_members=max_members, exact_author_reproduction=False)

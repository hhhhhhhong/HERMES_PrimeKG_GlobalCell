"""Label-independent, target-anchored selection before hypergraph propagation.

All kept edges are subsets of audited input facts. This is a deterministic
global evidence filter, not query-specific retrieval, GO semantic clustering,
or an assertion that a shared pathway proves synergy. Cell observations and
training-fold synergy edges are copied unchanged and never used for ranking.
"""
from collections import defaultdict
import math

DTI = ('drug_target', 'drug_enzyme', 'drug_carrier', 'drug_transporter')
FUNCTIONS = ('bioprocess_protein', 'molfunc_protein', 'cellcomp_protein', 'pathway_protein')


def select_graph(graph, bridges_per_target=16, max_function_members=200):
    if bridges_per_target < 1 or max_function_members < 2:
        raise ValueError('Invalid fixed evidence selection budget')
    types = graph['node_types']
    if hasattr(types, 'tolist'):
        types = types.tolist()
    kinds = [graph['type_names'][t] for t in types]
    groups = graph['groups']
    protein = lambda v: kinds[v] == 'gene_protein'
    drug = lambda v: v < graph['n_drug']
    target_drugs = defaultdict(set)
    for e in groups.get('drug_target', []):
        ds = {v for v in e['members'] if drug(v)}
        for v in e['members']:
            if protein(v):
                target_drugs[v].update(ds)
    targets = set(target_drugs)
    if not targets:
        raise ValueError('No audited drug_target anchors; do not invent evidence')
    neighbors = defaultdict(set)
    for e in groups.get('protein_protein', []):
        ps = e['members']
        if len(ps) != 2 or not all(protein(v) for v in ps):
            raise ValueError('PPI must remain a true binary protein fact')
        a, b = ps
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    candidates = {}
    for t in sorted(targets):
        for v in neighbors[t] - targets:
            linked = neighbors[v] & targets
            supporters = set().union(*(target_drugs[p] for p in linked))
            if len(linked) >= 2 and len(supporters) >= 2:
                # Prefer cross-drug support per unit of hub connectivity.
                score = len(supporters) / math.sqrt(max(len(neighbors[v]), 1))
                candidates[v] = (score, len(linked), len(supporters))
    bridges = set()
    for t in sorted(targets):
        ranked = sorted(neighbors[t] & candidates.keys(),
            key=lambda v: (-candidates[v][0], -candidates[v][1], v))
        bridges.update(ranked[:bridges_per_target])
    retained_proteins = targets | bridges
    selected = dict(groups)
    selected['protein_protein'] = [e for e in groups.get('protein_protein', [])
        if set(e['members']) <= retained_proteins and set(e['members']) & targets]
    function_audits = {}
    for family in FUNCTIONS:
        kept, rejected = [], defaultdict(int)
        for edge in groups.get(family, []):
            all_members = edge['members']
            proteins = [v for v in all_members if protein(v)]
            if len(proteins) > max_function_members:
                rejected['original_set_too_broad'] += 1
                continue
            members = [v for v in proteins if v in retained_proteins]
            direct = set(members) & targets
            supporters = set().union(*(target_drugs[p] for p in direct)) if direct else set()
            if len(members) < 2 or len(supporters) < 2:
                rejected['insufficient_cross_drug_target_support'] += 1
                continue
            take = {v for v in all_members if not protein(v)} | set(members)
            positions = [p for v, p in zip(all_members, edge['positions']) if v in take]
            new_members = [v for v in all_members if v in take]
            assert len(new_members) == len(positions) and set(new_members) <= set(all_members)
            kept.append(dict(edge, members=new_members, positions=positions,
                selection=dict(original_proteins=len(proteins), retained_proteins=len(members),
                    supporting_drugs=len(supporters), original_proof_retained=True)))
        selected[family] = kept
        function_audits[family] = dict(before=len(groups.get(family, [])), after=len(kept), rejected=dict(rejected))
    # Keep drug-disease meanings and weights, while restricting disease-protein
    # facts to proteins on the selected target/bridge mechanism backbone.
    selected['prime_disease_protein'] = [e for e in groups.get('prime_disease_protein', [])
        if any(protein(v) and v in retained_proteins for v in e['members'])]
    # Empty channels have no factual messages. The model keeps their original
    # trainable parameters and original token universe, with masks set false.
    selected = {name: edges for name, edges in selected.items() if edges}
    for name in DTI + ('cell_expression', 'Synergy_Hyperedge',
        'prime_drug_disease_indication', 'prime_drug_disease_contraindication',
        'prime_drug_disease_off_label_use'):
        assert selected.get(name, []) == groups.get(name, []), name
    result = dict(graph, groups=selected)
    audit = dict(policy='target-anchored global evidence; no labels in ranking',
        bridges_per_target=bridges_per_target, max_original_function_proteins=max_function_members,
        direct_target_proteins=len(targets), retained_bridge_proteins=len(bridges),
        target_nodes=sorted(targets), bridge_nodes=sorted(bridges),
        candidate_bridges=len(candidates),
        relations={k: dict(before=len(v), after=len(selected.get(k, []))) for k, v in groups.items()},
        functions=function_audits, Cell_expression_and_routes_unchanged=True,
        Drug_DTI_and_indication_weights_unchanged=True, synergy_unchanged=True,
        graph_node_IDs_and_CODER_unchanged=True, selection_uses_synergy_labels=False,
        query_specific=False, semantic_GO_deduplication=False)
    result['selection_audit'] = audit
    return result, audit

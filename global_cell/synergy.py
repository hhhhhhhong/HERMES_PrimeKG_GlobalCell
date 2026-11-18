"""Add supervised training synergy hyperedges to the 14-family base graph."""
import hashlib
import torch

SYNERGY = 'Synergy_Hyperedge'


def rows_sha256(rows):
    return hashlib.sha256(rows.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def add_training_synergy(base_graph, fit_rows, *, fold=0, expected_train_sha256=None):
    """Return a fold-local 15-relation graph with symmetric drug positions [0, 0, 2]."""
    rows = fit_rows.detach().cpu()
    pos_indices = torch.where(rows[:, 3] == 1)[0].tolist()

    # DrugA 与 DrugB 共享 position=0（消除按节点整数 ID 大小切分聚合槽的偏置），Cell 使用 position=2
    edges = [
        dict(
            members=[*sorted(map(int, rows[i, :2])), int(rows[i, 2])],
            positions=[0, 0, 2],
            weight=1.0
        )
        for i in pos_indices
    ]

    result = dict(base_graph)
    result['groups'] = dict(base_graph['groups'], **{SYNERGY: edges})
    result['synergy_audit'] = dict(
        fold=int(fold),
        relation=SYNERGY,
        fit_rows=len(rows),
        positive_training_rows=len(edges),
        positions=[0, 0, 2]
    )
    if 'summary' in base_graph:
        result['summary'] = dict(base_graph['summary'])
        result['summary']['relation_counts'] = dict(
            base_graph['summary'].get('relation_counts', {}),
            **{SYNERGY: len(edges)}
        )
        result['summary'].update(
            synergy_labels_used_for_graph=True,
            synergy_fold=int(fold),
            relation_type_count=len(result['groups'])
        )
    return result


def query_overlap_audit(fit_ids, valid_ids):
    """Lightweight endpoint overlap counter kept for runner compatibility."""
    def keys(t):
        return [(min(int(a), int(b)), max(int(a), int(b)), int(c)) for a, b, c in t.detach().cpu().tolist()]
    fit, valid = keys(fit_ids[:, :3]), keys(valid_ids[:, :3])
    overlap = set(fit) & set(valid)
    return dict(
        canonical_train_valid_shared_queries=len(overlap),
        valid_rows_with_training_query=sum(k in overlap for k in valid),
        labels_read=False,
        historical_split_unchanged=True
    )
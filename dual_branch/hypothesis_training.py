"""Exact per-query-batch target exclusion, accumulated to one update/epoch."""
import math
import torch


def train_epoch(model, optimizer, rows, labels, chunk=1024, max_grad_norm=1.):
    if len(rows) == 0 or chunk < 1 or len(labels) != len(rows):
        raise ValueError('Nonempty aligned supervision and positive query chunk required')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Cache only PRE-propagation modality/CODER projections, independent of
    # any Synergy incidence. The label-bearing evidence table is never shared.
    base, initial = model.base_features(), model.semantic.initial_nodes()
    leaf_base = base.detach().requires_grad_(True)
    leaf_initial = initial.detach().requires_grad_(True)
    total, exclusions = 0., []
    for start in range(0, len(rows), chunk):
        stop = min(start+chunk, len(rows))
        batch = rows[start:stop]
        with model.exclusion_context(batch):
            table = model.knowledge_table(leaf_base, initial=leaf_initial)
            prediction, _ = model(batch, table, leaf_base)
            loss = torch.nn.functional.binary_cross_entropy(prediction, labels[start:stop], reduction='sum') / len(rows)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite BCE; no optimizer update performed')
            # Must finish recomputation/backward before graph context restore.
            loss.backward()
            total += float(loss.detach())
            exclusions.append(dict(model.exclusion_audit))
        del table, prediction, loss
    grad_base = leaf_base.grad if leaf_base.grad is not None else torch.zeros_like(leaf_base)
    grad_initial = leaf_initial.grad if leaf_initial.grad is not None else torch.zeros_like(leaf_initial)
    torch.autograd.backward((base, initial), (grad_base, grad_initial))
    components = {name: math.sqrt(sum(float(p.grad.detach().float().square().sum())
        for p in params if p.grad is not None)) for name, params in model.components().items()}
    if any(not math.isfinite(value) for value in components.values()):
        raise FloatingPointError('Nonfinite gradients; no optimizer update performed')
    preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True))
    cf_query_grad = model.gamma.query[0].weight.grad[:, 3*model.semantic.dim:]
    optimizer.step()
    return total, dict(component_gradient_L2=components, global_gradient_L2_before_clip=preclip,
        clipping_scale=min(1., max_grad_norm/(preclip+1e-6)), max_gradient_L2=max_grad_norm,
        collaborative_query_weight_gradient_L2=float(cf_query_grad.norm()),
        updates_this_epoch=1, query_batches=len(exclusions), graph_tables_recomputed=len(exclusions),
        exclusions=exclusions, only_prepropagation_features_cached=True)


@torch.no_grad()
def evaluate(model, rows, chunk=1024):
    """One fixed evaluation graph excludes ALL evaluated query keys.

    This also removes an overlapping training query without reading its
    validation label. Query chunk size therefore cannot change this graph.
    """
    model.eval()
    with model.exclusion_context(rows):
        base = model.base_features()
        table = model.knowledge_table(base)
        return torch.cat([model(rows[start:start+chunk], table, base)[0].cpu()
                          for start in range(0, len(rows), chunk)])

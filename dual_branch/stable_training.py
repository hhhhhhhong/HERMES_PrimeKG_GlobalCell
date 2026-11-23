"""Exact-full-fit BCE with one finite, clipped update and persistent diagnostics."""
import math
import torch


def train_epoch(model, optimizer, rows, labels, chunk=1024, max_grad_norm=1.):
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
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite BCE; optimizer was not updated')
        loss.backward()
        total += float(loss.detach())
    torch.autograd.backward((table, base), (leaf_table.grad, leaf_base.grad))
    # Report true global L2 norms rather than sums of per-tensor norms.
    components = {k: math.sqrt(sum(float(p.grad.detach().float().square().sum())
        for p in ps if p.grad is not None)) for k, ps in model.components().items()}
    bad = [key for key, value in components.items() if not math.isfinite(value)]
    if bad:
        raise FloatingPointError('Nonfinite component gradients; update refused: ' + ', '.join(bad))
    preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
        error_if_nonfinite=True))
    optimizer.step()
    return total, dict(component_gradient_L2=components, global_gradient_L2_before_clip=preclip,
        clipping_scale=min(1., max_grad_norm / (preclip + 1e-6)),
        max_gradient_L2=max_grad_norm, updates_this_epoch=1)


@torch.no_grad()
def residual_audit(model, rows, chunk=256):
    model.eval()
    base, table = model.base_features(), model.knowledge_table()
    maxima = dict(knowledge=0., transformer=0., combined=0.)
    for start in range(0, len(rows), chunk):
        _, d = model(rows[start:start + chunk], table, base, capture=True)
        denom = torch.linalg.vector_norm(d['own'], dim=-1).clamp_min(1e-12)
        for key, value in [('knowledge', d['delta']), ('transformer', d['transformer_delta']),
                           ('combined', d['delta'] + d['transformer_delta'])]:
            maxima[key] = max(maxima[key], float((torch.linalg.vector_norm(value, dim=-1) / denom).max()))
    assert maxima['knowledge'] <= .25001 and maxima['transformer'] <= .10001
    assert maxima['combined'] <= .35002
    return dict(max_per_token_norm_ratio=maxima, rows=len(rows),
        reference='pre-knowledge modal token, detached scale', bounds_verified=True)

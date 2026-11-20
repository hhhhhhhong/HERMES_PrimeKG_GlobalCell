"""Optimizer-only diagnostic; no new layers or changes to model.forward.

The original Initializer and prediction head keep Adam's coupled decay 0.01.
All added knowledge parameters, including both sets of gates, use decay zero.
The matched mode exists for correctness checks, not an additional formal fit.
"""
import hashlib
import math
from pathlib import Path

import torch

from global_cell.common import read, write


def is_backbone(name):
    return name.startswith(('initializer.', 'head.'))


def parameter_groups(model, mode):
    assert mode in ('matched_decay', 'knowledge_no_decay')
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if mode == 'matched_decay':
        groups = [dict(params=[p for _, p in named], weight_decay=.01)]
    else:
        groups = [dict(params=[p for n, p in named if is_backbone(n)], weight_decay=.01),
                  dict(params=[p for n, p in named if not is_backbone(n)], weight_decay=0.)]
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(named)
    assert set(ids) == {id(p) for _, p in named}
    by_id = {id(p): n for n, p in named}
    metadata = [dict(weight_decay=g['weight_decay'], parameter_tensors=len(g['params']),
                     elements=sum(p.numel() for p in g['params']),
                     names=[by_id[id(p)] for p in g['params']]) for g in groups]
    return groups, metadata


def monitored(epoch):
    return epoch in (1, 2, 20) or epoch % 100 == 0


def category(name):
    if name.startswith('propagation_gates.'):
        return 'propagation_gates'
    if name in ('readout.drug_gate', 'readout.cell_gate'):
        return 'readout_gates'
    if name.startswith('external_projection.'):
        return 'coder_projection'
    if name.startswith(('encoder.', 'extra_encoders.')):
        return 'alpha_beta'
    if is_backbone(name):
        return 'backbone'
    return name.split('.')[0]


def optimizer_audit(model, groups):
    """Inspect supervised gradients before Adam adds its coupled decay term."""
    decay = {id(p): g['weight_decay'] for g in groups for p in g['params']}
    values = {}
    gates = {}
    for name, p in model.named_parameters():
        key = category(name)
        a = values.setdefault(key, dict(elements=0, parameter_sq=0., task_gradient_sq=0.,
                                        decay_gradient_sq=0., reference_decay_sq=0., dot=0.,
                                        parameters_without_gradient=0))
        value = p.detach()
        p2 = float(value.square().sum())
        a['elements'] += p.numel()
        a['parameter_sq'] += p2
        if p.grad is not None:
            g = p.grad.detach()
            a['task_gradient_sq'] += float(g.square().sum())
            a['decay_gradient_sq'] += decay[id(p)] ** 2 * p2
            a['reference_decay_sq'] += .01 ** 2 * p2
            a['dot'] += .01 * float((g * value).sum())
        else:
            a['parameters_without_gradient'] += 1
        if key in ('propagation_gates', 'readout_gates'):
            gates[name] = dict(rms=math.sqrt(p2 / p.numel()), abs_max=float(value.abs().max()))
    for a in values.values():
        a['parameter_rms'] = math.sqrt(a.pop('parameter_sq') / a['elements'])
        a['task_gradient_norm'] = math.sqrt(a.pop('task_gradient_sq'))
        a['actual_decay_gradient_norm'] = math.sqrt(a.pop('decay_gradient_sq'))
        a['reference_decay_gradient_norm'] = math.sqrt(a.pop('reference_decay_sq'))
        denominator = a['task_gradient_norm'] * a['reference_decay_gradient_norm']
        a['task_reference_decay_cosine'] = a.pop('dot') / denominator if denominator else None
        a['reference_decay_to_task_ratio'] = (a['reference_decay_gradient_norm'] / a['task_gradient_norm']
                                             if a['task_gradient_norm'] else None)
    return dict(groups=values, gates=gates,
                reference_decay_is_counterfactual_not_an_actual_second_update=True)


def observed_adam_class(original, model, directory, mode):
    """Keep the Adam implementation; change only explicit parameter groups."""
    class ObservedAdam(original):
        def __init__(self, params, *args, **kwargs):
            supplied = list(params)
            assert {id(p) for p in supplied} == {id(p) for p in model.parameters()}
            groups, metadata = parameter_groups(model, mode)
            super().__init__(groups, *args, **kwargs)
            self.diagnostic_epoch = 0
            write(directory / 'OPTIMIZER_GROUPS.json', dict(mode=mode, groups=metadata,
                implementation='torch.optim.Adam; coupled weight decay', learning_rate_unchanged=True))

        def step(self, closure=None):
            self.diagnostic_epoch += 1
            epoch = self.diagnostic_epoch
            diagnostic = optimizer_audit(model, self.param_groups) if monitored(epoch) else None
            result = super().step(closure)
            if diagnostic is not None:
                diagnostic.update(epoch=epoch, timing='supervised gradients and parameters before optimizer.step',
                                  propagation_forward=(model.propagation_audit
                                      if model.capture_propagation else None))
                write(directory / ('OPTIMIZER_epoch%d.json' % epoch), diagnostic)
            return result

    return ObservedAdam


def final_ablations(model, directory, raw, evaluate, get_metrics):
    """Fixed selected checkpoint; validation interventions never select a fit."""
    selection = read(directory / 'selection.json')
    checkpoint = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == selection['epoch']
    model.load_state_dict(checkpoint['model_state'], strict=True)
    device = next(model.parameters()).device
    valid = raw['valid']['rows'].to(device)
    if read(directory / 'EXPERIMENT.json')['stage'] == 'check':
        valid = valid[:1024]
    model.eval()
    readout_stats = []

    def capture_readout(module, inputs, output):
        own = inputs[0]
        readout_stats.append((float((output.detach() - own.detach()).square().sum()),
                              float(own.detach().square().sum()), own.numel()))

    hook = model.readout.register_forward_hook(capture_readout)
    model.capture_propagation = True
    real = evaluate(model, valid, chunk=1024)
    hook.remove()
    propagation = model.propagation_audit
    model.capture_propagation = False
    gate_values = {n: dict(rms=float(p.detach().square().mean().sqrt()),
                          abs_max=float(p.detach().abs().max()))
                   for n, p in model.named_parameters()
                   if n.startswith('propagation_gates.') or n in ('readout.drug_gate', 'readout.cell_gate')}
    saved = [p.detach().clone() for p in model.propagation_gates]
    with torch.no_grad():
        for p in model.propagation_gates:
            p.zero_()
    propagation_off = evaluate(model, valid, chunk=1024)
    with torch.no_grad():
        for p, value in zip(model.propagation_gates, saved):
            p.copy_(value)
    saved_readout = [model.readout.drug_gate.detach().clone(), model.readout.cell_gate.detach().clone()]
    with torch.no_grad():
        model.readout.drug_gate.zero_()
        model.readout.cell_gate.zero_()
    knowledge_off = evaluate(model, valid, chunk=1024)
    with torch.no_grad():
        model.readout.drug_gate.copy_(saved_readout[0])
        model.readout.cell_gate.copy_(saved_readout[1])
    restored = evaluate(model, valid, chunk=1024)
    torch.testing.assert_close(restored, real, rtol=0, atol=2e-6)
    labels = valid[:, 3].detach().cpu().to(real.dtype)
    metrics = {}
    for name, pred in [('real', real), ('propagation_off', propagation_off), ('knowledge_readout_off', knowledge_off)]:
        auc, ap = get_metrics(labels.numpy(), pred.numpy())
        difference = (pred - real).abs()
        metrics[name] = dict(valid_auc=float(auc), valid_aupr=float(ap),
                             BCE=float(torch.nn.functional.binary_cross_entropy(pred, labels)),
                             prediction_abs_change_mean=float(difference.mean()),
                             prediction_abs_change_max=float(difference.max()))
    assert abs(metrics['real']['valid_auc'] - selection['valid_auc']) < 1e-5
    assert abs(metrics['real']['valid_aupr'] - selection['valid_aupr']) < 1e-5
    delta_sq = sum(x[0] for x in readout_stats)
    own_sq = sum(x[1] for x in readout_stats)
    write(directory / 'SELECTED_DIAGNOSTICS.json', dict(status='passed', epoch=selection['epoch'],
        validation_rows=len(valid), fixed_checkpoint=True, checkpoint_state_restored=True,
        propagation=propagation, gates=gate_values,
        readout_delta_to_own_rms_ratio=math.sqrt(delta_sq / own_sq) if own_sq else None,
        interventions=metrics, test_evaluated=False, ablations_used_for_model_selection=False))


def state_hash(model):
    digest = hashlib.sha256()
    for p in model.parameters():
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

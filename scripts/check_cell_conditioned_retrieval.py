"""CPU-only, real complete-graph correctness checks; no formal training."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha, author_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], required=True)
    args = parser.parse_args()
    directory = ROOT / 'artifacts/cell_conditioned_retrieval_20261122' / ('cpu_' + args.dataset)
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    write(directory / 'progress.json', dict(status='running', phase='loading', pid=os.getpid()))
    import torch
    torch.set_num_threads(2)
    author_imports()
    from utils import set_seed_all
    from dual_branch.cell_conditioned_data import load
    from dual_branch.cell_conditioned_model import CellConditionedMechanismModel
    from dual_branch.gated_mechanism import GatedMechanismModel
    from dual_branch.stable_training import train_epoch
    from dual_branch.checks import balanced
    from global_cell.network import evaluate
    data = load(args.dataset, 0)
    set_seed_all(0)
    model = CellConditionedMechanismModel(data).eval()
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert all(p.requires_grad for p in model.parameters())
    ids = data['raw']['valid']['rows'][:, :3].long()
    paths = data['paths']
    # Pick a real drug pair with different candidate MEMBERSHIP across two
    # Cells. This is label-free and independent of validation outcomes.
    examples = None
    for row in ids:
        a, b = sorted(row[:2].tolist())
        slot = int(paths['pair_slot'][a * model.num_drug + b])
        if not slot:
            continue
        choices = paths['cell_pair_paths'][:, slot]
        first = frozenset(choices[0][choices[0] > 0].tolist())
        other = next((c for c in range(1, model.num_cell)
                      if frozenset(choices[c][choices[c] > 0].tolist()) != first), None)
        if other is not None:
            examples = (a, b, slot, other)
            break
    assert examples is not None, 'Real data must demonstrate changed membership'
    a, b, slot, other = examples
    query = torch.tensor([[a, b, model.num_drug], [a, b, model.num_drug + other]])
    selected, mask = model.select_paths(query)
    assert set(selected[0][mask[0]].tolist()) != set(selected[1][mask[1]].tolist())
    swapped = query[:, [1, 0, 2]]
    torch.testing.assert_close(model.select_paths(swapped)[0], selected, rtol=0, atol=0)
    with torch.no_grad():
        base, table = model.base_features(), model.knowledge_table()
        prediction, info = model(query, table, base, capture=True)
        torch.testing.assert_close(model(swapped, table, base)[0], prediction, rtol=0, atol=2e-6)
        separate = torch.cat([model(row[None], table, base)[0] for row in query])
        torch.testing.assert_close(separate, prediction, rtol=0, atol=2e-6)
        # Fix BOTH modal Cell and full global table, changing only the path
        # lookup. A context change therefore cannot be attributed to gamma's
        # existing Cell Query or to a recomputed global graph.
        saved_paths = model.cell_pair_paths[0, slot].clone()
        saved_mask = model.cell_pair_mask[0, slot].clone()
        try:
            model.cell_pair_paths[0, slot] = model.cell_pair_paths[other, slot]
            model.cell_pair_mask[0, slot] = model.cell_pair_mask[other, slot]
            changed, changed_info = model(query[:1], table, base, capture=True)
        finally:
            model.cell_pair_paths[0, slot] = saved_paths
            model.cell_pair_mask[0, slot] = saved_mask
        context_change = float((changed_info['context'] - info['context'][:1]).abs().max())
        assert context_change > 1e-8
        model.condition_on_cell = False
        off = model(query, table, base)[0]
        parent = GatedMechanismModel.forward(model, query, table, base)[0]
        torch.testing.assert_close(off, parent, rtol=0, atol=0)
        fixed, fixed_mask = model.select_paths(query)
        torch.testing.assert_close(fixed[0], fixed[1], rtol=0, atol=0)
        model.condition_on_cell = True
        no_knowledge = model(query, table, base, use_knowledge=False)[0]
        empty = model(query, table, base, empty_evidence=True)[0]
        torch.testing.assert_close(no_knowledge, empty, rtol=0, atol=0)
    del base, table, info, changed_info
    train = balanced(data['raw']['fit']['rows'], 64)
    valid = balanced(data['raw']['valid']['rows'], 64)
    optimizer = torch.optim.AdamW(model.optimizer_param_groups(), eps=1e-6)
    history = []
    for step in range(1, 4):
        loss, diagnostic = train_epoch(model, optimizer, train, train[:, 3].float())
        assert all(value > 0 for value in diagnostic['component_gradient_L2'].values())
        history.append(dict(step=step, BCE=loss, **diagnostic))
        write(directory / 'progress.json', dict(status='running', phase='correctness_updates',
            optimizer_updates=step, elapsed_seconds=time.monotonic() - started))
    model.eval()
    state_file = directory / 'correctness_final.pt'
    torch.save(dict(model_state=model.state_dict(), correctness_only=True,
        input_provenance=data['input_provenance'], condition_on_cell=True), state_file)
    set_seed_all(0)
    restored = CellConditionedMechanismModel(data).eval()
    restored_initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in restored.parameters())).hexdigest()
    assert restored_initial == initial
    restored.load_state_dict(torch.load(state_file, map_location='cpu', weights_only=True)['model_state'], strict=True)
    reference, reloaded = evaluate(model, valid), evaluate(restored, valid)
    torch.testing.assert_close(reference, reloaded, rtol=0, atol=0)
    previous = read(ROOT / 'artifacts/cell_conditioned_retrieval_20261122/ORIGINAL_SOURCE_SHA.json')
    assert all(sha(p) == value for p, value in previous.items())
    write(directory / 'CHECKS.json', dict(status='passed', dataset=args.dataset, seed=0, fold=0,
        full_global_graph=True, graph=data['graph']['summary'], supervised_rows=64, validation_rows=64,
        optimizer_updates=3, history=history, initialization_sha256=initial,
        all_parameters_trainable=True, formal_training=False, GPU_used=False,
        candidate_membership_changes_before_gamma=True, drug_exchange_passed=True,
        batch_isolation_passed=True, Cell_off_equals_original_gated_forward_error=float((off-parent).abs().max()),
        fixed_Query_and_global_table_path_intervention_context_change=context_change,
        fixed_Query_and_global_table_path_intervention_prediction_change=float((changed-prediction[:1]).abs().max()),
        strict_disk_reload_error=float((reference-reloaded).abs().max()),
        unknown_expression_not_removed=True, empty_knowledge_passed=True,
        original85_sources_unchanged=True, no_performance_claim=True,
        elapsed_seconds=time.monotonic() - started))
    write(directory / 'progress.json', dict(status='complete', optimizer_updates=3,
        elapsed_seconds=time.monotonic() - started, correctness_only=True))
    print(args.dataset, 'CPU complete-graph correctness passed', flush=True)


if __name__ == '__main__':
    main()

"""CPU-only full real graph check; no GPU/fit submission or old checkpoints."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import read, write, sha, author_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['ALMANAC', 'ONEIL'])
    args = parser.parse_args()
    directory = ROOT / 'artifacts/numerical_safety_20261122' / ('cpu_' + args.dataset)
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    write(directory / 'progress.json', dict(status='running', phase='loading', pid=os.getpid()))
    import torch
    torch.set_num_threads(2)
    author_imports()
    from utils import set_seed_all
    from dual_branch.cell_conditioned_data import load
    from dual_branch.cell_conditioned_model import CellConditionedMechanismModel
    from dual_branch.numerical_safety import SafeCellConditionedMechanismModel, SafeGatedMechanismModel
    from dual_branch.stable_training import train_epoch
    from dual_branch.checks import balanced
    from global_cell.network import evaluate
    import dual_branch.mechanism_model as mechanisms
    data = load(args.dataset, 0)
    set_seed_all(0)
    reference = CellConditionedMechanismModel(data).eval()
    original_rng = torch.random.get_rng_state().clone()
    set_seed_all(0)
    # Existing formal constructor guards forbid ALL state/checkpoint loading.
    # The numerical replacement only copies freshly initialized tensors.
    def forbidden(*args, **kwargs):
        raise AssertionError('Constructor attempted a checkpoint/state load')
    with patch.object(torch, 'load', forbidden), patch.object(torch.nn.Module, 'load_state_dict', forbidden):
        model = SafeCellConditionedMechanismModel(data).eval()
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert reference.state_dict().keys() == model.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    state_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    set_seed_all(0)
    gated = SafeGatedMechanismModel(data).eval()
    assert gated.state_dict().keys() == model.state_dict().keys()
    for key, value in gated.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    del gated
    rows = balanced(data['raw']['fit']['rows'], 64)
    valid = balanced(data['raw']['valid']['rows'], 64)
    with torch.no_grad():
        base, table = model.base_features(), model.knowledge_table()
        original_base, original_table = reference.base_features(), reference.knowledge_table()
        torch.testing.assert_close(base, original_base, rtol=0, atol=0)
        torch.testing.assert_close(table, original_table, rtol=0, atol=0)
        expected = reference(valid, original_table, original_base)[0]
        prediction = model(valid, table, base)[0]
        torch.testing.assert_close(prediction, expected, rtol=0, atol=2e-6)
        swap = valid.clone()
        swap[:, :2] = swap[:, [1, 0]]
        torch.testing.assert_close(model(swap, table, base)[0], prediction, rtol=0, atol=2e-6)
        chunked = torch.cat([model(valid[i:i+7], table, base)[0] for i in range(0, len(valid), 7)])
        torch.testing.assert_close(chunked, prediction, rtol=0, atol=2e-6)
        off = model(valid, table, base, use_knowledge=False)[0]
        empty = model(valid, table, base, empty_evidence=True)[0]
        torch.testing.assert_close(off, empty, rtol=0, atol=0)
    forward_error = float((prediction-expected).abs().max())
    del reference, base, table, original_base, original_table
    # Trace one actual complete-graph train forward. Alpha/beta samples three
    # independent message masks; the Refiner receives the FULL untouched
    # incidence and samples its own one structural mask, not 0.8^2 incidences.
    rand, incidence_fn = torch.rand, mechanisms.kept_synergy_incidence
    mask_shapes, witness = [], {}
    def trace_rand(*args, **kwargs):
        value = rand(*args, **kwargs)
        mask_shapes.append(tuple(value.shape))
        return value
    def trace_incidence(incidence, edges, training, device):
        assert torch.equal(incidence, model.synergy_incidence)
        assert edges == model.synergy_edges and training
        selected, weights = incidence_fn(incidence, edges, training, device)
        assert bool((weights == 1.).all())
        assert selected.shape[1] == 3*len(weights)
        witness.update(original_edges=edges, retained_refiner_edges=len(weights),
            Refiner_receives_full_original_incidence=True, Refiner_kept_weights_are_one=True)
        return selected, weights
    model.train()
    with torch.no_grad(), patch.object(torch, 'rand', trace_rand), patch.object(mechanisms, 'kept_synergy_incidence', trace_incidence):
        train_table = model.knowledge_table()
        assert bool(torch.isfinite(train_table).all())
    del train_table
    edge_count = model.synergy_edges
    assert mask_shapes.count((edge_count, 1, 1)) == 3, mask_shapes
    assert mask_shapes.count((edge_count,)) == 1, mask_shapes
    witness.update(alpha_beta_message_mask_calls=3, Refiner_structural_mask_calls=1,
        branch_masks_are_parallel=True, all_3_alpha_beta_groups_include_Synergy=True)
    optimizer = torch.optim.AdamW(model.optimizer_param_groups(), eps=1e-6)
    history = []
    for step in range(1, 3):
        loss, diagnostic = train_epoch(model, optimizer, rows, rows[:, 3].float())
        assert all(v > 0 for v in diagnostic['component_gradient_L2'].values()), diagnostic
        history.append(dict(step=step, BCE=loss, **diagnostic))
        write(directory / 'progress.json', dict(status='running', phase='correctness_updates', optimizer_updates=step))
    state_file = directory / 'correctness_final.pt'
    torch.save(dict(model_state=model.state_dict(), correctness_only=True), state_file)
    restored = SafeCellConditionedMechanismModel(data).eval()
    restored.load_state_dict(torch.load(state_file, weights_only=True, map_location='cpu')['model_state'], strict=True)
    model.eval()
    actual, reloaded = evaluate(model, valid), evaluate(restored, valid)
    torch.testing.assert_close(actual, reloaded, rtol=0, atol=0)
    old = read(ROOT / 'artifacts/numerical_safety_20261122/ORIGINAL_SOURCE_SHA.json')
    assert all(sha(p) == digest for p, digest in old.items())
    write(directory / 'CHECKS.json', dict(status='passed', dataset=args.dataset, seed=0, fold=0,
        supervised_rows=64, validation_rows=64, full_global_graph=True, optimizer_updates=2,
        history=history, state_initialization_sha256=state_hash, same_initial_parameters_and_RNG=True,
        constructor_checkpoint_load_forbidden_passed=True, all_downstream_trainable=all(p.requires_grad for p in model.parameters()),
        original_forward_error=forward_error, drug_exchange_passed=True, batch_isolation_passed=True,
        empty_knowledge_passed=True, strict_disk_reload_error=float((actual-reloaded).abs().max()),
        Synergy_Dropout_execution=witness, original93_sources_unchanged=True,
        GPU_used=False, formal_training=False, no_performance_claim=True,
        elapsed_seconds=time.monotonic()-started))
    write(directory / 'progress.json', dict(status='complete', optimizer_updates=2, correctness_only=True,
        elapsed_seconds=time.monotonic()-started))
    print(args.dataset, 'numerical safety full-graph CPU check passed', flush=True)


if __name__ == '__main__':
    main()

"""CPU-only real-data integration check; never submit or start formal fits."""
import argparse
import hashlib
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], default='ONEIL')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from dual_branch.common import ROOT, environment
    environment()
    import torch
    from dual_branch.data import load
    from dual_branch.checks import balanced, verify_gradients
    from dual_branch.model import DualBranchModel
    from dual_branch.transformer_prediction import TransformerDualBranchModel
    from global_cell.common import author_imports, read, write, sha
    from global_cell.network import train_epoch

    torch.set_num_threads(2)
    author_imports()
    from utils import set_seed_all
    out = ROOT / 'artifacts/prediction_transformer_20261120' / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'CPU_CHECK.json'
    assert not receipt.exists(), 'Preserve completed checks; choose a new dated check for reruns'
    started = time.monotonic()
    print('Reading existing real inputs on CPU:', args.dataset, flush=True)
    data = load(args.dataset, 0)
    set_seed_all(0)
    # Input caches have been read. Constructors must not load model weights.
    old_load, old_state_load = torch.load, torch.nn.Module.load_state_dict

    def forbidden(*unused, **kwargs):
        raise AssertionError('Fresh construction attempted to load a checkpoint')

    torch.load = torch.nn.Module.load_state_dict = forbidden
    try:
        model = TransformerDualBranchModel(data)
    finally:
        torch.load, torch.nn.Module.load_state_dict = old_load, old_state_load
    assert all(p.requires_grad for p in model.parameters())
    base_hash = hashlib.sha256(b''.join(
        p.detach().numpy().tobytes() for n, p in model.named_parameters()
        if not n.startswith('prediction_transformer.'))).hexdigest()
    reference = ROOT / 'runs/dual_branch_gemini_cv5_seed0_20261119' / args.dataset / 'fold0'
    assert base_hash == read(reference / 'EXPERIMENT.json')['initialization_sha256']
    groups = model.optimizer_param_groups()
    names = {id(p): n for n, p in model.named_parameters()}
    audit = [dict(name=g['name'], lr=g['lr'], weight_decay=g['weight_decay'],
                  tensor_names=[names[id(p)] for p in g['params']]) for g in groups]
    previous = read(reference / 'OPTIMIZER_GROUPS.json')['groups']
    for expected, actual in zip(previous, audit):
        for key in ['name', 'lr', 'weight_decay', 'tensor_names']:
            assert expected[key] == actual[key], key
    rows = balanced(data['raw']['fit']['rows'], maximum=64)
    valid = balanced(data['raw']['valid']['rows'], maximum=32)
    model.eval()
    with torch.no_grad():
        base = model.base_features()
        ids = valid[:, :3].long().clone()
        ids[:, :2] = ids[:, :2].sort(dim=1).values
        own = base[ids]
        original = DualBranchModel.decode(model, own)
        initial = model.decode(own)
        torch.testing.assert_close(initial, original, rtol=0, atol=0)
        initial_error = float((initial - original).abs().max())
    print('Original initialization, optimizer groups and zero-head identity passed', flush=True)
    optimizer = torch.optim.Adam(groups)
    updates = []
    for step in range(1, 4):
        loss, gradient = train_epoch(model, optimizer, rows, rows[:, 3].float(), chunk=32)
        verify_gradients(gradient, step)
        assert gradient['prediction_transformer_output'] > 0
        if step == 1:
            assert gradient['prediction_transformer'] == 0
        else:
            assert gradient['prediction_transformer'] > 0
        updates.append(dict(step=step, correctness_subset_BCE=loss, gradients=gradient))
        print('CPU correctness update', step, 'passed', flush=True)
    model.eval()
    with torch.no_grad():
        base = model.base_features()
        table = model.knowledge_table(base)
        pred = model(valid, table, base)[0]
        swapped = valid.clone()
        swapped[:, :2] = valid[:, [1, 0]]
        swap = model(swapped, table, base)[0]
        chunked = torch.cat([model(valid[i:i+7], table, base)[0]
                             for i in range(0, len(valid), 7)])
        torch.testing.assert_close(swap, pred, rtol=0, atol=2e-6)
        torch.testing.assert_close(chunked, pred, rtol=0, atol=2e-6)
    manifest = read(ROOT / 'artifacts/dual_branch_gemini_cv5_seed0_20261119/frozen/manifest.json')
    assert all(sha(p) == h for p, h in manifest['source'].items())
    write(receipt, dict(
        status='passed', dataset=args.dataset, seed=0, fold=0, device='cpu',
        purpose='correctness only; not a performance experiment',
        original_global_graph=True, graph_summary=data['graph']['summary'],
        train_subset_rows=len(rows), validation_subset_rows=len(valid), updates=updates,
        base_initialization_matches_original=True, base_initialization_sha256=base_hash,
        original_optimizer_groups_preserved=True, optimizer_groups=audit,
        initial_prediction_max_error=initial_error,
        active_drug_exchange_max_error=float((swap-pred).abs().max()),
        active_query_chunk_max_error=float((chunked-pred).abs().max()),
        added_parameters=sum(p.numel() for p in model.prediction_transformer.parameters()),
        recipe=model.prediction_recipe(), all_downstream_trainable=True,
        old_checkpoint_loaded=False, test_evaluated=False,
        original_frozen_sources_unchanged=len(manifest['source']),
        source_sha256={str(ROOT / name): sha(ROOT / name) for name in [
            'dual_branch/transformer_prediction.py', 'scripts/check_transformer_prediction.py',
            'tests/test_transformer_prediction.py']},
        seconds=time.monotonic()-started))
    print('Passed:', receipt, flush=True)


if __name__ == '__main__':
    main()

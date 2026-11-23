"""CPU-only correctness check on the real full graph; not a performance run.

Run from the project root with ``python -m dual_branch.multimodal_checks``.
The few supervised updates verify connectivity and serialization. They do not
launch formal training, use held-out test labels, or submit any GPU jobs.
"""
import argparse
import hashlib
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], required=True)
    parser.add_argument('--result-directory', type=Path, required=True)
    args = parser.parse_args()
    from .common import ROOT, environment
    environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from global_cell.common import write, read, sha, author_imports
    directory = args.result_directory.resolve()
    if not str(directory).startswith(str(ROOT) + '/') or not str(directory).startswith('/ifs2/'):
        raise ValueError('New check outputs must be inside the IFS2 project')
    if directory.exists():
        raise FileExistsError('Preserve earlier checks; use a new output directory')
    directory.mkdir(parents=True)
    meta = dict(dataset=args.dataset, fold=0, seed=0, device='cpu', pid=os.getpid(),
                formal_training=False, test_evaluated=False, no_performance_claim=True)
    write(directory / 'progress.json', dict(meta, status='running', phase='loading'))
    try:
        import torch
        torch.set_num_threads(2)
        from .kpaths_data import load
        from .checks import balanced
        from .multimodal_fusion import MultimodalKPathsModel
        from .stable_training import train_epoch
        from .kpaths_checks import check_path_reading
        author_imports()
        from utils import set_seed_all
        # Construction can read fixed input caches, but may not initialize by
        # loading a checkpoint. Forbid both loading APIs inside the constructor.
        data = load(args.dataset, 0)
        set_seed_all(0)
        old_load, old_state_load = torch.load, torch.nn.Module.load_state_dict

        def forbidden(*unused, **kwargs):
            raise AssertionError('Constructor attempted a checkpoint load')

        torch.load = torch.nn.Module.load_state_dict = forbidden
        try:
            model = MultimodalKPathsModel(data)
        finally:
            torch.load, torch.nn.Module.load_state_dict = old_load, old_state_load
        assert all(p.requires_grad for p in model.parameters())
        assert not hasattr(model, 'prediction_transformer')
        assert not hasattr(model.fusion, 'drug_out')
        assert not hasattr(model, 'knowledge_fraction')
        initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        rows = balanced(data['raw']['fit']['rows'], 64)
        valid = balanced(data['raw']['valid']['rows'], 64)
        for part, key in [('fit', 'train'), ('valid', 'valid')]:
            actual = hashlib.sha256(data['raw'][part]['rows'].numpy().tobytes()).hexdigest()
            assert actual == data['raw']['metadata']['split'][key + '_array_sha256']
        write(directory / 'EXPERIMENT.json', dict(meta, recipe=model.prediction_recipe(),
            initialization_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
            graph=data['graph']['summary'], inputs=data['input_provenance'],
            supervised_rows=len(rows), validation_rows=len(valid), full_global_graph=True))
        groups = model.optimizer_param_groups()
        write(directory / 'OPTIMIZER_GROUPS.json', [dict(name=g['name'], lr=g['lr'],
            weight_decay=g['weight_decay'], parameters=sum(p.numel() for p in g['params'])) for g in groups])
        optimizer = torch.optim.AdamW(groups, eps=1e-6)
        for step in range(1, 4):
            loss, audit = train_epoch(model, optimizer, rows, rows[:, 3].float(), chunk=32)
            # No zero output blocks the graph: all components must receive a
            # real supervised gradient in the FIRST backward pass, too.
            assert all(value > 0 for value in audit['component_gradient_L2'].values()), audit
            write(directory / ('GRADIENT_step%d.json' % step), dict(step=step, train_BCE=loss, **audit))
            write(directory / 'progress.json', dict(meta, status='running', phase='correctness_updates',
                optimizer_updates=step))
            print(dict(dataset=args.dataset, check_update=step, gradients='all_nonzero'), flush=True)
        model.eval()
        path_checks = check_path_reading(model, valid)
        with torch.no_grad():
            base, table = model.base_features(), model.knowledge_table()
            prediction, detail = model(valid, table, base, capture=True)
            swapped = valid.clone()
            swapped[:, :2] = swapped[:, [1, 0]]
            swap = model(swapped, table, base)[0]
            chunked = torch.cat([model(valid[i:i+7], table, base)[0] for i in range(0, len(valid), 7)])
            torch.testing.assert_close(swap, prediction, rtol=0, atol=2e-6)
            torch.testing.assert_close(chunked, prediction, rtol=0, atol=2e-6)
            no_knowledge = model(valid, table, base, use_knowledge=False)[0]
            empty = model(valid, table, base, empty_evidence=True)[0]
            torch.testing.assert_close(empty, no_knowledge, rtol=0, atol=0)
            knowledge_effect = float((prediction - no_knowledge).abs().max())
            assert knowledge_effect > 1e-8
            no_3d = model.base_features(use_3d=False)
            torch.testing.assert_close(no_3d[model.num_drug:], base[model.num_drug:], rtol=0, atol=0)
            three_d_effect = float((no_3d[:model.num_drug] - base[:model.num_drug]).abs().max())
            assert three_d_effect > 1e-8
        checkpoint = directory / 'correctness_only.pt'
        torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        recipe=model.prediction_recipe(), correctness_only=True), checkpoint)
        restored = MultimodalKPathsModel(data).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
        with torch.no_grad():
            replay = restored(valid, restored.knowledge_table(), restored.base_features())[0]
        torch.testing.assert_close(replay, prediction, rtol=0, atol=2e-6)
        # Recheck every old frozen training source after our isolated addition.
        original = read(ROOT / 'artifacts/kpaths_cv5_seed0_20261120/frozen/manifest.json')['source']
        assert all(sha(path) == signature for path, signature in original.items())
        result = dict(meta, status='passed', full_global_graph=True, optimizer_updates=3,
            supervised_rows=len(rows), validation_rows=len(valid), old_checkpoint_loaded=False,
            all_components_first_backward_nonzero=True, old_training_sources_unchanged=len(original),
            path_reading=path_checks, swap_error=float((swap - prediction).abs().max()),
            query_chunk_error=float((chunked - prediction).abs().max()),
            empty_evidence_error=float((empty - no_knowledge).abs().max()),
            real_knowledge_prediction_effect=knowledge_effect, actual_3D_feature_effect=three_d_effect,
            no_3D_Cell_error=0., strict_fresh_instance_reload_error=float((replay - prediction).abs().max()),
            checkpoint_sha256=sha(checkpoint), recipe=model.prediction_recipe())
        write(directory / 'CHECKS.json', result)
        write(directory / 'progress.json', dict(meta, status='complete', optimizer_updates=3))
        print(dict(dataset=args.dataset, status='passed', report=str(directory / 'CHECKS.json')), flush=True)
    except Exception as error:
        write(directory / 'progress.json', dict(meta, status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

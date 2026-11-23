"""New mechanism model: correctness checks or explicitly requested fresh training.

No GPU job is submitted by this module. GPU invocations must come through the
existing shared queue with the global per-user process limit. All paths are IFS2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], required=True)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--stage', choices=['check', 'train'], default='check')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--result-directory', type=Path, required=True)
    parser.add_argument('--expected-manifest', type=Path)
    args = parser.parse_args()
    from .common import ROOT, environment
    environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device.split(':')[-1]
    from global_cell.common import read, write, sha, author_imports
    directory = args.result_directory.resolve()
    if not str(directory).startswith(str(ROOT) + '/') or not str(directory).startswith('/ifs2/'):
        raise ValueError('New results must reside inside the IFS2 project')
    if directory.exists():
        raise FileExistsError('Use a new output directory; preserve previous attempts')
    if args.stage == 'train' and not args.expected_manifest:
        raise ValueError('Formal training requires a frozen source/input manifest')
    directory.mkdir(parents=True)
    meta = dict(dataset=args.dataset, fold=args.fold, seed=0, stage=args.stage, physical_device=args.device,
                pid=os.getpid(), formal_training=args.stage == 'train', test_evaluated=False)
    write(directory / 'progress.json', dict(meta, status='running', phase='loading'))
    started = time.monotonic()
    try:
        import torch
        torch.set_num_threads(2)
        if args.expected_manifest:
            frozen = read(args.expected_manifest)
            for kind in ['source', 'inputs']:
                assert all(sha(p) == digest for p, digest in frozen[kind].items()), kind
        author_imports()
        from utils import set_seed_all, get_metrics, TruncatedExponentialLR
        from .mechanism_data import load
        from .mechanism_model import MechanismMultimodalModel
        from .mechanism_training import CheckpointSelection, check_mechanism_reading
        from .checks import balanced
        from .stable_training import train_epoch
        from global_cell.network import evaluate
        data = load(args.dataset, args.fold)
        coverage = {}
        for part in ['fit', 'valid']:
            ids = data['raw'][part]['rows'][:, :2].long().sort(dim=1).values
            pair = ids[:, 0] * data['raw']['num_drug'] + ids[:, 1]
            available = data['paths']['pair_mask'][pair].any(-1)
            coverage[part] = dict(rows=len(ids), rows_with_pair_paths=int(available.sum()),
                                  pair_path_percent=100 * float(available.float().mean()))
        write(directory / 'COVERAGE.json', dict(coverage, no_labels_used_for_retrieval=True,
            single_drug_root_context_read_even_without_pair_paths=True,
            missing_target_facts_not_fabricated=True))
        set_seed_all(0)
        original_load, original_state_load = torch.load, torch.nn.Module.load_state_dict

        def forbidden(*unused, **kwargs):
            raise AssertionError('Fresh model constructor attempted a checkpoint load')

        torch.load = torch.nn.Module.load_state_dict = forbidden
        try:
            model = MechanismMultimodalModel(data)
        finally:
            torch.load, torch.nn.Module.load_state_dict = original_load, original_state_load
        assert all(p.requires_grad for p in model.parameters())
        initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        model.to(torch.device('cpu' if args.device == 'cpu' else 'cuda:0'))
        groups = model.optimizer_param_groups()
        optimizer = torch.optim.AdamW(groups, eps=1e-6)
        scheduler = TruncatedExponentialLR(optimizer, .9997, min_lr=2e-6)
        recipe = dict(model.prediction_recipe(), epochs=3000, optimizer='AdamW', lr=1e-4,
            eps=1e-6, weight_decay=1e-4, propagation_gate_decay=0., clip_global_L2=1.,
            loss='original BCE only', one_full_fit_update_per_epoch=True, query_chunk=1024,
            evaluation='epochs1/2/3; every5 before1000; every epoch from1000; no early stopping',
            selected_auc='best AUROC from first evaluation',
            legacy_selected_auc='best AUROC among epochs>=1000, separate historical diagnostic')
        write(directory / 'EXPERIMENT.json', dict(meta, recipe=recipe, initialization_sha256=initial,
            parameters=sum(p.numel() for p in model.parameters()), inputs=data['input_provenance'],
            graph=data['graph']['summary'], selection_audit=model.selection_audit,
            source_manifest=str(args.expected_manifest)))
        write(directory / 'OPTIMIZER_GROUPS.json', [dict(name=g['name'], lr=g['lr'], weight_decay=g['weight_decay'],
            parameters=sum(p.numel() for p in g['params'])) for g in groups])
        device = next(model.parameters()).device
        rows, valid = (data['raw'][part]['rows'].to(device) for part in ['fit', 'valid'])
        if args.stage == 'check':
            rows, valid = balanced(rows, 64), balanced(valid, 64)
        selector = CheckpointSelection()
        epochs = 3 if args.stage == 'check' else 3000
        for epoch in range(1, epochs + 1):
            model.semantic.capture_propagation = epoch <= 3 or epoch % 100 == 0
            loss, diagnostic = train_epoch(model, optimizer, rows, rows[:, 3].float(), chunk=1024)
            scheduler.step()
            if epoch <= 3:
                assert all(value > 0 for value in diagnostic['component_gradient_L2'].values()), diagnostic
            write(directory / ('GRADIENT_epoch%d.json' % epoch), diagnostic)
            row = dict(epoch=epoch, train_BCE=loss, elapsed_seconds=time.monotonic() - started, **diagnostic)
            if epoch <= 3 or epoch % 5 == 0 or epoch >= 1000:
                prediction = evaluate(model, valid, chunk=1024)
                auc, ap = get_metrics(valid[:, 3].cpu().numpy(), prediction.numpy())
                row.update(valid_auc=float(auc), valid_aupr=float(ap))
                names = selector.consider(epoch, float(auc), float(ap))
                if names:
                    checkpoint = dict(model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                        recipe=recipe, epoch=epoch, initialization_sha256=initial,
                        input_provenance=data['input_provenance'], correctness_only=args.stage == 'check')
                    for name in names:
                        temp = directory / (name + '.pt.tmp')
                        torch.save(checkpoint, temp)
                        temp.replace(directory / (name + '.pt'))
                        write(directory / (name + '.json'), dict(epoch=epoch, valid_auc=float(auc),
                            valid_aupr=float(ap), checkpoint_sha256=sha(directory / (name + '.pt')),
                            correctness_only=args.stage == 'check'))
                    del checkpoint
            with (directory / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            write(directory / 'progress.json', dict(meta, status='running', phase='correctness_updates' if args.stage == 'check' else 'training',
                optimizer_updates=epoch, **row))
            if epoch <= 3 or epoch % 20 == 0:
                print(dict(dataset=args.dataset, epoch=epoch, stage=args.stage, train_BCE=loss), flush=True)
        if args.stage == 'check':
            result = check_mechanism_reading(model, valid)
            state_file = directory / 'correctness_final.pt'
            torch.save(dict(model_state=model.state_dict(), correctness_only=True), state_file)
            restored = MechanismMultimodalModel(data).to(device).eval()
            restored.load_state_dict(torch.load(state_file, map_location=device, weights_only=True)['model_state'], strict=True)
            a, b = evaluate(model, valid), evaluate(restored, valid)
            torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
            write(directory / 'CHECKS.json', dict(result, optimizer_updates=3, full_global_graph=True,
                supervised_rows=len(rows), validation_rows=len(valid), all_components_first_step_nonzero=True,
                strict_reload_error=float((a - b).abs().max()), all_downstream_trainable=True,
                old_checkpoint_loaded=False, no_performance_claim=True))
        write(directory / 'progress.json', dict(meta, status='complete', optimizer_updates=epochs))
    except Exception as error:
        write(directory / 'progress.json', dict(meta, status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

"""K-Paths evidence model: seed0 original CV1 folds, fresh end-to-end training.

Keeps the stability-repair optimizer, residual limits and selection rules.
Offline candidates are label-free; synergy hyperedges use only this fold's fit.
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
    parser.add_argument('--output')
    parser.add_argument('--device', required=True)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], default='ALMANAC')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--stage', choices=['check', 'train'], required=True)
    parser.add_argument('--check-rows', type=int, default=64)
    parser.add_argument('--result-directory', type=Path, required=True)
    parser.add_argument('--expected-manifest', type=Path)
    args = parser.parse_args()
    from .common import ROOT, environment
    from global_cell.common import read, write, sha, author_imports
    environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device.split(':')[-1]
    directory = args.result_directory.resolve()
    assert str(directory).startswith(str(ROOT) + '/')
    assert not (directory / 'progress.json').exists(), 'Keep all previous attempts'
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=args.dataset, fold=args.fold, seed=0, stage=args.stage,
        knowledge_mode='target_kpaths', pid=os.getpid(), physical_device=args.device)
    write(directory / 'progress.json', dict(meta, status='running', epoch=0, phase='importing_environment'))
    start = time.monotonic()
    try:
        import numpy as np
        import torch
        from .kpaths_data import load
        from .model import DualBranchModel
        from .kpaths_model import KPathsModel
        from .stable_training import train_epoch, residual_audit
        from .checks import balanced, verify_gradients
        from global_cell.network import evaluate
        torch.set_num_threads(2)
        if args.expected_manifest:
            manifest = read(args.expected_manifest)
            for kind in ['source', 'inputs']:
                assert all(sha(p) == expected for p, expected in manifest[kind].items()), kind
        author_imports()
        from utils import set_seed_all, get_metrics, TruncatedExponentialLR
        data = load(args.dataset, args.fold)
        for part, key in [('fit', 'train'), ('valid', 'valid')]:
            assert hashlib.sha256(data['raw'][part]['rows'].numpy().tobytes()).hexdigest() == data['raw']['metadata']['split'][key + '_array_sha256']
        set_seed_all(0)
        old_load, old_state_load = torch.load, torch.nn.Module.load_state_dict
        def forbidden(*unused, **kwargs):
            raise AssertionError('Fresh constructor attempted a checkpoint load')
        torch.load = torch.nn.Module.load_state_dict = forbidden
        try:
            model = KPathsModel(data)
        finally:
            torch.load, torch.nn.Module.load_state_dict = old_load, old_state_load
        assert all(p.requires_grad for p in model.parameters())
        initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        device = torch.device('cpu' if args.device == 'cpu' else 'cuda:0')
        model.to(device)
        groups = model.optimizer_param_groups()
        recipe = dict(version='target_kpaths_global_Cell_cv5_20261120', knowledge_mode='target_kpaths',
            seed=0, fold=args.fold, epochs=3000, all_downstream_trainable=True, old_checkpoint_loaded=False,
            Cell_inputs_and_encoder_unchanged=True, optimizer='AdamW (decoupled weight decay)',
            lr=1e-4, knowledge_lr=1e-4, eps=1e-6, weight_decay=1e-4,
            zero_decay='propagation gates and residual output projections', clip_global_L2=1.,
            query_chunk=1024, one_full_fit_update_per_epoch=True, loss='original BCE only',
            scheduler_gamma=.9997, min_lr=2e-6, early_stopping=False,
            knowledge_delta_norm_max_fraction=.25, Transformer_delta_norm_max_fraction=.10,
            combined_delta_norm_max_fraction=.35, bound_reference='same pre-knowledge modal token; detached scale',
            primary_comparable_selection='internal AUROC, epochs1000-3000, selected.pt',
            additional_diagnostic_selection='AUPRC from first evaluation, best_aupr.pt; not legacy-comparable selection',
            original_HERMES_Refiner=False, test_evaluated=False,
            path_reader='ordered entity/relation GRU + DrugPair x Cell gamma',
            main_knowledge='target-anchored shared protein/PPI/GO/Reactome paths only',
            global_Cell_and_fit_synergy=True, path_K=10, no_clinical_or_enzyme_path_candidates=True)
        write(directory / 'EXPERIMENT.json', dict(meta, recipe=recipe,
            initialization_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
            source_manifest=str(args.expected_manifest), inputs=data['input_provenance'],
            graph_summary=data['graph']['summary'], selection_audit=model.selection_audit))
        write(directory / 'KNOWLEDGE_SELECTION.json', model.selection_audit)
        write(directory / 'OPTIMIZER_GROUPS.json', dict(AdamW_decoupled=True, eps=1e-6,
            groups=[dict(name=g['name'], lr=g['lr'], weight_decay=g['weight_decay'],
                parameter_count=sum(p.numel() for p in g['params'])) for g in groups]))
        rows, valid = [data['raw'][part]['rows'].to(device) for part in ['fit', 'valid']]
        if args.stage == 'check':
            rows, valid = balanced(rows, args.check_rows), balanced(valid, min(args.check_rows, 128))
        model.eval()
        with torch.no_grad():
            base = model.base_features()
            table = model.knowledge_table(base)
            sample = valid[:16]
            ids = sample[:, :3].long().clone()
            ids[:, :2] = ids[:, :2].sort(dim=1).values
            actual, detail = model(sample, table, base, capture=True)
            expected = DualBranchModel.decode(model, base[ids])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert float(detail['delta'].abs().max()) == float(detail['transformer_delta'].abs().max()) == 0.
        del base, table
        optimizer = torch.optim.AdamW(groups, eps=1e-6)
        scheduler = TruncatedExponentialLR(optimizer, .9997, min_lr=2e-6)
        epochs = 3 if args.stage == 'check' else 3000
        best_auc, best_aupr, selected_epoch = -1., -1., None
        for epoch in range(1, epochs + 1):
            model.semantic.capture_propagation = epoch <= 3 or epoch % 100 == 0
            loss, diagnostic = train_epoch(model, optimizer, rows, rows[:, 3].float(), chunk=1024)
            scheduler.step()
            if epoch <= 3:
                verify_gradients(diagnostic['component_gradient_L2'], epoch)
            # Persist every epoch, including the period before a possible crash.
            write(directory / ('GRADIENT_epoch%d.json' % epoch), diagnostic)
            row = dict(epoch=epoch, train_BCE=loss, elapsed_seconds=time.monotonic() - start,
                lr=optimizer.param_groups[0]['lr'], **diagnostic)
            if epoch <= 3 or epoch % 20 == 0 or epoch >= 1000:
                prediction = evaluate(model, valid, chunk=1024)
                auc, ap = get_metrics(valid[:, 3].cpu().numpy(), prediction.numpy())
                row.update(valid_auc=float(auc), valid_aupr=float(ap))
                save_auc = (epoch >= 1000 or args.stage == 'check') and auc > best_auc
                save_ap = ap > best_aupr
                if save_auc or save_ap:
                    checkpoint = dict(model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                        recipe=recipe, epoch=epoch, initialization_sha256=initial, input_provenance=data['input_provenance'])
                    for save, name in [(save_auc, 'selected'), (save_ap, 'best_aupr')]:
                        if save:
                            temporary = directory / (name + '.pt.tmp')
                            torch.save(checkpoint, temporary)
                            temporary.replace(directory / (name + '.pt'))
                            write(directory / (name + '.json' if name != 'selected' else 'selection.json'),
                                dict(epoch=epoch, valid_auc=float(auc), valid_aupr=float(ap),
                                    checkpoint_sha256=sha(directory / (name + '.pt'))))
                    del checkpoint
                if save_auc:
                    best_auc, selected_epoch = float(auc), epoch
                if save_ap:
                    best_aupr = float(ap)
            if epoch <= 3 or epoch % 100 == 0:
                audit = residual_audit(model, valid[:128])
                projection = {key: float(module[0].weight.detach().square().mean().sqrt())
                    for key, module in model.semantic.type_projection.items()}
                write(directory / ('RESIDUAL_epoch%d.json' % epoch), dict(audit, semantic_projection_weight_RMS=projection,
                    propagation=model.semantic.propagation_audit))
            with (directory / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            write(directory / 'progress.json', dict(meta, status='running', phase='training',
                optimizer_updates=epoch, selected_epoch=selected_epoch, **row))
            if epoch <= 3 or epoch % 20 == 0:
                print(row, flush=True)
        if args.stage == 'check':
            model.eval()
            with torch.no_grad():
                base, table = model.base_features(), model.knowledge_table()
                pred = model(valid, table, base)[0]
                swapped = valid.clone()
                swapped[:, :2] = swapped[:, [1, 0]]
                swap = model(swapped, table, base)[0]
                chunks = torch.cat([model(valid[i:i+7], table, base)[0] for i in range(0, len(valid), 7)])
                torch.testing.assert_close(swap, pred, atol=2e-6, rtol=0)
                torch.testing.assert_close(chunks, pred, atol=2e-6, rtol=0)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                restored = KPathsModel(data).to(device).eval()
                restored.load_state_dict(state, strict=True)
                replay = evaluate(restored, valid, chunk=1024).to(device)
                torch.testing.assert_close(replay, pred, atol=2e-6, rtol=0)
            from .kpaths_checks import check_path_reading
            path_check = check_path_reading(model, valid)
            write(directory / 'PATH_READING_CHECKS.json', path_check)
            write(directory / 'CHECKS.json', dict(status='passed', path_reading=path_check, full_global_graph=True,
                knowledge_mode='target_kpaths', optimizer_updates=3, training_rows=len(rows),
                validation_rows=len(valid), initial_prediction_error=0., residual_caps_verified=True,
                nonzero_step2_step3_component_gradients_verified=True,
                active_drug_swap_error=float((swap-pred).abs().max()),
                active_query_chunk_error=float((chunks-pred).abs().max()),
                strict_fresh_instance_reload_error=float((replay-pred).abs().max()),
                all_downstream_trainable=True, old_checkpoint_loaded=False, test_evaluated=False,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
                no_performance_claim=True))
        write(directory / 'progress.json', dict(meta, status='complete', epoch=epochs,
            epochs=epochs, optimizer_updates=epochs, selected_epoch=selected_epoch, test_evaluated=False))
    except Exception as error:
        write(directory / 'progress.json', dict(meta, status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

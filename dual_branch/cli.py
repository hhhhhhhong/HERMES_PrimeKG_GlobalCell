"""Runnable correctness check / original-schedule fit for the dual-branch model."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback
from .common import ROOT, DISPATCH, DATA, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--device', required=True)
    parser.add_argument('--dataset', choices=['ALMANAC', 'ONEIL'], required=True)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--stage', choices=['check', 'train'], required=True)
    parser.add_argument('--result-directory', type=Path, required=True)
    parser.add_argument('--expected-manifest', type=Path)
    parser.add_argument('--check-rows', type=int, default=128,
        help='Correctness-only rows; use 1024 on GPU to match the training query chunk')
    args = parser.parse_args()
    assert args.check_rows >= 2 and args.check_rows % 2 == 0
    environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device.split(':')[-1]
    import json
    import numpy as np
    import torch
    from global_cell.common import read, write, sha, author_imports
    from global_cell.network import train_epoch, evaluate
    from .data import load
    from .model import DualBranchModel
    from .checks import before_updates, balanced, verify_gradients
    torch.set_num_threads(2)
    directory = args.result_directory.resolve()
    assert str(directory).startswith(str(ROOT) + '/')
    assert not (directory / 'progress.json').exists(), 'Keep previous attempts intact'
    if args.expected_manifest:
        expected = read(args.expected_manifest)
        assert all(sha(p) == h for p, h in expected['source'].items())
        assert all(sha(p) == h for p, h in expected['inputs'].items())
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=args.dataset, fold=args.fold, stage=args.stage, seed=0,
        pid=os.getpid(), physical_device=args.device)
    write(directory / 'progress.json', dict(meta, status='running', phase='preparing', epoch=0))
    start = time.monotonic()
    try:
        author_imports()
        from utils import set_seed_all, get_metrics, TruncatedExponentialLR
        data = load(args.dataset, args.fold)
        for part, key in [('fit', 'train'), ('valid', 'valid')]:
            assert hashlib.sha256(data['raw'][part]['rows'].numpy().tobytes()).hexdigest() == data['raw']['metadata']['split'][key + '_array_sha256']
        device = torch.device('cpu' if args.device == 'cpu' else 'cuda:0')
        set_seed_all(0)
        old_load, old_state_load = torch.load, torch.nn.Module.load_state_dict

        def forbidden(*unused, **kwargs):
            raise AssertionError('Fresh dual-branch construction attempted a checkpoint load')

        torch.load = torch.nn.Module.load_state_dict = forbidden
        try:
            model = DualBranchModel(data).to(device)
        finally:
            torch.load, torch.nn.Module.load_state_dict = old_load, old_state_load
        assert all(p.requires_grad for p in model.parameters())
        component_ids = [id(p) for group in model.components().values() for p in group]
        assert len(component_ids) == len(set(component_ids)) == len(list(model.parameters()))
        initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        parameter_groups = model.optimizer_param_groups()
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        group_audit = [dict(name=group['name'], lr=group['lr'], weight_decay=group['weight_decay'],
            parameter_count=sum(parameter.numel() for parameter in group['params']),
            tensor_names=[names[id(parameter)] for parameter in group['params']]) for group in parameter_groups]
        write(directory / 'OPTIMIZER_GROUPS.json', dict(groups=group_audit,
            all_trainable_parameters_covered_exactly_once=True, Adam_coupled_L2=True))
        recipe = dict(version='dual_branch_gemini_v2_20261119', all_nodes_CODER=True,
            graph_initialization_independent_of_modal_encoder=True, graph_layers=3,
            graph_propagation_gates_initial=.1, original_15_relation_graph=True,
            Drug='reused BRICS 2D + cached Uni-Mol 3D; learned 256D output',
            Cell='original HERMES expression MLP', conditional_gamma_query='modal DrugPair and Cell',
            fusion='role-specific zero output projection, Cell router without duplicate input, tanh(context RMS) scaling',
            fusion_initial_prediction='exact modal-only prediction in evaluation mode',
            fusion_not_guaranteed_equal_to_original_HERMES=True, original_HERMES_Refiner=False,
            seed=0, epochs=3000, early_stopping=False, select_from_epoch=1000,
            selection_metric='internal validation AUROC', loss='BCE only',
            optimizer='Adam with supplied Gemini parameter groups', lr=1e-3,
            knowledge_lr=2e-3, weight_decay=1e-4, zero_decay='propagation_gates and fusion_output',
            optimizer_groups=[{key: value for key, value in group.items() if key != 'tensor_names'} for group in group_audit],
            scheduler_gamma=.9997, min_lr=2e-6, query_chunk=1024,
            one_full_fit_update_per_epoch=True, all_downstream_trainable=True,
            pretrained_CODER_and_UniMol_are_fixed_inputs=True, old_checkpoint_loaded=False,
            original_test_not_opened=True, no_auxiliary_reconstruction_loss=True)
        write(directory / 'EXPERIMENT.json', dict(meta, recipe=recipe,
            initialization_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
            source_manifest=str(args.expected_manifest), inputs=data['input_provenance'],
            graph_summary=data['graph']['summary'], synergy_audit=data['graph']['synergy_audit']))
        rows, valid = [data['raw'][part]['rows'].to(device) for part in ['fit', 'valid']]
        if args.stage == 'check':
            rows, valid = balanced(rows, args.check_rows), balanced(valid, args.check_rows)
            write(directory / 'progress.json', dict(meta, status='running', phase='interventions', epoch=0))
            write(directory / 'INTERVENTIONS.json', before_updates(model, data, valid[:16]))
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()} if args.stage == 'check' else None
        optimizer = torch.optim.Adam(parameter_groups)
        scheduler = TruncatedExponentialLR(optimizer, .9997, min_lr=2e-6)
        epochs = 3 if args.stage == 'check' else 3000
        best = -1.
        selected = None
        for epoch in range(1, epochs + 1):
            model.semantic.capture_propagation = epoch <= 3 or epoch % 100 == 0
            tick = time.monotonic()
            loss, gradients = train_epoch(model, optimizer, rows, rows[:, 3].float(), chunk=1024)
            scheduler.step()
            assert np.isfinite(loss)
            if epoch <= 3:
                verify_gradients(gradients, epoch)
                write(directory / ('GRADIENT_epoch%d.json' % epoch), gradients)
            if model.semantic.capture_propagation:
                write(directory / ('PROPAGATION_epoch%d.json' % epoch), model.semantic.propagation_audit)
            row = dict(epoch=epoch, train_BCE=loss, train_seconds=time.monotonic() - tick,
                elapsed_seconds=time.monotonic() - start, lr=optimizer.param_groups[0]['lr'],
                group_learning_rates={group['name']: group['lr'] for group in optimizer.param_groups})
            if args.stage == 'check' or epoch <= 3 or epoch % 20 == 0 or epoch >= 1000:
                prediction = evaluate(model, valid, chunk=1024)
                auc, ap = get_metrics(valid[:, 3].cpu().numpy(), prediction.numpy())
                row.update(valid_auc=float(auc), valid_aupr=float(ap))
                if (epoch >= 1000 or args.stage == 'check') and auc > best:
                    best, selected = float(auc), epoch
                    checkpoint = dict(model_state={n: t.detach().cpu().clone() for n, t in model.state_dict().items()},
                        recipe=recipe, epoch=epoch, initialization_sha256=initial,
                        input_provenance=data['input_provenance'], old_checkpoint_loaded=False)
                    temporary = directory / 'selected.pt.tmp'
                    torch.save(checkpoint, temporary)
                    temporary.replace(directory / 'selected.pt')
                    write(directory / 'selection.json', dict(epoch=epoch, valid_auc=float(auc),
                        valid_aupr=float(ap), checkpoint_sha256=sha(directory / 'selected.pt')))
            with (directory / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            write(directory / 'progress.json', dict(meta, status='running', phase='training',
                optimizer_updates=epoch, selected_epoch=selected, **row))
            if epoch <= 3 or epoch % 20 == 0:
                print(row, flush=True)
        if args.stage == 'check':
            changes = {n: float((p.detach().cpu() - before[n]).abs().max()) for n, p in model.named_parameters()}
            component_changes = {}
            for name, parameters in model.components().items():
                ids = {id(p) for p in parameters}
                component_changes[name] = max(changes[n] for n, p in model.named_parameters() if id(p) in ids)
                assert component_changes[name] > 0, name
            # A newly constructed instance tests bound checkpoint closures as
            # well as serialization. Reusing the same live instance is weaker.
            checkpoint = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            expected_prediction = evaluate(model, valid, chunk=1024)
            restored = DualBranchModel(data).to(device)
            restored.load_state_dict(checkpoint['model_state'], strict=True)
            actual = evaluate(restored, valid, chunk=1024)
            torch.testing.assert_close(actual, expected_prediction, atol=2e-6, rtol=0)
            auc, ap = get_metrics(valid[:, 3].cpu().numpy(), actual.numpy())
            selection = read(directory / 'selection.json')
            assert abs(auc - selection['valid_auc']) < 1e-5 and abs(ap - selection['valid_aupr']) < 1e-5
            with torch.no_grad():
                base = model.base_features()
                table = model.knowledge_table(base)
                real, detail = model(valid, table, base, capture=True)
                modal_only, _ = model(valid, table, base, use_knowledge=False)
                ratio = {}
                for role, part in [('Drug', slice(0, 2)), ('Cell', slice(2, 3))]:
                    ratio[role] = float(detail['delta'][:, part].square().mean().sqrt() /
                        detail['own'][:, part].square().mean().sqrt().clamp_min(1e-12))
                write(directory / 'FUSION_AFTER_CHECK.json', dict(fixed_selected_checkpoint=True,
                    delta_to_own_rms=ratio, router_mean=float(detail['gates'].mean()),
                    knowledge_off_prediction_change_max=float((real - modal_only).abs().max()),
                    no_performance_claim=True))
            write(directory / 'CHECKS.json', dict(status='passed', full_global_graph=True,
                training_rows=len(rows), validation_rows=len(valid), optimizer_updates=3,
                graph_node_count=len(data['graph']['node_types']), all_nodes_CODER=True,
                all_components_supervised_gradient_nonzero_after_step3=True,
                initial_zero_fusion_and_staged_gradients_verified=True,
                actual_component_parameter_changes=component_changes,
                fresh_instance_checkpoint_reload_max_error=float((actual - expected_prediction).abs().max()),
                old_checkpoint_loaded=False, peak_cuda_memory_mib=(torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else None),
                peak_cuda_reserved_mib=(torch.cuda.max_memory_reserved() / 1024**2 if device.type == 'cuda' else None),
                requested_check_rows=args.check_rows,
                seconds=time.monotonic() - start, no_performance_claim=True))
        write(directory / 'metrics.json', dict(read(directory / 'selection.json'), epochs=epochs,
            test_evaluated=False, no_performance_claim=args.stage == 'check'))
        write(directory / 'progress.json', dict(meta, status='complete', epoch=epochs, epochs=epochs,
            optimizer_updates=epochs, selected_epoch=selected, test_evaluated=False))
    except Exception as error:
        write(directory / 'progress.json', dict(meta, status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

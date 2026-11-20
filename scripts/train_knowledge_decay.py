"""ALMANAC seed0/fold0 optimizer diagnostic using the frozen CV5 trainer.

Runtime adapters live only in this worker. The old trainer/model files and
their exact-zero identity checks are never edited or weakened.
"""
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--decay-mode', choices=['matched_decay', 'knowledge_no_decay'], default='knowledge_no_decay')
    options, remaining = parser.parse_known_args()
    original_argv = sys.argv[:]
    assert remaining[remaining.index('--dataset') + 1] == 'ALMANAC'
    assert remaining[remaining.index('--fold') + 1] == '0'
    directory = Path(remaining[remaining.index('--result-directory') + 1]).resolve()
    stage = remaining[remaining.index('--stage') + 1]
    assert stage == 'check' or options.decay_mode == 'knowledge_no_decay'
    # The original entry establishes CUDA visibility before importing torch.
    import os
    device_arg = remaining[remaining.index('--device') + 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if device_arg == 'cpu' else device_arg.split(':')[-1]
    import torch
    from global_cell import common
    from models import zero_gated_checkpoint
    from scripts.knowledge_decay import observed_adam_class, final_ablations, state_hash
    from scripts.train_zero_gated_hgnn_cv5 import main as original_main
    old_factory = zero_gated_checkpoint.CheckpointedZeroGatedModel
    old_adam, old_write = torch.optim.Adam, common.write
    context = {}
    reference = common.read(ROOT / 'runs/zero_gated_hgnn3_cv5_seed0_20261118/ALMANAC/fold0/EXPERIMENT.json')
    variant_sources = {str(p): common.sha(p) for p in [Path(__file__).resolve(), ROOT / 'scripts/knowledge_decay.py']}

    def factory(cache, graph, coder):
        model = old_factory(cache, graph, coder)
        context.update(model=model, raw=cache)
        assert state_hash(model) == reference['initialization_sha256']
        return model

    def adam(params, *args, **kwargs):
        return observed_adam_class(old_adam, context['model'], directory, options.decay_mode)(params, *args, **kwargs)

    def record(path, value):
        path = Path(path)
        if path.name == 'EXPERIMENT.json':
            recipe = value['recipe']
            recipe.update(optimization_diagnostic=True, decay_mode=options.decay_mode,
                weight_decay_scope='all parameters' if options.decay_mode == 'matched_decay' else 'initializer and prediction head only',
                knowledge_weight_decay=.01 if options.decay_mode == 'matched_decay' else 0.,
                original_CV5_training_entry=str(ROOT / 'scripts/train_zero_gated_hgnn_cv5.py'),
                architecture_changed=False, exact_zero_identity_required=True)
            value['variant_sources'].update(variant_sources)
            value['reference_checkpoint_not_loaded'] = True
        if path.name == 'progress.json' and value.get('status') == 'complete':
            context['completed_progress'] = value
            value = dict(value, status='running', phase='fixed_checkpoint_diagnostics')
        return old_write(path, value)

    try:
        zero_gated_checkpoint.CheckpointedZeroGatedModel = factory
        torch.optim.Adam = adam
        common.write = record
        sys.argv = [original_argv[0]] + remaining
        original_main()
        assert context.get('completed_progress')
        from global_cell.network import evaluate
        from utils import get_metrics
        final_ablations(context['model'], directory, context['raw'], evaluate, get_metrics)
        old_write(directory / 'progress.json', context['completed_progress'])
    except Exception as error:
        old_write(directory / 'progress.json', dict(status='failed', dataset='ALMANAC', fold=0,
            stage=stage, error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        zero_gated_checkpoint.CheckpointedZeroGatedModel = old_factory
        torch.optim.Adam, common.write = old_adam, old_write
        sys.argv = original_argv


if __name__ == '__main__':
    main()

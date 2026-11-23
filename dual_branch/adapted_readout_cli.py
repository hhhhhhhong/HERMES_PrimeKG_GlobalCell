"""Reuse the unchanged full-fit trainer with an entity-adapted model factory.

This process-local substitution does not edit the original CLI/module files
or affect other Python workers. GPU runs still require the shared queue.
"""

def main():
    from .common import environment
    environment()
    # Resolve CUDA visibility BEFORE importing the model or torch.
    import os
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()
    device = args.device
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device.split(':')[-1]
    from .entity_token_adapter import EntityAdaptedHypothesisModel
    from . import hypothesis_evidence, hypothesis_cli
    original = hypothesis_evidence.HypothesisEvidenceModel
    try:
        hypothesis_evidence.HypothesisEvidenceModel = EntityAdaptedHypothesisModel
        hypothesis_cli.main()
    finally:
        hypothesis_evidence.HypothesisEvidenceModel = original

if __name__ == '__main__':
    main()

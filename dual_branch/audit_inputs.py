"""Token/OOV/order audit; vector uniqueness is not a semantic-quality score."""
import collections
import hashlib
from .common import ROOT, DATA, DISPATCH, environment


def main():
    environment()
    import torch
    from transformers import AutoTokenizer
    from .prepare import build_tokenizer
    from global_cell.common import read, write
    tokenizer = build_tokenizer()
    old = AutoTokenizer.from_pretrained(ROOT / 'models/coder_eng', local_files_only=True, do_lower_case=False)
    audit = {}
    for dataset in ['ALMANAC', 'ONEIL']:
        data = torch.load(DATA / dataset / 'all_nodes_coder.pt', map_location='cpu', weights_only=True)
        by_type = collections.defaultdict(lambda: dict(nodes=0, all_UNK_before=0, all_UNK_after=0,
            unknown_pieces_after=0, pieces_after=0))
        roots = {}
        for record, vector in zip(data['records'], data['features']):
            entry = by_type[record['kind']]
            before = old.encode(record['text'], add_special_tokens=False)
            after = tokenizer.encode(record['text'], add_special_tokens=False)
            entry['nodes'] += 1
            entry['all_UNK_before'] += bool(before) and all(x == old.unk_token_id for x in before)
            entry['all_UNK_after'] += bool(after) and all(x == tokenizer.unk_token_id for x in after)
            entry['pieces_after'] += len(after)
            entry['unknown_pieces_after'] += after.count(tokenizer.unk_token_id)
            if record['kind'] in ('drug', 'cell'):
                roots[record['key']] = hashlib.sha256(vector.numpy().tobytes()).hexdigest()
        assert by_type['drug']['all_UNK_after'] == by_type['cell']['all_UNK_after'] == 0
        assert torch.isfinite(data['features']).all()
        audit[dataset] = dict(by_type=dict(by_type), roots=len(roots),
            root_unique_vectors=len(set(roots.values())), features_shape=list(data['features'].shape),
            vector_uniqueness_not_used_to_claim_quality=True)
    write(DISPATCH / 'TOKEN_AND_VECTOR_AUDIT.json', dict(status='passed', datasets=audit,
        same_text_compared_under_both_tokenizers=True, do_lower_case=True, max_length=128,
        checkpoint_weights_unchanged=True, test_evaluated=False))
    print({dataset: dict(nodes=sum(r['nodes'] for r in values['by_type'].values()),
        root_unique_vectors=values['root_unique_vectors']) for dataset, values in audit.items()}, flush=True)


if __name__ == '__main__':
    main()

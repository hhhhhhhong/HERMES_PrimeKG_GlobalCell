"""CPU audit of the actual installed HypergraphConv and masking edge cases."""
import hashlib
import inspect
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_branch.common import environment
environment()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from global_cell.common import write, sha


def main():
    import torch
    from torch_geometric.nn import HypergraphConv
    torch.set_num_threads(2)
    source = Path(inspect.getsourcefile(HypergraphConv))
    incidence = torch.tensor([[0, 1, 2, 0, 3, 4], [0, 0, 0, 1, 1, 1]])
    x = torch.ones(5, 1)
    conv = HypergraphConv(1, 1, bias=False).eval()
    with torch.no_grad():
        conv.lin.weight.fill_(1.)
        full = conv(x, incidence, torch.ones(2))
        selected = incidence[:, :3]
        thinned = conv(x, selected, torch.ones(1))
        rescaled = conv(x, selected, torch.full((1,), 1.25))
    torch.testing.assert_close(full, torch.ones_like(full), rtol=0, atol=0)
    torch.testing.assert_close(thinned[:3], torch.ones(3, 1), rtol=0, atol=0)
    torch.testing.assert_close(thinned[3:], torch.zeros(2, 1), rtol=0, atol=0)
    weight_scale_ratio = float(rescaled[0] / thinned[0])
    # This is an observation of the pinned installed implementation, not a
    # general mathematical assertion about every HypergraphConv version.
    literal_results = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        key = str(dtype)
        value = torch.tensor([[1., 2., 3.]], dtype=dtype, requires_grad=True)
        available = torch.tensor([[True, False, True]])
        probability = value.masked_fill(~available, -torch.inf).softmax(-1)
        (probability.float() * torch.tensor([1., 2., 3.])).sum().backward()
        row = dict(valid_inf_softmax_forward_finite=bool(torch.isfinite(probability).all()),
                   valid_inf_softmax_gradient_finite=bool(torch.isfinite(value.grad).all()),
                   unguarded_all_inf_softmax_finite=bool(torch.isfinite(torch.full_like(value, -torch.inf).softmax(-1)).all()))
        try:
            value.detach().masked_fill(~available, -1e9)
            row['literal_minus_1e9_masked_fill'] = 'succeeded'
        except RuntimeError as error:
            row['literal_minus_1e9_masked_fill'] = repr(error)
        literal_results[key] = row
    forward_source = inspect.getsource(HypergraphConv.forward)
    message_source = inspect.getsource(HypergraphConv.message)
    directory = ROOT / 'artifacts/numerical_safety_20261122'
    (directory / 'INSTALLED_HYPERGRAPHCONV.py.txt').write_text(forward_source + '\n' + message_source)
    result = dict(status='passed', torch_version=torch.__version__,
        installed_HypergraphConv_source=str(source), installed_source_sha256=sha(source),
        installed_forward_sha256=hashlib.sha256(forward_source.encode()).hexdigest(),
        degree_normalization_observed=True, full_unit_features_output=full.flatten().tolist(),
        thinned_connected_output=thinned[:3].flatten().tolist(),
        thinned_isolated_output=thinned[3:].flatten().tolist(),
        weight_125_connected_message_ratio=weight_scale_ratio,
        expectation_note='Graph Dropout is not exactly unbiased: a degree-d node is isolated with probability 0.2**d; nonlinear gates and BatchNorm also matter.',
        conclusion='Keep unit weights for structurally thinned Refiner; do not blindly multiply hyperedge weights by 1/0.8.',
        masking=literal_results, CPU_only=True, GPU_jobs_submitted=0, no_performance_claim=True)
    write(directory / 'SYNERGY_NORMALIZATION_AUDIT.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()

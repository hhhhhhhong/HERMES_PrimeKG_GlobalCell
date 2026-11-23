# Masked softmax and Synergy Dropout audit

This is a numerical-only implementation for subsequent experiments, separate
from the frozen gated fivefold suite and the previous Cell-retrieval snapshot.
The existing experiment sources, cached inputs, jobs and results are retained.

## Masked attention

The original fusion already rejects an entity whose entire source mask is
False. Its valid-input `-inf` softmax forward and backward were finite in local
CPU fp16/bf16/fp32 tests. Thus `-inf` alone is not evidence of an existing loss
spike, nor has a silent NaN been demonstrated in the active FP32 training.
An unguarded all-`-inf` row would, however, produce NaNs.

The suggested literal `-1e9` is not representable in fp16: the local
`masked_fill` raises `value cannot be converted to type at::Half without
overflow`. Also `1e-12` underflows when the normalization denominator is fp16.
The old gamma uses these literals and needs the same defensive change.

`dual_branch/numerical_safety.py::finite_masked_softmax` now:

1. Requires boolean masks, validates available logits, and broadcasts masks.
2. Promotes fp16/bf16 logits to FP32 for the entire normalization.
3. Uses the working dtype's finite minimum for unavailable logits.
4. Computes softmax, masks exact zeros, and renormalizes with a representable
   dtype-aware positive floor before casting back.
5. Returns exact zero weights and finite zero gradients for fully masked rows.
   Fusion still explicitly rejects entities with no available input source;
   gamma continues accepting truly empty evidence and returns zero knowledge.

The actual implementations are `SafeQueryGatedFusion` and
`SafeConditionalGamma`. `SafeGatedMechanismModel` and
`SafeCellConditionedMechanismModel` install these without changing state keys,
parameters, initialization tensors or constructor RNG. They copy only the
freshly constructed module state, without loading any prior checkpoint.

Seven unit checks cover all-masked/single-source/extreme inputs, unavailable
weights/gradients, nonfinite input handling, dtype fp16/bf16/fp32/fp64,
fusion/gamma forward and backward, CPU bf16 autocast, both gamma interfaces,
legacy numerical parity, and constructor RNG preservation. CUDA AMP was not
tested; no GPU job was submitted for this audit.

## Two distinct Dropout operations

The unified graph still includes `Synergy_Hyperedge`. Its alpha/beta Dropout is
active, not dead code. Each of the three propagation rounds samples an edge
mask **after beta normalization**, scales retained messages by `1/0.8`, and
leaves the complete incidence groups intact. Conditional on those messages,
this inverted message mask preserves their first-moment expectation.

The parallel author Refiner takes `initial[:nroots]` and the original full
`model.synergy_incidence`. It does not consume the thinned alpha/beta output or
an incidence tensor modified by alpha/beta. It independently deletes whole
edges, compresses edge IDs and recomputes HypergraphConv degrees. Thus the
Refiner does not receive incidences with an effective `0.8 * 0.8` keep rate.
These are separate regularization choices; their predictive effect has not
been isolated, and neither operation is removed in this numerical patch.

`scripts/check_numerical_safety.py` traces a real complete-graph train forward:
three alpha/beta masks of shape `[E,1,1]`, one Refiner mask of shape `[E]`,
and a check that the Refiner receives the complete original incidences and
returns unit weights on its retained graph. It also checks all components'
finite nonzero supervised gradients in two CPU correctness updates.

## Refiner normalization

The actual author module is in `vendor/hermes/Model/HERMES/model.py`. Each of
its three layers uses `torch_geometric.nn.HypergraphConv` with BatchNorm,
ReLU and learned residual gates. The installed HypergraphConv explicitly
computes inverse node degree and inverse hyperedge size for propagation.
It is not unnormalized neighbor summation.

`scripts/audit_synergy_normalization.py` saves the installed implementation
source and SHA, then tests two 3-node hyperedges sharing one node:

- Constant unit inputs and unit projection give output 1 at connected nodes
  with both edges, and still output 1 at connected nodes after one edge is
  removed. Nodes with no remaining incidence receive zero convolution output.
- In this installed implementation, changing the retained hyperedge weight
  from 1 to 1.25 changes the no-bias convolution message to approximately 0.8.
  This observation concerns the pinned implementation; other versions may
  treat weights differently. It demonstrates that this parameter is not a
  standalone inverted message-Dropout multiplier.

Consequently Refiner's unit edge weights are retained. **No blind `1/0.8`
compensation is applied.** Structural Dropout can still change neighborhoods,
isolation probability (`0.2**degree` under independent masks), gates and
BatchNorm behavior. Degree normalization does not establish that the entire
nonlinear Refiner is an unbiased estimator of its evaluation-time output.

## Use and evidence

The latest Cell-conditioned implementation is constructed as:

```python
from dual_branch.cell_conditioned_data import load
from dual_branch.numerical_safety import SafeCellConditionedMechanismModel

data = load('ALMANAC', fold=0)
model = SafeCellConditionedMechanismModel(data, condition_on_cell=True)
```

Within the normal project environment and author imports, both true/false Cell
selection modes have the same neural capacity. This change submits no formal
training or fivefold repetition. The active gated suite retains its original
frozen recipe and implementation; its results cannot be attributed to this fix.

Evidence is under `artifacts/numerical_safety_20261122`: unit logs,
`SYNERGY_NORMALIZATION_AUDIT.json`, complete-graph CPU checks and `VERIFIED.json`.
All new files, temporary files, caches and correctness checkpoints are on IFS2.

## Verified complete-graph CPU results

Both ALMANAC and ONEIL checks passed with 64 fit and 64 validation rows and two
supervised correctness updates. All parameter groups had finite nonzero task
gradients. Initial state tensors and RNG matched the old model exactly; the
maximum initial prediction difference was `1.7881393432617188e-7` in each
dataset. Strict disk checkpoint reload error was zero. Constructor checks
forbidding all checkpoint/state loading passed.

The real train-forward Dropout trace observed three alpha/beta message masks
and one independent Refiner structural mask in each dataset. The Refiner
received all 6515 / 1201 original ALMANAC / ONEIL incidences by edge count and
retained 5236 / 958 edges in that particular draw, with unit weights. These
counts verify execution, not the predictive quality of either regularizer.

# Held-out collaborative hypotheses guide biological evidence reading

This version implements two requested changes together: exact exclusion of
the current target Synergy edges, and a Refiner-to-gamma Query connection.
It builds on the Cell-conditioned candidate selection and numerical safety
versions. All existing experiment sources/results remain immutable.

## Exact target exclusion

For every training query batch, the model canonicalizes `(Drug A, Drug B, Cell)`
using only entity columns. It removes **every matching positive training edge**,
including duplicate and reversed drug-order instances, from both:

- the Synergy relation in the unified global alpha/beta propagation graph;
- the incidence tensor passed to the original author Refiner.

The current query label is not consulted. Thus a negative query with an
overlapping positive training edge is also protected. Other Cells for the same
drug pair remain distinct queries. Validation/held-out labels never build edges.

Incidences are physically removed before message passing. Degrees, alpha/beta
softmax denominators, active role slots, ordered relation combinations and
root routes are recomputed. Relation descriptors map to the original fixed
trainable token schema; target-dependent stale route masks are not reused.
If no Synergy edge remains, that relation family is absent from propagation,
and Refiner receives a genuine empty incidence tensor without a dummy edge.

The usual 20% Refiner structural Dropout still applies to remaining edges.
Increasing Dropout to 60–70% is not used as a substitute: random deletion cannot
guarantee that the current target edge is absent. No `1/0.8` Refiner edge-weight
compensation is added to the degree-normalized convolution.

`HypothesisEvidenceModel.exclusion_context(rows)` restores the original graph
buffers after use, including on exceptions. Nested contexts, queries outside
the current scope, unmasked table construction and stale/copied tables are
rejected. **Backward must complete inside the context**, because activation
checkpoint recomputation must see the same held-out relation buffers.

## Collaborative representation conditions gamma

The gamma Query receives symmetric features from both modalities and the
held-out Refiner output:

```text
q_input = concat(
    mA + mB, abs(mA - mB), mCell,
    cA + cB, abs(cA - cB), cCell
)
q = MLP(q_input)                         # 1536 -> 256 -> 256
```

Here `c` is non-affine LayerNorm of Refiner output. Drug roles/parameters remain
shared and permutation symmetric. Refiner features are **not detached**: the
evidence-reading loss can update the collaborative representation through Query.

The later three-source gate, Transformer and original prediction head remain.
Their inputs now have a directed dependency, `collaboration -> evidence Query
-> knowledge representation`, rather than being independent until fusion.
`collaborative_query=False` zeros only the collaborative Query terms, retaining
the same neural parameters/candidate budget for a possible matched comparison.

This is **hypothesis-guided evidence reading**, not biological verification.
The upstream candidate path pool is still selected from real paths with Cell
relative-expression support; Refiner does not initiate a new graph search.
The global knowledge propagation also contains other remaining training Synergy
edges, so it is not a statistically isolated pure-external-knowledge branch.
Attention/gate weights do not prove that a mechanism is true, and masking cannot
guarantee better AUPRC or force higher biological-branch weights.

## Training and evaluation semantics

`dual_branch/hypothesis_training.py::train_epoch` keeps one optimizer update per
epoch. It caches only pre-propagation molecular/Cell features and initial CODER
projections. Each query chunk has a new excluded graph and evidence table, and
its BCE is scaled by the full epoch row count before backward. Gradients to the
cached input leaves are accumulated and then propagated through their encoders
once. Graph propagation/Refiner/Query/decoder gradients accumulate normally.

The old `stable_training.train_epoch` is incompatible: it builds one full graph
table before processing target batches. The new model refuses that invocation.

Evaluation constructs one fixed graph excluding **all evaluated query keys**,
without reading evaluation labels, then chunks only neural readout. This also
protects overlapping training queries and makes evaluation chunk size irrelevant
to graph construction. Validation/test rows never supply positive graph edges.
No sealed test is accessed by this implementation/check procedure.

Costs and interpretation:

- With query chunk 1024, the existing fit sizes require 54 graph tables per
  ALMANAC epoch and 14 per ONEIL epoch, versus one table in the old trainer.
  A fresh graph topology is compiled per chunk. Old epoch-time estimates cannot
  be reused; this correctness-first implementation needs GPU profiling before
  any formal fivefold budget is claimed.
- Only selected paths are GRU-encoded for each context; all global Drug/Cell/
  PrimeKG propagation is retained. There is no graph-size or batch reduction.
- Refiner BatchNorm sees each chunk-specific graph rather than one graph per
  epoch. This and graph Dropout change stochastic training behavior; exact
  exclusion does not make the whole nonlinear network expectation invariant.
- Predictions under training graph exclusion can depend on which *other target
  edges* are removed in the same batch. That is part of the declared training
  protocol. Evaluation uses a fixed full-query exclusion set.
- BCE, AdamW learning rates/decay, clipping and all downstream trainability are
  retained. No prior synergy checkpoint, new encoder, auxiliary loss or RL is
  introduced. Current running fivefold results belong to the older gated model.

## Files and CPU checks

- `dual_branch/query_exclusion.py`: canonical matching and topology rebuilding.
- `dual_branch/hypothesis_evidence.py`: model, scoped graph and new gamma Query.
- `dual_branch/hypothesis_training.py`: required exclusion-aware training/eval.
- `tests/test_hypothesis_evidence.py`: six synthetic mechanism checks, including
  duplicate/reversed queries, no remaining edges, biological gradients,
  collaborative Query gradients, exchange symmetry, empty evidence and fp16/bf16.
- `scripts/check_hypothesis_evidence.py`: two datasets, seed0/fold0, full real
  graph, 64 fit/64 validation rows, two CPU correctness updates with two target
  batches per update. Checks physically absent target-edge oracles, arbitrary
  target weights/duplicates, stale-table rejection, graph restoration,
  evidence-context-only gradients to Refiner, optimizer step count and reload.

CPU entry, from the project directory with its py38 environment:

```bash
python scripts/check_hypothesis_evidence.py --dataset ALMANAC
python scripts/check_hypothesis_evidence.py --dataset ONEIL
```

Checks, frozen sources and receipts are under
`artifacts/hypothesis_evidence_20261122`. No formal training/GPU job is submitted
by these scripts. All new files, logs, temporary files and checkpoints use IFS2.

## Verification and data-overlap limits

Both complete-graph CPU checks passed. Predicted probabilities were bitwise
identical to a source graph physically missing the held-out target edges, and
also identical when those forbidden edges had weight 100000 and were duplicated.
The baseline and oracle use identical no-grad execution, retaining exact zero
tolerance. Actual evidence-context-only loss produced nonzero Refiner gradients
through the new collaborative Query. All component groups had finite nonzero
task gradients in two updates, strict checkpoint reload error was zero, and
evaluation chunk errors were below 2.4e-7.

The CPU checks use 64 supervised rows and two 32-row graph contexts, not full
training or a performance estimate. Two-context epoch times were about 20–22s
for ALMANAC and 9–10s for ONEIL on CPU; they are not GPU/full-epoch estimates.
Failure records before successful checks are retained: the first diagnostic
mistakenly selected the negative prefix for a positive-edge oracle; the second
compared grad-enabled and no-grad inference fastpaths. Both were corrected in
the checker without weakening the physical-exclusion oracle tolerance.

The original fold0 has 10 / 30 extra positive edge instances from repeated
ALMANAC / ONEIL query keys. Furthermore, 38 / 30 validation records match a
positive training query, and 357 / 73 validation records match *some* training
query. `scripts/audit_hypothesis_overlap.py` audits all original fivefold inputs
without using validation labels; its output is `TARGET_OVERLAP_AUDIT.json`.

The new evaluation removes all matching graph edges, but **does not erase
supervised duplicate-query overlap** from model fitting. Strict generalization
without that overlap requires a separately declared split grouped by canonical
Drug–Drug–Cell query (or a stricter pair/drug/Cell split). Original splits remain
unchanged for historical compatibility; the new code must not be described as
having eliminated all data leakage or established biological mechanisms.

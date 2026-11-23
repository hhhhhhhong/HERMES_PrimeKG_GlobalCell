# Cell-conditioned candidate evidence

This version moves Cell conditioning into **path membership selection before
gamma**, using the existing HERMES expression data and witnessed PrimeKG facts.
The implementation is separate from the currently running gated fivefold suite.

## Computation

1. Enumerate a wider, label-free pool for each unordered drug pair: shared
   target, target–PPI–target, target–GO/Reactome–target, and a PPI route with one
   intermediate protein. Every relation retains source row, direction and role.
2. Retain at most 64 candidate paths with up to eight entity instances per
   ordered relation sequence. Enumeration is bounded at 4096 candidates per
   route generator (GO and Reactome share the function generator), and 128
   degree-ranked PPI bridge proteins. Functional annotations use original
   member sets of at most 200 proteins. All truncation is reported. This is
   neither exhaustive retrieval nor a literal reproduction of K-Paths.
3. Rank each mapped gene's expression **across the fixed, unlabeled Cell panel**,
   separately for each gene. Average tied ranks and map to [-1,1]. This follows
   the existing transductive feature protocol; it is not an unseen-Cell protocol.
   A constant gene has support zero. The original expression matrix is not
   transformed in place, and its raw sign is never an on/off activity indicator.
4. Score a candidate path for Cell c as

   `score(path,c) = -heuristic_cost(path) + relative_expression_support(path,c)`

   where `heuristic_cost = hops + 0.1 * sum(log1p(internal node degree))` and
   support is the sum of measured proteins' relative expression divided by
   **all** proteins on the path. Unmapped proteins contribute zero (unknown).
   The coefficient is fixed at one without examining synergy outcomes.
5. Select K <= 10 via deterministic family round-robin, allowing at most two
   instances per ordered relation sequence. Missing expression never creates
   a fabricated measurement or triggers hard path deletion. No connecting path
   remains an explicitly masked empty path set.
6. `(Drug A, Drug B, Cell)` indexes that selected set **before** gamma. Gamma
   then learns to read the selected evidence. Original root biomedical channels,
   the author Refiner, three-source gate, Transformer and prediction head remain
   part of the model.

## Scope and interpretation

- This is **deterministic expression-guided retrieval**, not a learned policy,
  an RL module, a causal mechanism detector, or evidence of protein activity.
- The unified global graph contains the union of retained candidate facts plus
  existing Cell expression edges. Alpha/beta still propagates on this shared
  graph; a separate global propagation graph is **not** built for each query.
  The subset of path evidence supplied to gamma changes with the Cell.
- Only already encoded nodes are admitted. New source-backed PPI or functional
  memberships between those nodes may enter the new cache; no new encoder is
  invoked. Excluded candidates are counted.
- Fit-positive Synergy edges are added per fold by the existing loader. Neither
  candidate generation nor selection opens a supervised fold or uses a label.
  Knowledge propagation still shares fit-Synergy information, as in the current
  gated model; this is not an isolated pure-external-knowledge experiment.
- Different selections are evidence that the code is conditional, not evidence
  of better predictions. Expression coverage and retrieval truncation limit the
  biological interpretation; connected paths do not establish synergistic MoA.

## Matched switch

`CellConditionedMechanismModel(data, condition_on_cell=False)` uses the same
expanded candidate pool, unified graph, neural parameters and Top-K budget,
with the expression term removed. `True` uses Cell-guided membership.
There are no additional neural parameters or auxiliary losses. The control's
Cell representation still conditions gamma. The switch isolates **candidate
selection**, not every possible use of Cell information.

The already running fivefold suite uses an older, narrower candidate pool and
graph, so a future comparison to that suite alone would not isolate Cell
conditioning. This change does not start an additional formal experiment.

## Files and CPU checks

- `dual_branch/cell_conditioned_retrieval.py`: witnessed candidate expansion,
  expression support and deterministic diverse selection.
- `dual_branch/cell_conditioned_data.py`: independent cache and original splits.
- `dual_branch/cell_conditioned_model.py`: query-specific path lookup and model.
- `scripts/prepare_cell_conditioned_retrieval.py`: CPU-only preparation, refuses
  overwrites, records pool proof and selection audits.
- `scripts/check_cell_conditioned_retrieval.py`: full real graph, 64 fit/64
  validation rows, three CPU correctness updates, gradients, drug exchange,
  query batch isolation, fixed-query path interventions and strict reload.
- `tests/test_cell_conditioned_retrieval.py`: ranking, unknowns, source witnesses,
  drug symmetry, actual membership changes and explicit enumeration budgets.

From `/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell`, using the project's
`/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python`:

```bash
python scripts/prepare_cell_conditioned_retrieval.py --dataset ALMANAC
python scripts/prepare_cell_conditioned_retrieval.py --dataset ONEIL
python scripts/check_cell_conditioned_retrieval.py --dataset ALMANAC
python scripts/check_cell_conditioned_retrieval.py --dataset ONEIL
```

New caches: `data/cell_conditioned_retrieval_20261122/{dataset}`.
Checks/audits: `artifacts/cell_conditioned_retrieval_20261122`.
Everything is on IFS2. Formal GPU training would require new frozen sources,
GPU checks and the existing global resource-limited queue; none is submitted
by preparation or correctness scripts.

## Offline audit, 2026-11-22

| Dataset | Pairs with paths | Expanded pool paths | Pairs whose selected membership varies across Cells |
| --- | ---: | ---: | ---: |
| ALMANAC | 2094 / 3741 | 22024 | 1453 / 2094 (69.39%) |
| ONEIL | 351 / 703 | 4514 | 283 / 351 (80.63%) |

These percentages use **pairs with paths** as the denominator, not all dataset
samples. Drug-pair connectivity coverage did not increase in this change.
Five deterministic retrieval tests passed. All 26538 paths were independently
checked against the immutable, previously raw-CSV-audited fact index: every
endpoint, source row, inverse relation, role, graph membership and packed
encoder sequence passed. This did not rescan the full original PrimeKG CSV.

The retained pools include at least one expression-mapped protein on 11424 /
22024 ALMANAC paths and 2953 / 4514 ONEIL paths. Each dataset uses 642 uniquely
mapped genes; unmeasured proteins remain unknown. The original 121 ALMANAC and
30 ONEIL pairs affected by the PPI bridge enumeration budget are still flagged.
Pool diversity/size constraints also discard candidate variants and are reported
in `SELECTION_AUDIT.json`; no exhaustive best-path claim is made.

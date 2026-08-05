# Report EXP-013: Late Whole-State EMA
- **Created**: 2026-07-24

## Goal
Raise accepted CIFAR-10 `best_test_acc` 94.07% to at least 94.17% within 300 counted seconds.

## Idea & Hypothesis
From the 65% hard-label boundary, maintain decay-0.999 EMA of all parameters and BN state and evaluate only EMA thereafter. The hypothesis predicted at least 95% throughput, 134.8 passes, and 94.17% accuracy.

## Approach
Implemented an external unregistered FP32 shadow dict. First eligible post-SGD state was copied directly; later floating tensors used EMA and integral counters copied. Evaluation swapped state into existing objects and restored live state in `finally`. Architecture, live optimizer/training, seed, data, and cadence were unchanged.

## Execution
Fail-closed semantics verified key coverage, object/optimizer integrity, complete live equality, and restoration after partial swap failure. Preflight retained 99.05% throughput and projected 140.55 passes. One H20 run completed 27,533 steps / 142 epochs in 300.0 counted / 340.5 total seconds; EMA initialized once at step 17,738 and made exactly 9,795 updates.

## Results
- **Primary metric**: 94.10% (baseline 94.07%, delta +0.03 points, +0.03%)
- **Observations**: 140.96896 passes preserved 99.34% of accepted exposure. EMA peaked at epoch 115 with 94.10% / 0.2196 loss, then declined to 93.79% / 0.2596 terminal. Epoch 110 loss was 0.2065 at 94.06%, lower than accepted loss without improving top-1. Peak VRAM was 1,094.0 MiB.
- **Analysis**: EMA worked operationally and improved probabilistic loss/calibration at mid-tail evaluations, but did not move enough classifications across the top-1 boundary. Later EMA states degraded steadily, consistent with trajectory lag or approximate averaged BN moments rather than missing exposure. This rejects exact 65%-start, 0.999 whole-state EMA as a standalone accuracy improvement, not all averaging windows.
- **Key Learning**: Whole-state EMA improves mid-tail loss but only gains 0.03 accuracy points; this window lags useful terminal top-1 refinement.

## Verification
- **Conditions**: process integrity passed; 94.17% metric condition failed.
- **Review Notes**: Trustworthy fixed-seed, one-run, one-H20 result with exact state semantics, 140.97 passes, one init/transition, correct update count/cadence, and `train.py`-only source diff.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid best 94.10% misses the required margin by 0.07; no rerun.

## Unexplored Avenues
- Parameter-only averaging with recalibrated BN is distinct but requires an extra train-data traversal and stronger fairness rationale.
- A shorter EMA horizon might reduce lag, but post-result decay tuning is not justified as an immediate retry.

## Next Steps
- **Medium confidence**: test safe zero-initialized residual endpoints, the remaining developed orthogonal treatment.
- **Low confidence**: develop new optimization evidence before revisiting averaging or architecture.
- **Low confidence**: defer transition label smoothing because local soft-target evidence is negative.

## Exit Action Results
No exit actions were defined.

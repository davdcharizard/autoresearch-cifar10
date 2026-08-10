# Brainstorm EXP-003 (Sweep)
**Created**: 2026-08-05

## Experimental History Review

- The lineage improved BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23%. EXP-002 is the global-best tip and has no failed children.
- EXP-002 established that front-loaded `CUTMIX_PROB=0.5` with `MAX_DROP_PATH=0.08` improves the time-aware WRN by 0.61 points, but both values were fixed hypotheses rather than tuned optima.
- The parent late training loss still approached zero, while CutMix consumed about 3% of optimizer steps. This suggests regularization strength, not capacity or memory, is the current operating-point question.
- CutMix and drop path both regularize the first 75% of charged time. Their strengths may be redundant: reducing drop path could improve fit under CutMix, while changing CutMix probability could better balance clean and mixed supervision.
- No parameter sweep has been run under this goal. Global seed, architecture, optimizer, time schedule, CutMix alpha/end, evaluator, and hardware remain fixed.

## Sweep Plan

**Rationale**: The method and architecture are validated, and the highest-value uncertainty is the balance between the two existing early regularizers. A small grid gives attributable coverage without introducing new code logic or broad low-leverage tuning.

| param | range / choices | scale | mechanism (how it moves the metric) |
|-------|-----------------|-------|-------------------------------------|
| `CUTMIX_PROB` | `{0.35, 0.50, 0.65}` | linear | Controls the proportion of eligible batches receiving input/label mixing; lower values preserve clean fitting, higher values strengthen generalization. |
| `MAX_DROP_PATH` | `{0.04, 0.08}` | linear | Controls residual-path stochastic regularization; 0.04 tests whether CutMix makes the parent's 0.08 rate redundantly strong. |

**Proposed optimizer**: grid - **rough trial budget**: 6 grid points, with parent `(0.50, 0.08) = 95.23%` reused as an existing measured point; run 5 new search trials and one confirmation of the best new candidate.

All trials use physical GPU 0, seed 42, the frozen 300-second charged training budget, 600-second outer timeout, unchanged `CUTMIX_ALPHA=1.0` and `CUTMIX_END=0.75`, and the exact goal verification harness. Trial results are search evidence; only the predeclared winning configuration receives the confirmation run used for the EXP-003 verdict.

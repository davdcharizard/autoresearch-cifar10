# Report EXP-008: Earlier First LR Drop Without Second Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Log**: logs/exp-log-008.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `train.py`-only, single-GPU, 300s training-budget benchmark. The current experiment-index baseline is EXP-002 at 91.95%, and the tightened goal requires a meaningful improvement of at least +0.10 percentage points, so this experiment needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
The chosen idea was to preserve the successful FP32 compile/channels-last ResNet-20 recipe while moving the first LR drop earlier and making the second drop unreachable. The hypothesis was that milestones `[30000, 64000]` would provide more LR 0.01 refinement than EXP-002 without repeating EXP-003's harmful LR 0.001 phase.

## Approach
`train.py` changed only the `MultiStepLR` milestones from `[32000, 48000]` to `[30000, 64000]`. Architecture, optimizer, augmentation, batch size, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation were preserved.

## Execution
One local single-GPU run completed cleanly with output captured to `run.log`. The run reached the planned first LR drop at step 30,000 and stayed at LR 0.0100 through completion; no LR 0.0010 phase occurred. There were no tracebacks, CUDA OOMs, compile failures, or timeouts.

## Results
- **Primary metric**: 91.65% (baseline: 91.95%, delta: -0.30 points, -0.33%)
- **Observations**: The run completed 46,331 steps and 119 epochs, more than EXP-002's 43,398 steps, but peaked at only 91.65%. Accuracy rose quickly after the first drop, reached its best at epoch 102, and then drifted down to 90.94% final accuracy.
- **Analysis**: The hypothesis failed. More LR 0.01 exposure from an earlier 30k drop did not improve the best metric; it appears the original 32k first drop is better calibrated for this ResNet-20 recipe. This narrows the viable schedule space: LR 0.001 at 40k is too low, and LR 0.01 starting at 30k is also too early.
- **Key Learning**: Moving the first LR drop to 30k while avoiding LR 0.001 increases step count but lowers peak accuracy to 91.65%.

## Verification
- **Conditions**: Primary metric condition failed; completion, schedule-behavior, scope, and validation-cadence checks passed.
- **Review Notes**: Results are trustworthy: the run produced numeric metrics, used one GPU, changed only `train.py`, hit the planned 30k LR transition, avoided the second drop, and preserved the fixed evaluator.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.65%` is below both the 91.95% baseline and the required 92.05% improvement threshold.

## Unexplored Avenues
- A first LR drop between 30k and 32k is unlikely to produce the required +0.10 gain, but a very small adjustment such as 31k remains untested.
- A later first drop or a smooth schedule with a 0.01 floor may still be distinct from the failed 30k and 40k milestone variants, but evidence for schedule-only gains is weakening.
- Future work may need a non-schedule mechanism, such as a very mild augmentation or a compact architecture change, while preserving the EXP-002 FP32 throughput path.

## Next Steps
- **High confidence**: Stop broad schedule-only retuning around this baseline; both earlier first drop and reachable second drop reduced accuracy.
- **Medium confidence**: Try a very mild regularization change that preserves step budget, such as weaker/smaller masking rather than full 16x16 cutout.
- **Medium confidence**: Revisit compact WRN-style capacity only with measured, much earlier schedule calibration, since plain ResNet-20 schedule space is narrowing.

## Exit Action Results

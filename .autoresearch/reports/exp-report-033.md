# Report EXP-033: Lower Label Smoothing to 0.03
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Log**: logs/exp-log-033.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline before EXP-033 was 93.70% from EXP-032, and the goal requires at least +0.10 percentage points to count as an improvement, so the concrete EXP-033 threshold was 93.80%.

## Idea & Hypothesis
The chosen idea was to keep the successful EXP-032 anchor but reduce label smoothing from 0.05 to 0.03. The hypothesis was that a slightly milder confidence regularizer would preserve the late-stability benefit while improving peak class separation enough to clear 93.80%.

## Approach
EXP-033 changed only the training loss call from `label_smoothing=0.05` to `label_smoothing=0.03`. The reflected crop padding, 28/56/112 ResNet-20 architecture, batch size, optimizer, weight decay, 21k/64k LR milestones, FP32 channels-last compile path, fixed seed, and once-per-epoch validation were preserved. There were no deviations from the plan.

## Execution
One local single-GPU run was launched on GPU 0 and completed successfully. Startup confirmed CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. The planned first LR drop fired at step 21000 with `lr: 0.0100`; the second milestone at step 64000 was not reached. The run exited cleanly in 398.4 total seconds.

## Results
- **Primary metric**: 93.79% (baseline: 93.70%, delta: +0.09 percentage points, +0.10% relative)
- **Observations**: The run reached 93.77% by epoch 91 and finished with `best_test_acc = 93.79%`, one hundredth of a point below the 93.80% threshold.
- **Analysis**: The hypothesis was directionally plausible but did not meet the goal's noise rule. Lower smoothing improved over the baseline numerically, but the +0.09 point gain is too small to treat as a real improvement under the explicit +0.10 rule.
- **Key Learning**: Lowering label smoothing to 0.03 is not enough to beat the 0.05 anchor under the tightened improvement criterion.

## Verification
- **Conditions**: metric threshold failed; all process and integrity checks passed
- **Review Notes**: Results are trustworthy. The run completed without crash, produced numeric metrics, changed only `train.py`, preserved the fixed 300s training budget, stayed under the 10-minute wall-clock limit, preserved parameter count and batch count, hit the first LR drop at step 21000, avoided the unreachable second LR drop, and had no error/OOM/NaN/Inf signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc = 93.79%` did not reach the required 93.80% threshold.

## Unexplored Avenues
- Test the stronger side of the smoothing axis, such as `label_smoothing=0.08`, if the next brainstorm judges the added regularization risk acceptable.
- Keep `label_smoothing=0.05` and try a separate late-stability mechanism, since 0.03 appeared slightly sharper but did not provide a reliable enough peak.

## Next Steps
Medium confidence: retain `label_smoothing=0.05` as the anchor and move to a non-smoothing low-overhead probe. Medium confidence: consider the 0.08 smoothing side only if the aim is explicitly to test whether stronger late stability beats the underconfidence risk.

## Exit Action Results

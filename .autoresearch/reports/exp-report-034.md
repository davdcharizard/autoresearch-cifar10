# Report EXP-034: Move First LR Drop to 22k on Label-Smoothed Anchor
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Log**: logs/exp-log-034.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline before EXP-034 was 93.70% from EXP-032, and the goal requires at least +0.10 percentage points to count as an improvement, so the concrete EXP-034 threshold was 93.80%.

## Idea & Hypothesis
The chosen idea was to keep the successful label-smoothed reflection anchor but move the first LR drop from step 21000 to step 22000. The hypothesis was that `label_smoothing=0.05` might benefit from slightly more LR 0.1 fitting before LR 0.01 refinement.

## Approach
EXP-034 changed only `LR_MILESTONES` from `[21000, 64000]` to `[22000, 64000]`. The second milestone remained unreachable, and `label_smoothing=0.05`, reflected crop padding, 28/56/112 ResNet-20 architecture, batch size, optimizer, weight decay, FP32 channels-last compile path, fixed seed, and once-per-epoch validation were preserved. There were no deviations from the plan.

## Execution
One local single-GPU run was launched on GPU 0 and completed successfully. Startup confirmed CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. Step 21000 stayed at `lr: 0.1000`, and step 22000 dropped to `lr: 0.0100` as planned. The run exited cleanly in 400.5 total seconds.

## Results
- **Primary metric**: 93.79% (baseline: 93.70%, delta: +0.09 percentage points, +0.10% relative)
- **Observations**: Accuracy reached 93.79% by epoch 64 and never crossed the 93.80% threshold during late refinement.
- **Analysis**: The later first drop was directionally competitive but did not meet the goal's noise rule. It matched EXP-033's 93.79% result, suggesting nearby schedule/smoothing tweaks are sitting just below the required margin rather than clearly improving the anchor.
- **Key Learning**: A 22k first LR drop is not enough to beat the 21k label-smoothed anchor under the +0.10 threshold.

## Verification
- **Conditions**: metric threshold failed; all process and integrity checks passed
- **Review Notes**: Results are trustworthy. The run completed without crash, produced numeric metrics, changed only `train.py`, preserved the fixed 300s training budget, stayed under the 10-minute wall-clock limit, preserved parameter count and batch count, kept step 21000 at LR 0.1, dropped at step 22000, avoided the unreachable second LR drop, and had no error/OOM/NaN/Inf signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc = 93.79%` did not reach the required 93.80% threshold.

## Unexplored Avenues
- Avoid additional tiny first-drop retunes unless paired with a distinct non-schedule change, since both 22k and lower smoothing landed at 93.79%.
- Consider a more distinct no-overhead lever such as mild batch size 112 or the stronger smoothing side only if the next brainstorm accepts the known risks.

## Next Steps
Medium confidence: move away from adjacent 93.79 local probes and test a distinct mechanism. Low-to-medium confidence: batch size 112 is a plausible stochasticity probe but needs strict throughput verification because batch size 96 already failed.

## Exit Action Results

# Report EXP-056: Strong Weight Decay on Weights Only
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Log**: logs/exp-log-056.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The current experiment-index baseline is 93.97% from commit `755be2c`; with the explicit +0.10 percentage-point noise guard, EXP-056 needed at least 94.07% to count as an improvement.

## Idea & Hypothesis
The selected idea was to keep the successful `WEIGHT_DECAY = 2e-4` anchor for convolution and linear weights while excluding BatchNorm affine parameters and all biases from decay. The hypothesis was that EXP-038's stronger-decay gain might come from true weight shrinkage, while decaying normalization affine and bias parameters could harm late calibration.

## Approach
`train.py` was changed to build two SGD parameter groups before `torch.compile`: one group with `weight_decay=2e-4` for non-BatchNorm, non-bias trainable parameters, and one group with `weight_decay=0.0` for BatchNorm direct parameters and all biases. The optimizer uses those groups with the same LR and momentum as the anchor. The implementation also prints decay/no-decay parameter counts at startup for verification.

## Execution
One local foreground run was launched on GPU0 with stdout/stderr captured to `run.log`. Preflight checks passed, the run completed cleanly, and no retries were needed. Startup reported `Weight decay groups: decay_params=820,372, no_decay_params=2,418`, `Batches per epoch: 390`, and `num_params=822,790`. The first LR drop occurred at step 21000 with `lr: 0.0100`.

## Results
- **Primary metric**: 93.68% (baseline: 93.97%, delta: -0.29 percentage points, -0.31%)
- **Observations**: Post-drop accuracy rose quickly to 93.68% by epoch 63, then plateaued and drifted downward, ending at 93.47%. The run used the full 300.0s training budget, took 403.7s total, reached 107 epochs and 41,416 steps, and used 660.4 MiB peak VRAM.
- **Analysis**: The result rejects the hypothesis for this anchor. Removing decay from BatchNorm and bias parameters did not improve calibration; it weakened the known-good global coupled `2e-4` decay recipe. Together with EXP-027, this indicates the recipe prefers simple global SGD L2 decay over parameter-group exceptions, even after the later reflection-padding, label-smoothing, and stronger-decay improvements.
- **Key Learning**: Excluding BatchNorm and bias from `2e-4` decay preserves throughput but underperforms, so global coupled decay remains part of the anchor.

## Verification
- **Conditions**: all passed.
- **Review Notes**: Results are trustworthy. The run completed, produced numeric summary metrics, respected the `train.py`-only scope, preserved batch geometry and parameter count, and reached the first LR drop.
- **Verdict**: no-improvement.
- **Verdict Basis**: `best_test_acc=93.68%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Retune the decay magnitude only for true weights while leaving BatchNorm/bias un-decayed. This could test whether the no-decay split needs a different weight-decay value, but EXP-027 and EXP-056 make the family low priority.
- Apply decay exceptions only to BatchNorm affine while continuing to decay biases, or vice versa. The current result suggests limited upside, so this should wait behind more distinct mechanisms.

## Next Steps
Prefer mechanisms that preserve the proven global `2e-4` coupled decay anchor. Medium-confidence directions include small classifier-head regularization, a coupled mixup/no-label-smoothing balance test, or a non-schedule optimizer tweak that keeps LR 0.1, momentum 0.9, batch 128, and the 21k first drop unchanged.

## Exit Action Results

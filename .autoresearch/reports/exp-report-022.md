# Report EXP-022: 20k First LR Drop on 28/56/112
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Log**: logs/exp-log-022.md

## Goal
EXP-022 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was a schedule-only bracket on the current 28/56/112 ResNet-20 anchor. Prior local results improved when the first LR drop moved from 23k to 22k to 21k, so EXP-022 tested whether moving one adjacent step earlier to 20k would continue that trend.

## Approach
`train.py` changed only `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]`. The architecture, optimizer, weight decay, batch size, augmentation, FP32 compile/channels-last throughput path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 1 with stdout/stderr captured to `run.log`. Startup was clean, the run used the expected 822,790-parameter model, and the planned first LR drop fired at step 20000 with `lr: 0.0100`. The run completed normally after the fixed 300s training budget.

## Results
- **Primary metric**: 93.18% (baseline: 93.23%, delta: -0.05 points, -0.05%)
- **Observations**: Accuracy jumped after the first LR drop, reaching 92.81% by epoch 55. It later peaked at 93.18% around epoch 92 but did not cross the baseline or the 93.33% improvement threshold.
- **Analysis**: The result rejects the hypothesis that 20k is better than 21k on the exact 28/56/112 anchor. It is close enough to show the schedule neighborhood is meaningful, but the 21k first drop remains the best observed local bracket.
- **Key Learning**: A 20k first drop reached 93.18%, showing 21k remains the better local first-drop bracket for the 28/56/112 anchor.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed without traceback, OOM, NaN, or Inf patterns; only `train.py` changed; validation remained once per epoch; and total wall-clock was 392.7 seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.18%` is below the 93.23% baseline and the required 93.33% threshold.

## Unexplored Avenues
- A schedule-only 20.5k first drop is possible but likely too fine-grained relative to the +0.10 threshold and noise margin.
- A mild weight-decay reduction on the 21k anchor remains a cleaner next non-capacity perturbation because it preserves the validated first-drop bracket.
- Later work could pair a 21k first drop with a carefully scoped regularization reduction rather than moving the schedule farther earlier.

## Next Steps
Medium confidence: test lower weight decay (`1e-4` to `5e-5`) on the unchanged 28/56/112, 21k-drop anchor. Low confidence: explore 20.5k schedule granularity only if the loop needs to exhaust schedule brackets; the expected effect may be too small to clear the explicit +0.10 rule.

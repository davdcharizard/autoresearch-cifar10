# Report EXP-024: Reachable Second LR Drop on 28/56/112
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Log**: logs/exp-log-024.md

## Goal
EXP-024 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to preserve the current 28/56/112, 21k first-drop anchor and make the second LR drop reachable. The hypothesis was that changing `LR_MILESTONES` from `[21000, 64000]` to `[21000, 36000]` would create a short late `lr=0.001` refinement phase and lift the peak above the 93.33% threshold.

## Approach
`train.py` changed only `LR_MILESTONES` from `[21000, 64000]` to `[21000, 36000]`. Architecture, weight decay, batch size, optimizer, augmentation, FP32 compile/channels-last path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 1 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, the first LR drop fired at step 21000 with `lr=0.0100`, and the new second drop fired at step 36000 with `lr=0.0010`. The run completed normally with no traceback, OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 93.13% (baseline: 93.23%, delta: -0.10 points, -0.11%)
- **Observations**: Before the second drop, the run plateaued around 92.93%. After the step-36000 drop, the best rose to 93.13% by epoch 103, and final accuracy was 92.92% with final loss 0.2986.
- **Analysis**: The second drop did provide the intended late refinement, but the effect was not large enough to beat the current anchor baseline or the explicit 93.33% improvement threshold. This weakly validates that late LR 0.001 can help, while rejecting 36k as a sufficient isolated schedule change.
- **Key Learning**: A 36k second drop improved late refinement to 93.13%, but remained below the 93.33% threshold.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit both planned LR drops, and finished in 390.5 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.13%` is below the 93.23% baseline and the required 93.33% improvement threshold.

## Unexplored Avenues
- A later second drop such as 38k could preserve more LR 0.01 time, but the 36k result is still 0.20 points short of the threshold.
- A coupled non-schedule change could use the 36k drop as a refinement component, but schedule-only changes are now less attractive.
- Batch-size or optimizer hyperparameter experiments remain more distinct levers than more fine-grained second-drop tuning.

## Next Steps
Medium confidence: move to a non-schedule lever on the 28/56/112, 21k anchor, such as batch size with explicit schedule recalibration. Low confidence: test a later second drop at 38k only if schedule-only exploration must be exhausted. Low confidence: isolated cosine remains possible but carries horizon-tuning risk and one confounded negative prior.

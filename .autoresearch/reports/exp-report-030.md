# Report EXP-030: Reflection Anchor With 32k Second LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Log**: logs/exp-log-030.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness by testing whether the current reflection-padding anchor can exceed the `93.58%` baseline. Under the goal's +0.10 percentage-point rule, EXP-030 needed `best_test_acc >= 93.68%` to count as an improvement.

## Idea & Hypothesis
The selected idea was to preserve the successful EXP-029 reflection-padding anchor and make only the second LR drop reachable by changing `LR_MILESTONES` from `[21000, 64000]` to `[21000, 32000]`. The hypothesis was that EXP-029's late LR 0.01 oscillation could be converted into steadier LR 0.001 refinement, lifting the best accuracy above the noise margin.

## Approach
Implemented the planned one-line schedule change in `train.py` only. Reflected `RandomCrop`, `STAGE_WIDTHS = (28, 56, 112)`, batch size 128, classical momentum, weight decay, FP32 channels-last compile path, seed, and once-per-epoch validation were preserved. There were no implementation deviations.

## Execution
One local run was launched on GPU 1 with output captured to `run.log`. Startup confirmed CUDA, `822,790` parameters, the fixed 300s training budget, and `Batches per epoch: 390`. The first LR drop fired at step 21000 with `lr: 0.0100`; the planned second drop fired at step 32000 with `lr: 0.0010`. The run exited cleanly in 398.9 total seconds.

## Results
- **Primary metric**: 93.33% (baseline: 93.58%, delta: -0.25 points, -0.27%)
- **Observations**: Accuracy reached 93.27% shortly after the first drop, flattened around 93.28% through the 32k second drop, then only nudged to 93.33% by epoch 96.
- **Analysis**: The hypothesis failed. The LR 0.001 phase was reachable and correctly applied, but it reduced useful late exploration instead of improving the reflection-padding anchor. This is now consistent with earlier schedule-only second-drop misses: second LR drops can refine loss but have not produced a meaningful accuracy lift under this fixed budget.
- **Key Learning**: Schedule-only reachable second LR drops are not a productive next lever for the reflection-padding anchor.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed normally, used the fixed budget, preserved scope, hit both planned LR milestones, and reported numeric metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.33%` is below both the `93.58%` baseline and the `93.68%` improvement threshold.

## Unexplored Avenues
- A later second drop could preserve more LR 0.01 exploration, but EXP-024 and EXP-030 together make schedule-only second-drop tuning low priority.
- A second drop might be useful only when paired with a new independently beneficial regularizer or averaging method, but not as the next isolated lever.

## Next Steps
Test sibling crop boundary modes such as symmetric padding with medium confidence; EXP-029 showed padding mode matters, while EXP-030 suggests schedule-only refinement is stale.

Try a low-overhead augmentation or normalization perturbation with medium confidence; it should preserve the reflection anchor and avoid reducing the step budget.

Revisit late averaging only with a short-window or sparse/BN-aware implementation with low confidence; prior averaging attempts were fragile, but EXP-029/030 still show late metric oscillation.

## Exit Action Results

# Report EXP-020: Final-Stage Width 128 with 20k First LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Log**: logs/exp-log-020.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed harness, modifying only `train.py`. The current experiment-index baseline before EXP-020 was 93.23%, and the active goal requires at least +0.10 percentage points over that baseline, so EXP-020 needed `best_test_acc >= 93.33%`.

## Idea & Hypothesis
EXP-020 tested final-stage-only widening: keep the successful 28/56 early-stage widths but widen the final 8x8 stage from 112 to 128 channels. The hypothesis was that final-stage capacity would be cheaper than proportional widening and, with a 20k first LR drop, could exceed the 93.33% improvement threshold.

## Approach
The implementation changed only two constants in `train.py`: `STAGE_WIDTHS = (28, 56, 128)` and `LR_MILESTONES = [20000, 64000]`. Depth, optimizer, augmentation, seed, batch size, compile/channels-last settings, fixed time budget, and once-per-epoch validation were preserved. No implementation deviations were needed.

## Execution
One local single-GPU run was launched on physical GPU 1 because GPU 0 was busy. Startup confirmed CUDA execution, a 300s training budget, 390 batches per epoch, and 1,004,006 parameters. The first LR drop was reached at step 20000 during epoch 52. The run completed cleanly with no traceback, CUDA OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 92.60% (baseline: 93.23%, delta: -0.63 points, -0.68%)
- **Observations**: The run completed 40,989 steps and reached the planned 20k first LR drop, but post-drop accuracy plateaued near 92.6%. Final accuracy was 92.35% with final test loss 0.3503.
- **Analysis**: The hypothesis failed. Unlike EXP-017, this was not mainly a severe step-budget collapse: the run reached more steps than the current-best 28/56/112 report, but the added final-stage capacity still degraded accuracy. This suggests the 28/56/112 channel balance is a better anchor than a heavier final stage under the current recipe.
- **Key Learning**: Final-stage-only widening added parameters but plateaued at 92.60%, so targeted late-stage capacity did not improve the anchor.

## Verification
- **Conditions**: All process and integrity checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run used one GPU, preserved validation cadence, modified only `train.py`, reached the planned LR drop, stayed under 10 minutes total wall-clock, and reported numeric metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=92.60%` is below the required 93.33% threshold.

## Unexplored Avenues
- A smaller final-stage-only width, such as 120 channels with the original 21k first drop, could test whether 128 was too disruptive, but the effect size needed is large and confidence is low.
- Sparse post-drop weight averaging on the 28/56/112 anchor remains distinct from capacity scaling and may target late-epoch fluctuation without changing architecture.

## Next Steps
1. **Sparse post-drop weight averaging** (medium confidence): keep 28/56/112 and 21k schedule, but average weights at low frequency after the first drop to avoid the per-step EMA overhead failure.
2. **Schedule-only local bracket at 20k on 28/56/112** (low-medium confidence): isolate whether the first-drop improvement trend from 22k to 21k continues without architecture changes.
3. **Non-capacity optimizer or regularization micro-change** (low confidence): avoid more widening and look for low-overhead changes that preserve the proven throughput path.

## Exit Action Results

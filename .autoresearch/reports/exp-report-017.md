# Report EXP-017: ResNet-20 Width 30/60/120 with 20k First Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Log**: logs/exp-log-017.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget benchmark while modifying only `train.py`. The current baseline before this experiment was 93.23% from EXP-016, and the tightened success rule required at least +0.10 percentage points, so EXP-017 needed `best_test_acc >= 93.33%`.

## Idea & Hypothesis
EXP-017 tested whether a cautious next width step, from 28/56/112 to 30/60/120 channels, could raise the accuracy ceiling if paired with an earlier 20k first LR drop. The hypothesis was that the extra capacity would improve peak accuracy while the 20k schedule preserved enough LR 0.01 refinement time.

## Approach
The implementation changed only two constants in `train.py`: `STAGE_WIDTHS` from `(28, 56, 112)` to `(30, 60, 120)`, and `LR_MILESTONES` from `[21000, 64000]` to `[20000, 64000]`. Depth, optimizer, augmentation, batch size, FP32 precision, compile/channels-last settings, seed, and once-per-epoch evaluation were preserved.

## Execution
One local run was launched with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` on a single NVIDIA H20. Startup was clean, the model reported 944,200 parameters, and no traceback, CUDA OOM, NaN/Inf, or compile-failure patterns were found. The planned first LR drop was reached at step 20000 during epoch 52 with about 96s remaining.

## Results
- **Primary metric**: 93.16% (baseline: 93.23%, delta: -0.07 percentage points, -0.08%)
- **Observations**: The run completed only 27,400 steps and 71 epochs, far fewer than EXP-016's 34,208 steps. After the drop, accuracy jumped to 92.47% at epoch 52 and peaked at 93.16% at epoch 64, but never reached the 93.33% threshold.
- **Analysis**: The hypothesis was not supported. The wider 30/60/120 model did gain capacity but lost too much step budget relative to 28/56/112, and the 20k first drop left only about 7.4k post-drop steps. This result does not disprove width scaling broadly, but it suggests the next width step needs either a smaller capacity increment, a more aggressive schedule, or a throughput-preserving architecture change.
- **Key Learning**: The 30/60/120 width step reduced the step budget to 27.4k and peaked at 93.16%, below the 28/56/112 baseline.

## Verification
- **Conditions**: primary metric condition failed; all process and scope checks passed
- **Review Notes**: Results are trustworthy. The run completed normally, reported numeric final metrics, used one visible GPU, modified only `train.py`, preserved the fixed evaluator, reached the planned 20k drop, and kept validation to one call per epoch.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=93.16%` is below both the 93.23% baseline and the 93.33% improvement threshold.

## Unexplored Avenues
- Try a smaller capacity step such as 29/58/116 with a 20k or 19k first drop to reduce the throughput penalty while testing whether any capacity headroom remains.
- Test an earlier first drop such as 18k on 30/60/120 only if the goal is to isolate whether the failed result was schedule-missed rather than capacity-limited.
- Explore projection shortcuts or a targeted architecture change on the proven 28/56/112, since it may add representational power with less broad compute growth than widening every stage.

## Next Steps
High confidence: return to the 28/56/112, 21k baseline and test a targeted architectural change such as projection shortcuts at downsample transitions, because broad width scaling just showed a costly throughput penalty.

Medium confidence: try a smaller 29/58/116 width step with an earlier first drop if continuing the width axis, but require close throughput monitoring.

Low-medium confidence: test sparse late averaging on the 28/56/112, 21k recipe if implementation can avoid per-step overhead and BatchNorm-state pitfalls.

## Exit Action Results
- No exit actions defined.

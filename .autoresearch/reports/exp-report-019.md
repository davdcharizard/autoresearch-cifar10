# Report EXP-019: Minimal Width Step 29/58/116 with 19k First LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Log**: logs/exp-log-019.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness by modifying only `train.py`. The active baseline before this experiment was 93.23%, and the goal requires at least +0.10 percentage points to count as an improvement, so EXP-019 needed `best_test_acc >= 93.33%`.

## Idea & Hypothesis
EXP-019 tested whether width scaling still had one small usable increment beyond the successful 28/56/112 recipe. The chosen idea was to use `STAGE_WIDTHS = (29, 58, 116)` with an earlier `LR_MILESTONES = [19000, 64000]`, hoping the smaller width step and earlier drop would preserve enough optimization steps while adding capacity.

## Approach
The implementation changed only two constants in `train.py`: `STAGE_WIDTHS` from `(28, 56, 112)` to `(29, 58, 116)` and `LR_MILESTONES` from `[21000, 64000]` to `[19000, 64000]`. Depth, optimizer, augmentation, seed, batch size, precision, compile/channels-last settings, fixed training budget, and once-per-epoch validation were unchanged.

## Execution
One local run was launched on physical GPU 1 with `CUDA_VISIBLE_DEVICES=1` because GPU 0 was busy. Preflight checks passed, startup confirmed CUDA and `882,451` parameters, and the run reached the 19k first LR drop during epoch 49 with 136s of training budget remaining. The process completed normally before the 10-minute wall-clock cap.

## Results
- **Primary metric**: 92.59% (baseline: 93.23%, delta: -0.64 points, -0.69%)
- **Observations**: The run completed 93 epochs and 36,139 optimizer steps, with `training_seconds=300.0`, `total_seconds=390.2`, and `peak_vram_mb=691.8`. Accuracy jumped after the 19k drop but plateaued at 92.59% from epoch 73 onward.
- **Analysis**: The hypothesis failed. Even this minimal proportional width increase did not preserve enough of the useful 28/56/112 behavior, and the earlier first drop did not recover the lost accuracy. Compared with EXP-017's 30/60/120 result, the step budget improved but the metric was much worse, suggesting the issue is not just schedule timing at wider widths.
- **Key Learning**: Width scaling beyond 28/56/112 is not currently productive under the fixed 300s budget, even with a minimal 29/58/116 step and earlier LR drop.

## Verification
- **Conditions**: All hard/process conditions passed; the improvement condition failed.
- **Review Notes**: Results are trustworthy. The run exited cleanly, reported numeric metrics, used a single selected GPU, preserved the fixed training budget, reached the planned 19k LR drop, kept validation once per epoch, and modified only `train.py` during execution.
- **Verdict**: no-improvement
- **Verdict Basis**: The result was valid, but `best_test_acc=92.59%` was below the required 93.33% threshold.

## Unexplored Avenues
- Final-stage-only widening, such as 28/56/128, may add capacity where spatial maps are cheaper without repeating the broad proportional-width cost.
- Sparse late weight averaging on the current 28/56/112, 21k recipe remains plausible because it targets late low-LR instability without widening the model.

## Next Steps
Try a targeted final-stage capacity change with medium confidence, because proportional widening beyond 28/56/112 now has two negative results but cheaper late-stage capacity is still untested.

Consider a carefully scoped sparse late-averaging experiment with medium confidence, avoiding the per-step EMA overhead that previously failed.

Keep 28/56/112 with a 21k first drop as the schedule anchor with high confidence until a targeted change beats 93.33%.

## Exit Action Results


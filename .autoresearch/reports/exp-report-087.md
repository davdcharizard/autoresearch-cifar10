# Report EXP-087: Fine Upper Flip Bracket p=0.425 Under Padding 3
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-087.md
- **Plan**: plans/plan-087.md
- **Log**: logs/exp-log-087.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The current experiment-index baseline before EXP-087 was 94.51% at commit `83d4e94`; with the +0.10 percentage-point noise guard, EXP-087 needed at least 94.61% to count as an improvement.

## Idea & Hypothesis
The selected idea was a fine upper horizontal-flip bracket on the EXP-085 spatial anchor: keep reflection crop padding 3 and raise `RandomHorizontalFlip` from p=0.4 to p=0.425. The hypothesis was that padding 3 might have slightly under-regularized horizontal invariance, and that a small flip increase could clear 94.61% without returning to the failed p=0.45 setting.

## Approach
`train.py` changed only the training transform's horizontal flip probability from `p=0.4` to `p=0.425`, plus the matching startup print marker. Reflection crop padding 3, unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last path, fixed 300s budget, and validation cadence were preserved. There were no deviations from the plan.

## Execution
One local attached foreground run was launched on GPU0 using `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup markers confirmed the intended crop/flip settings, unchanged CutMix, 822,790 parameters, and the 300s budget. The run reached the first LR drop at step 21000 with `lr: 0.0100`, completed cleanly, and produced final metrics without crashes, NaNs, or OOMs.

## Results
- **Primary metric**: 94.34% (baseline: 94.51%, delta: -0.17pp, -0.18%)
- **Observations**: The run peaked at 94.34% on epoch 75 and never exceeded that value through epoch 102. Final accuracy fell to 93.20%, with 39,425 steps, 102 epochs, 300.0 training seconds, 395.8 total seconds, and 660.4 MB peak VRAM.
- **Analysis**: The hypothesis was not supported. A small upward flip move under padding 3 restored too much spatial regularization or otherwise worsened the local tradeoff. With EXP-084's p=0.45 failure and EXP-087's p=0.425 failure, the upper flip side is now closed enough that p=0.4 should remain the spatial anchor.
- **Key Learning**: Raising horizontal flip probability above p=0.4 under the current spatial anchor regresses accuracy, so the missing gain is not upper-side flip restoration.

## Verification
- **Conditions**: All scope, runtime, scheduler, metric, and hard-constraint checks passed; the improvement threshold failed.
- **Review Notes**: Results are trustworthy. The implementation changed only `train.py`, startup markers matched the intended settings, the LR drop was reached, the metric was numeric, and error scanning found no crash signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=94.34%` is below both the 94.51% baseline and the 94.61% threshold required by the noise guard.

## Unexplored Avenues
- A fine lower flip bracket at p=0.375 under padding 3 remains untested, but its confidence is lower after crop padding 2 and older p=0.35 both underperformed.
- A smaller upper move such as p=0.4125 could test an even tighter bracket, but its expected gain is likely too small to clear the +0.10pp noise guard.
- A non-spatial coupled tweak on the padding-3 / p=0.4 anchor may be more promising now that crop-below-3 and flip-above-0.4 have both failed.

## Next Steps
Try the symmetric fine lower flip bracket p=0.375 under padding 3 with low-to-medium confidence, mainly to close the local spatial bracket cleanly. If it fails, move away from isolated spatial probability/crop tuning toward a small non-spatial coupled mechanism on the EXP-085 anchor.

## Exit Action Results

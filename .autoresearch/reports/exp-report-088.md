# Report EXP-088: Fine Stronger Weight Decay 2.5e-4 on Spatial Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-088.md
- **Plan**: plans/plan-088.md
- **Log**: logs/exp-log-088.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The current experiment-index baseline before EXP-088 was 94.51% at commit `83d4e94`; with the +0.10 percentage-point noise guard, EXP-088 needed at least 94.61% to count as an improvement.

## Idea & Hypothesis
The selected idea was a fine stronger weight-decay retune on the EXP-085 spatial anchor. The hypothesis was that after reducing spatial augmentation strength with padding 3 and flip p=0.4, slightly stronger non-spatial shrinkage (`WEIGHT_DECAY=2.5e-4`) might recover generalization and improve beyond 94.61% without jumping to the older failed `3e-4` setting.

## Approach
`train.py` changed `WEIGHT_DECAY` from `2e-4` to `2.5e-4` and added the startup marker `Weight decay: 0.00025`. Reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer type, LR milestones, batch size, seed, compile/channels-last behavior, fixed 300s budget, and validation cadence were preserved.

## Execution
One local attached foreground run was launched on GPU0 using `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup markers confirmed the intended settings, unchanged CutMix, 822,790 parameters, and the 300s budget. The first LR drop reached step 21000 with `lr: 0.0100`, the run completed cleanly, and no crash, OOM, NaN, or infrastructure error occurred.

## Results
- **Primary metric**: 94.07% (baseline: 94.51%, delta: -0.44pp, -0.47%)
- **Observations**: Pre-drop best reached 87.91%. The LR drop lifted accuracy quickly to 93.46% by epoch 59, but the run peaked at only 94.07% on epoch 71 and then stayed below that through epoch 101. Final accuracy was 93.12%, with 39,060 steps, 101 epochs, 300.0 training seconds, 395.4 total seconds, and 660.4 MB peak VRAM.
- **Analysis**: The hypothesis was not supported. Instead of complementing the milder spatial augmentation, `2.5e-4` weakened the current anchor. Together with the older `3e-4` failure, this indicates the useful `2e-4` decay setting does not have headroom on the stronger side.
- **Key Learning**: Increasing weight decay above `2e-4` weakens the current spatial anchor, so stronger shrinkage is not the missing regularization balance.

## Verification
- **Conditions**: All scope, runtime, scheduler, metric, and hard-constraint checks passed; the improvement threshold failed.
- **Review Notes**: Results are trustworthy. The implementation changed only `train.py`, startup markers matched the intended settings, the LR drop was reached, the metric was numeric, and error scanning found no crash signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=94.07%` is below both the 94.51% baseline and the 94.61% threshold required by the noise guard.

## Unexplored Avenues
- A lower fine weight-decay bracket under the spatial anchor, such as `1.75e-4`, is possible but low confidence because the older `1.5e-4` bracket was already weak.
- A different non-spatial regularization axis, such as a very small CutMix probability adjustment, might interact with the spatial anchor, but existing CutMix brackets are a medium-importance failed family.
- The remaining p=0.375 flip bracket is still the cleanest local spatial closure test, though expected improvement is low after EXP-086 and EXP-087.

## Next Steps
Try p=0.375 under padding 3 with low-to-medium confidence if the priority is closing the local spatial bracket. Otherwise, move to a distinct non-spatial coupled mechanism and avoid stronger weight decay, label-smoothing changes, LR startup changes, and broad CutMix brackets.

## Exit Action Results

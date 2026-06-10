# Report EXP-085: Crop Padding 3 on Flip p=0.4 Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-085.md
- **Plan**: plans/plan-085.md
- **Log**: logs/exp-log-085.md

## Goal
The goal is to maximize CIFAR-10 `best_test_acc` under the fixed harness, where higher is better. The active baseline before EXP-085 was EXP-082 at 94.36% from commit `e859ac5`; with the +0.10 percentage-point noise guard, EXP-085 needed at least 94.46% to count as an improvement.

## Idea & Hypothesis
EXP-085 tested whether the prior reflection crop padding-3 near miss becomes useful after adopting the EXP-082 `RandomHorizontalFlip(p=0.4)` anchor. The hypothesis was that reducing both flip pressure and crop-translation jitter would form a coherent mild spatial de-regularization interaction, raising `best_test_acc` from 94.36% to at least 94.46%.

## Approach
The implementation modified only `train.py`. It changed the training transform from `transforms.RandomCrop(32, padding=4, padding_mode="reflect")` to `padding=3`, preserved `RandomHorizontalFlip(p=0.4)`, and added the startup marker `RandomCrop padding: 3 reflect`. Unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last behavior, validation cadence, and fixed-budget behavior were all preserved.

## Execution
One local foreground run was launched on GPU0 with output captured in `run.log`. Startup confirmed CUDA execution, the intended crop and flip markers, 822,790 parameters, unchanged CutMix settings, and the fixed 300s budget. The first LR drop reached step 21000 with `lr: 0.0100`; post-drop convergence crossed the threshold at epoch 74 with 94.51%. No retries, crashes, NaNs, or infrastructure errors occurred.

## Results
- **Primary metric**: 94.51% (baseline: 94.36%, delta: +0.15pp, +0.16%)
- **Observations**: Pre-drop best reached 88.43% and stayed there through epoch 53. The scheduled LR drop immediately lifted accuracy to 92.01% at epoch 54, then the run peaked at 94.51% at epoch 74. Final test accuracy was 94.10%, and the run completed 102 epochs / 39,685 steps in the fixed 300s training budget.
- **Analysis**: The hypothesis was validated. Padding 3 alone was previously only a sub-threshold result, but the same crop-strength reduction becomes useful when paired with the p=0.4 flip anchor. This suggests the current recipe benefits from a coordinated reduction in spatial augmentation strength, not merely from scalar flip tuning.
- **Key Learning**: Reflection crop padding 3 becomes a real improvement when paired with `RandomHorizontalFlip(p=0.4)`, confirming a productive spatial de-regularization interaction.

## Verification
- **Conditions**: all process, integrity, and improvement-threshold conditions passed.
- **Review Notes**: Results are trustworthy. The run used only `train.py`, preserved the harness, completed cleanly, reported numeric metrics, reached the step-21000 LR drop, preserved parameter count, and used the fixed 300s training budget.
- **Verdict**: improvement
- **Verdict Basis**: all verification conditions passed and `best_test_acc=94.51%` exceeds the 94.36% baseline by +0.15pp, clearing the required 94.46% threshold.

## Unexplored Avenues
- Crop padding 2 on the p=0.4 flip anchor could test whether the successful direction has more room or whether padding 3 is already the lower spatial-jitter edge.
- A fine flip bracket around the new padding-3 anchor, such as p=0.375 or p=0.425, may reveal whether the crop/flip balance shifts after reducing crop jitter.
- Crop padding 5 remains a conceptual control for whether more translation jitter can complement lower flip probability, but EXP-085 makes weaker spatial augmentation the higher-confidence direction.

## Next Steps
Prioritize crop padding 2 on the new padding-3 plus flip p=0.4 anchor with medium confidence. If it regresses, bracket the new anchor with a fine flip-probability adjustment under padding 3. Avoid returning to closed CutMix, label-smoothing, LR-startup, and broad flip-probability families unless coupled to a new mechanism.

## Exit Action Results

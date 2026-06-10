# Report EXP-086: Crop Padding 2 on Padding-3 / Flip p=0.4 Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-086.md
- **Plan**: plans/plan-086.md
- **Log**: logs/exp-log-086.md

## Goal
The goal is to maximize CIFAR-10 `best_test_acc` under the fixed harness, where higher is better. The active baseline before EXP-086 was EXP-085 at 94.51% from commit `83d4e94`; with the +0.10 percentage-point noise guard, EXP-086 needed at least 94.61% to count as an improvement.

## Idea & Hypothesis
EXP-086 tested the lower side of the newly successful crop-padding bracket by reducing reflection crop padding from 3 to 2 while preserving `RandomHorizontalFlip(p=0.4)`. The hypothesis was that EXP-085 might still have too much crop-translation regularization, so reducing padding one more notch could improve `best_test_acc` to at least 94.61%.

## Approach
The implementation modified only `train.py`. It changed `transforms.RandomCrop(32, padding=3, padding_mode="reflect")` to `padding=2` and updated the startup marker to `RandomCrop padding: 2 reflect`. Horizontal flip p=0.4, unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last behavior, fixed 300s budget, and validation cadence were all preserved.

## Execution
One local foreground run was launched on GPU0 with output captured in `run.log`. Startup confirmed CUDA execution, the intended crop and flip markers, 822,790 parameters, unchanged CutMix settings, and the fixed 300s budget. The first LR drop reached step 21000 with `lr: 0.0100`, so the run is a valid schedule comparison. No retries, crashes, NaNs, CUDA errors, or infrastructure issues occurred.

## Results
- **Primary metric**: 94.22% (baseline: 94.51%, delta: -0.29pp, -0.31%)
- **Observations**: Pre-drop accuracy peaked at 88.85% and remained flat through the first LR drop. Post-drop refinement reached 94.11% by epoch 81 and peaked at 94.22% in epoch 86, then stayed below the 94.61% improvement threshold through epoch 102.
- **Analysis**: The lower-side crop-padding bracket did not support the hypothesis. Padding 2 appears to remove too much translation/crop diversity from the successful padding-3 / flip-p=0.4 spatial anchor, regressing below both the baseline and the improvement threshold.
- **Key Learning**: Crop padding 3 is a local lower-edge optimum for the flip p=0.4 anchor; padding 2 under-regularizes and loses accuracy.

## Verification
- **Conditions**: all process and integrity conditions passed; improvement threshold failed.
- **Review Notes**: Results are trustworthy. The run used only `train.py`, preserved the fixed harness, completed cleanly, reported numeric metrics, reached the step-21000 LR drop, and used the fixed 300s training budget.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, but `best_test_acc=94.22%` is below the 94.51% baseline and below the required 94.61% improvement threshold.

## Unexplored Avenues
- A fine upper-side crop bracket such as padding 3 with a small adjacent regularizer remains possible, but padding 2 closes the simple "less crop jitter is better" direction.
- A fine flip bracket around p=0.4 under padding 3, such as p=0.375 or p=0.425, remains untested; expected effects may be too small for the noise guard, but it directly tests the crop/flip interaction.
- A mild optimizer or regularization coupling on top of padding 3 / flip p=0.4 could be more plausible than further reducing spatial augmentation alone.

## Next Steps
- **Medium confidence**: Test `RandomHorizontalFlip(p=0.375)` under reflection padding 3 to see whether the successful crop change shifts the flip optimum slightly lower.
- **Medium confidence**: Test `RandomHorizontalFlip(p=0.425)` under reflection padding 3 as the symmetric fine bracket around p=0.4.
- **Low confidence**: Test a small non-spatial regularization adjustment on the EXP-085 anchor only if fine flip brackets also fail, because many isolated regularizer families are already closed.

## Exit Action Results

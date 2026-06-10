# Report EXP-083: Horizontal Flip Probability 0.35
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-083.md
- **Plan**: plans/plan-083.md
- **Log**: logs/exp-log-083.md

## Goal
The goal is to maximize CIFAR-10 `best_test_acc` under the fixed harness, where higher is better. The active baseline before EXP-083 was EXP-082 at 94.36% from commit `e859ac5`; with the +0.10 percentage-point noise guard, EXP-083 needed at least 94.46% to count as an improvement.

## Idea & Hypothesis
EXP-083 tested the lower side of the newly successful horizontal-flip bracket by changing `RandomHorizontalFlip(p=0.4)` to `p=0.35`. The hypothesis was that the EXP-082 anchor might still be slightly over-regularized, so a modest additional reduction in flip frequency could improve late clean accuracy past 94.46%.

## Approach
The implementation modified only `train.py`. It changed the training transform from `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.35)` and updated the startup marker to `RandomHorizontalFlip p: 0.35`. Reflection crop padding 4, unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last behavior, and validation cadence were all preserved.

## Execution
One local foreground run was launched on GPU0 with output captured in `run.log`. Startup confirmed CUDA execution, the intended flip marker, 822,790 parameters, unchanged CutMix settings, and the fixed 300s budget. The first LR drop reached step 21000 with `lr: 0.0100`, so the run is a valid schedule comparison. No retries, crashes, NaNs, or infrastructure errors occurred.

## Results
- **Primary metric**: 94.17% (baseline: 94.36%, delta: -0.19pp, -0.20%)
- **Observations**: Pre-drop accuracy reached 88.48% by epoch 53. Post-drop refinement climbed to 94.02% by epoch 75 and peaked at 94.17% in epoch 79, then remained below that through epoch 102.
- **Analysis**: The lower-side bracket did not support the hypothesis. Reducing flip probability from 0.4 to 0.35 appears to remove too much useful horizontal invariance for this CutMix anchor, while preserving normal throughput and schedule behavior.
- **Key Learning**: `RandomHorizontalFlip(p=0.35)` regresses from the 94.36% `p=0.4` anchor, so the flip optimum is likely at or above 0.4 rather than lower.

## Verification
- **Conditions**: all process and integrity conditions passed; improvement threshold failed.
- **Review Notes**: Results are trustworthy. The run used only `train.py`, preserved the harness, completed cleanly, reported numeric metrics, reached the step-21000 LR drop, and used the fixed 300s training budget.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, but `best_test_acc=94.17%` is below the 94.36% baseline and below the required 94.46% improvement threshold.

## Unexplored Avenues
- Test the upper-side flip bracket at `p=0.45`; it may preserve most of the EXP-082 de-regularization benefit while restoring useful horizontal invariance.
- Test crop padding 3 on the `p=0.4` anchor only after the flip bracket is locally closed; EXP-081 was a near miss under the older flip setting.

## Next Steps
Prioritize `RandomHorizontalFlip(p=0.45)` with medium confidence. This is the symmetric bracket around the validated `p=0.4` setting and directly tests whether EXP-082 landed below the local optimum.

## Exit Action Results

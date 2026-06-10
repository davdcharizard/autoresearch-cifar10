# Report EXP-084: Horizontal Flip Probability 0.45
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-084.md
- **Plan**: plans/plan-084.md
- **Log**: logs/exp-log-084.md

## Goal
The goal is to maximize CIFAR-10 `best_test_acc` under the fixed harness, where higher is better. The active baseline before EXP-084 was EXP-082 at 94.36% from commit `e859ac5`; with the +0.10 percentage-point noise guard, EXP-084 needed at least 94.46% to count as an improvement.

## Idea & Hypothesis
EXP-084 tested the upper side of the horizontal-flip bracket by changing `RandomHorizontalFlip(p=0.4)` to `p=0.45`. The hypothesis was that the EXP-082 anchor might have removed slightly too much useful horizontal invariance, so restoring some flip probability could improve late clean accuracy past 94.46%.

## Approach
The implementation modified only `train.py`. It changed the training transform from `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.45)` and updated the startup marker to `RandomHorizontalFlip p: 0.45`. Reflection crop padding 4, unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last behavior, and validation cadence were all preserved.

## Execution
One local foreground run was launched on GPU0 with output captured in `run.log`. Startup confirmed CUDA execution, the intended flip marker, 822,790 parameters, unchanged CutMix settings, and the fixed 300s budget. The first LR drop reached step 21000 with `lr: 0.0100`, so the run is a valid schedule comparison. No retries, crashes, NaNs, or infrastructure errors occurred.

## Results
- **Primary metric**: 94.05% (baseline: 94.36%, delta: -0.31pp, -0.33%)
- **Observations**: Pre-drop accuracy reached 88.75% by epoch 41. Post-drop refinement climbed to 93.66% by epoch 59, then peaked at 94.05% in epoch 96 and stayed below the 94.46% improvement threshold.
- **Analysis**: The upper-side bracket did not support the hypothesis. Increasing flip probability from 0.4 to 0.45 appears to restore too much of the over-regularization that EXP-082 escaped, while preserving normal throughput and schedule behavior.
- **Key Learning**: `RandomHorizontalFlip(p=0.45)` regresses from the 94.36% `p=0.4` anchor, so the local flip-probability bracket now favors p=0.4.

## Verification
- **Conditions**: all process and integrity conditions passed; improvement threshold failed.
- **Review Notes**: Results are trustworthy. The run used only `train.py`, preserved the harness, completed cleanly, reported numeric metrics, reached the step-21000 LR drop, and used the fixed 300s training budget.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, but `best_test_acc=94.05%` is below the 94.36% baseline and below the required 94.46% improvement threshold.

## Unexplored Avenues
- A very fine upper bracket at `p=0.425` remains untested, but EXP-084 makes it lower priority because the expected effect is likely too small to clear the +0.10pp guard.
- Crop padding 3 on the `p=0.4` anchor remains plausible because it combines the validated flip setting with a prior spatial near miss, but the interaction risk is higher than a single-axis bracket.

## Next Steps
Prioritize crop padding 3 on the `RandomHorizontalFlip(p=0.4)` anchor with medium confidence. This tests whether the validated flip de-regularization and the prior padding-3 near miss interact constructively while preserving the current CutMix and schedule anchors.

## Exit Action Results

# Report EXP-089: Fine Lower Flip Bracket p=0.375 Under Padding 3
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-089.md
- **Plan**: plans/plan-089.md
- **Log**: logs/exp-log-089.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness, modifying only `train.py`. The active baseline before EXP-089 was 94.51% at commit `83d4e94`, and the goal's +0.10 percentage-point noise guard required `best_test_acc >= 94.61%` to count as an improvement.

## Idea & Hypothesis
EXP-089 tested the remaining lower-side horizontal-flip bracket around the current spatial anchor. The chosen idea was to keep reflection crop padding 3 and reduce `RandomHorizontalFlip` from p=0.4 to p=0.375, checking whether the p=0.4 anchor still had slightly too much horizontal-invariance regularization.

The hypothesis was that a smaller reduction than the failed p=0.35 trial, now paired with the stronger padding-3 crop anchor, could raise `best_test_acc` from 94.51% to at least 94.61%.

## Approach
The implementation changed only `train.py`: `transforms.RandomHorizontalFlip(p=0.4)` became `transforms.RandomHorizontalFlip(p=0.375)`, and the startup marker was updated to `RandomHorizontalFlip p: 0.375`.

All other anchor settings were preserved: reflection crop padding 3, unit-std normalization, CutMix alpha 1.0, CutMix probability 0.5, CutMix label smoothing 0.05, clean-batch label smoothing 0.05, `STAGE_WIDTHS=(28, 56, 112)`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `LR_MILESTONES=[21000, 64000]`, batch size 128, seed 42, FP32 compile/channels-last, fixed 300s budget, and once-per-epoch validation.

## Execution
One local attached single-GPU run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA execution, the intended p=0.375 flip setting, reflection crop padding 3, 822,790 parameters, unchanged CutMix settings, and the 300s training budget.

The first LR drop was reached at step 21000 with `lr: 0.0100`. Pre-drop best was 88.29% through epoch 53; post-drop convergence reached 94.07% by epoch 72 and peaked at 94.29% by epoch 81. The run completed cleanly with no error signatures.

## Results
- **Primary metric**: 94.29% (baseline: 94.51%, delta: -0.22pp, -0.23%)
- **Observations**: The run preserved schedule integrity and completed 39,286 steps over 101 epochs, with peak VRAM 660.4 MB and 822,790 parameters.
- **Analysis**: The hypothesis was not supported. A slight lower flip move under padding 3 underperformed the p=0.4 anchor and did not approach the 94.61% improvement threshold. Combined with EXP-083, EXP-084, and EXP-087, this closes the local flip-probability bracket around p=0.4.
- **Key Learning**: Lowering flip to p=0.375 under padding 3 peaked at 94.29%, closing the lower side around the p=0.4 spatial anchor.

## Verification
- **Conditions**: All integrity and execution conditions passed; the improvement-threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, syntax and ruff checks passed, startup markers matched the plan, the LR drop occurred at step 21000, the run produced numeric metrics, and no crash/error signatures appeared.
- **Verdict**: no-improvement
- **Verdict Basis**: EXP-089 produced a valid metric but did not exceed the 94.51% baseline or the 94.61% noise-guard threshold.

## Unexplored Avenues
- None identified for isolated horizontal flip probability. The lower and upper local brackets around p=0.4 have now both failed, so future spatial work should use a distinct coupled mechanism rather than more isolated flip-probability tuning.

## Next Steps
- Medium confidence: try a small CutMix probability reduction to p=0.4 on the padding-3 / flip-p=0.4 anchor, because spatial tuning is now bracketed and CutMix strength is the next plausible regularization interaction.
- Medium confidence: test a small lower weight-decay bracket such as 1.75e-4 only if CutMix probability remains unhelpful; scalar decay evidence increasingly favors 2e-4.
- Low confidence: consider a tightly scoped late-training mechanism that does not alter validation cadence or miss the 21k LR drop.

## Exit Action Results

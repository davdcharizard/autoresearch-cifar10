# Report EXP-070: Standard CIFAR Channel-Std Normalization
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md
- **Plan**: plans/plan-070.md
- **Log**: logs/exp-log-070.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed benchmark harness, with higher better. The current baseline before EXP-070 was EXP-064 at 94.11%, and the goal requires at least a +0.10 percentage-point gain, so EXP-070 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
The chosen idea was to replace the current training transform's unit channel standard deviation with standard CIFAR-10 channel standard deviations while preserving the EXP-064 CutMix anchor. The hypothesis was that unscaled input variance might be limiting first-layer conditioning or optimization, and standard normalization could lift the current 94.11% anchor.

## Approach
`train.py` was changed only in `main()`: the normalization mean stayed `(0.4914, 0.4822, 0.4465)`, while `std` changed from `(1, 1, 1)` to `(0.2470, 0.2435, 0.2616)`. A startup print was added to make the normalization auditable in `run.log`. Architecture, optimizer, weight decay, LR milestones, reflection crop padding, CutMix alpha/probability, endpoint label smoothing, compile/channels-last, and once-per-epoch validation were unchanged.

## Execution
One foreground local run was executed on GPU1 with output captured to `run.log`. Startup markers confirmed CUDA, ResNet-20 with 822,790 parameters, unchanged CutMix settings, the standard CIFAR std tuple, a 300s training budget, and 390 batches per epoch. The run completed cleanly in 393.4s total with no traceback, CUDA OOM, non-finite, `nan`, or `inf` markers. The first LR drop occurred at step 21000 in epoch 54.

## Results
- **Primary metric**: 75.03% (baseline: 94.11%, delta: -19.08 pp, -20.27%)
- **Observations**: Pre-drop accuracy reached only 61.13%, and the best post-drop checkpoint reached only 75.03% at epoch 79. Final accuracy fell to 67.83%, indicating unstable and severely degraded evaluation after the input rescaling.
- **Analysis**: The hypothesis is rejected. Standard CIFAR std scaling was not a small conditioning improvement for this tuned recipe; it changed the effective input scale enough to badly miscalibrate optimization under the fixed LR, BN, CutMix, and time budget.
- **Key Learning**: Standard CIFAR channel std badly miscalibrated the CutMix anchor, peaking at 75.03%, so keep unit-std input scaling.

## Verification
- **Conditions**: All hard/process conditions passed; the improvement threshold failed.
- **Review Notes**: Results are trustworthy. The run completed cleanly, reported a numeric primary metric, stayed within the time limit, preserved the fixed harness, and modified only `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run with `best_test_acc=75.03%`, far below the 94.21% improvement threshold.

## Unexplored Avenues
- Couple standard CIFAR std with a much lower initial LR: this might address the scale shock, but it would be a broader retuning and conflicts with the local evidence that isolated LR deviations from 0.1 already weaken the anchor.
- Partial input rescaling, such as interpolating between unit std and CIFAR std: this could test whether a gentler conditioning change exists, but the magnitude of EXP-070's degradation makes it low priority.

## Next Steps
- High confidence: keep unit-std input scaling and avoid further isolated normalization changes.
- Medium confidence: test CIFAR AutoAugment on the CutMix anchor now that CutMix scalar/probability brackets and normalization have been exhausted.
- Medium confidence: test fan-out Kaiming initialization as a narrow non-augmentation lever with low implementation risk.

## Exit Action Results

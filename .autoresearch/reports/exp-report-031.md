# Report EXP-031: Symmetric Padding for RandomCrop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Log**: logs/exp-log-031.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness by testing whether a sibling crop-boundary mode can exceed the current `93.58%` reflection-padding baseline. Under the goal's +0.10 percentage-point rule, EXP-031 needed `best_test_acc >= 93.68%` to count as an improvement.

## Idea & Hypothesis
The selected idea was to replace reflected `RandomCrop` padding with symmetric padding while preserving the entire EXP-029 anchor. The hypothesis was that symmetric edge mirroring would keep the no-zero-border benefit from reflection while changing boundary statistics enough to lift `best_test_acc` above the noise margin.

## Approach
Implemented the planned one-line augmentation change in `train.py`: `padding_mode="reflect"` became `padding_mode="symmetric"`. The 28/56/112 ResNet-20 width, batch size 128, optimizer, weight decay, `[21000, 64000]` LR schedule, FP32 channels-last compile path, seed, and once-per-epoch validation were preserved. There were no implementation deviations.

## Execution
One local run was launched on GPU 0 with output captured to `run.log`. Startup confirmed CUDA, `822,790` parameters, the fixed 300s training budget, and `Batches per epoch: 390`. The first LR drop fired at step 21000 with `lr: 0.0100`; the preserved second milestone at 64000 remained unreachable. The run exited cleanly in 397.1 total seconds.

## Results
- **Primary metric**: 93.48% (baseline: 93.58%, delta: -0.10 points, -0.11%)
- **Observations**: Accuracy climbed to 93.48% by epoch 73 after the first LR drop, then oscillated below that peak through completion. The run completed 43,464 steps, similar to the reflection-padding anchor's step budget.
- **Analysis**: The hypothesis failed. Symmetric padding preserved throughput and process validity, but it did not match reflection padding's peak accuracy and remained below the 93.68% improvement threshold. The result strengthens the interpretation that avoiding zero crop borders matters, but the exact mirror semantics also matter; reflection is the better local boundary-fill choice.
- **Key Learning**: Symmetric crop padding is a valid no-overhead sibling, but reflection remains the stronger crop-boundary anchor for this recipe.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed normally, used the fixed budget, preserved scope, retained the expected batch count and parameter count, hit the planned first LR drop, and did not reach the unreachable second drop.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.48%` is below both the `93.58%` baseline and the `93.68%` improvement threshold.

## Unexplored Avenues
- Other `RandomCrop` padding modes are now lower priority; constant zero padding was inferior before EXP-029, and symmetric underperformed reflection here.
- Boundary-fill ideas might still help if paired with a different crop size or padding amount, but that would broaden the augmentation mechanism and should be treated as a separate experiment.

## Next Steps
Try a mild batch-size or stochasticity adjustment around the reflection anchor with medium confidence, while carefully preserving first-drop reachability.

Evaluate a short-window late averaging variant with low-to-medium confidence; it targets late oscillation without retesting schedule-only second drops.

Look for another one-line no-overhead augmentation or optimizer-adjacent change with medium confidence, because capacity and schedule-only spaces are increasingly saturated.

## Exit Action Results

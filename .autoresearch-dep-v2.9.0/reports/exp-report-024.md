# Report EXP-024: BN Bias 64x LR Multiplier
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Log**: logs/exp-log-024.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Direction: higher is better. Threshold: >96.56% (baseline + 0.1pp).

## Idea & Hypothesis
Apply a 64× learning rate multiplier to all BatchNorm bias parameters while keeping other parameters at the base LR. Sourced from airbench96 (`bias_scaler=64.0`). Hypothesis: faster BN bias convergence would allow BN biases to quickly find optimal ReLU operating points, improving feature utilization throughout training by 0.1-0.3pp with zero throughput cost.

## Approach
Replaced the single `optim.SGD(model.parameters(), ...)` call with parameter group separation. Iterated `model.named_parameters()` to identify 19 BN bias parameters (name contains `bn` and ends with `.bias`). Created two optimizer param groups: `norm_biases` with `lr=LR*64.0=12.8` and `weight_decay=0`, and `other_params` with `lr=LR=0.2` and `weight_decay=5e-4`. The existing `LambdaLR` cosine warmup+decay scheduler applies multiplicatively to both groups, maintaining the 64x ratio throughout training. No deviations from plan.

## Execution
Single run, completed within 300s budget. 99 epochs at 16ms/step — zero throughput cost confirmed. However, severe training instability observed: BN bias LR ramped to 12.8 during the 5-epoch warmup, causing wild test accuracy oscillations (26-65% range through epoch 20). The model only stabilized after epoch 80 when cosine decay brought the effective BN bias LR down sufficiently. No crashes or infrastructure issues.

## Results
- **Primary metric**: 94.47% (baseline: 96.46%, delta: -1.99pp, -2.06%)
- **Observations**: The 64x multiplier was catastrophically aggressive for our setup. The effective BN bias LR peaked at 12.8 (vs 0.2 for other params), causing the model to spend ~80% of training in an unstable regime. The model recovered only in the final ~20 epochs as LR decayed toward zero, but could not catch up to baseline performance. Zero throughput cost was the one positive confirmation.
- **Analysis**: The hypothesis was wrong in magnitude. The 64x multiplier was tuned for airbench96's very different setup (custom CNN, lr=9.0, 37 epochs, batch 1024). At lr=9.0, BN biases can tolerate the high absolute LR because the entire training is short and aggressive. At our lr=0.2 with 5-epoch warmup and 99-epoch cosine decay, the 64x ratio creates an extreme mismatch — BN biases dominate early gradient updates and destabilize batch statistics. The failure is in the multiplier magnitude, not the concept.
- **Key Learning**: BN bias LR scaling from airbench96 (64x) is not transferable to standard SGD setups with lower base LR; the multiplier must be calibrated to the effective peak LR, not borrowed from a different recipe.

## Verification
- **Conditions**: Condition 1 FAILED (94.47% << 96.56% threshold). Conditions 2-3 PASSED (clean completion, 99 evals for 99 epochs).
- **Review Notes**: Results confirmed trustworthy — the instability was visible in training dynamics and the metric is plausible given the oscillation pattern observed.
- **Verdict**: no-improvement
- **Verdict Basis**: Verification condition 1 failed — primary metric 1.99pp below baseline, well below the 0.1pp improvement threshold.

## Unexplored Avenues
- **Lower BN bias LR multiplier (4x-8x instead of 64x)**: Would keep effective BN bias LR at 0.8-1.6 during warmup — potentially stable while still accelerating BN bias convergence. The 64x was tuned for lr=9.0; scaling down proportionally to our lr=0.2 suggests ~1.4x (9.0/0.2 * 64 ≈ 1.4 effective ratio adjustment), meaning 4-8x may be the right range.
- **BN bias warmup-only scaling**: Apply the higher LR only during the 5-epoch warmup phase, then revert to base LR. This gives BN biases an early boost without the instability during the main training phase.
- **Weight decay removal for all biases (not just BN)**: The zero-WD aspect of this experiment may have value independent of the LR scaling. Removing WD from all bias parameters is a common practice that could be tested in isolation.

## Next Steps
- **Alternating flip augmentation** (medium confidence): Deterministic flip of all images every other epoch from airbench96. Zero throughput cost, orthogonal to all current techniques. Unclear interaction with TrivialAugmentWide.
- **Deeper architecture NUM_BLOCKS=4** (low-medium confidence): ResNet-26 with ~33% more parameters. Risk of throughput regression (~74 epochs) similar to SE blocks failure.
- **Lower BN bias LR multiplier (4-8x)** (low confidence): Retry the same concept with a much smaller multiplier calibrated to our setup. Evidence is indirect — needs empirical validation.

## Exit Action Results
- Log cleanup: Deleted 6 .log files from repo root (exp-013.log, exp-018.log, exp-023-run.log, run-015.log, run-016.log, run.log). `*.log` already in .gitignore — no git tracking to remove.

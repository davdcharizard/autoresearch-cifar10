# Report EXP-030: Proper Per-Channel Std Normalization
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Log**: logs/exp-log-030.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46%. Threshold: >96.56%.

## Idea & Hypothesis
Change normalization std from (1,1,1) to proper CIFAR-10 per-channel std (0.2470, 0.2435, 0.2616). Hypothesis: better Kaiming init calibration from unit-variance inputs would improve training.

## Approach
Single constant change: std from (1,1,1) to (0.2470, 0.2435, 0.2616).

## Execution
Single run, 99 epochs at 16ms/step. Zero throughput cost.

## Results
- **Primary metric**: 94.67% (baseline: 96.46%, delta: -1.79pp)
- **Observations**: Catastrophic regression. The 4x wider input distribution ([-2,2] instead of [-0.5,0.5]) effectively changed the learning rate for the first conv layer by ~4x, disrupting the optimization dynamics that were tuned for the narrow input range. All hyperparameters (LR=0.2, WD=5e-4, cosine schedule) were calibrated for std=(1,1,1). Changing input normalization without retuning the optimizer is equivalent to changing the effective LR.
- **Key Learning**: Input normalization and optimizer hyperparameters are coupled. The std=(1,1,1) normalization is load-bearing — the entire training recipe (LR, WD, schedule) was tuned for this input scale. Changing it requires retuning everything.

## Verification
- **Conditions**: Condition 1 FAILED (94.67% << 96.56%).
- **Verdict**: no-improvement

## Unexplored Avenues
- Proper std with retuned LR (LR=0.05 to compensate for 4x wider inputs). But this reintroduces multi-variate hyperparameter tuning.

## Next Steps
- Focus on combinations of individually-near-miss changes (Nesterov + reflect padding).
- Or try entirely different data efficiency approaches (knowledge distillation, progressive augmentation).

## Exit Action Results

# Report EXP-013: EMA of Model Weights (Polyak Averaging)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Log**: logs/exp-log-013.md

## Goal
Maximize CIFAR-10 best_test_acc (%, higher is better). Current baseline: 95.39% (EXP-009, commit cfe19c2). Threshold for improvement: >95.49%.

## Idea & Hypothesis
EMA of model weights with β=0.999 — maintain an exponential moving average of parameters updated each step, swap EMA weights in for evaluation. Hypothesis: smoothing SGD noise in final weights provides implicit regularization at zero throughput cost, expected +0.1-0.3pp. Chosen for strongest evidence (2024 dynamics paper), clearest mechanism, safest risk profile, and zero per-step overhead.

## Approach
Three localized changes to train.py following the plan exactly: (1) After model creation (line 152), initialized `ema_shadow` as a dict mapping parameter names to cloned tensors. (2) After `scaler.update()` in training loop (lines 227-229), added `torch.no_grad()` block updating each shadow parameter via `ema_shadow[name].mul_(0.999).add_(p.data, alpha=0.001)`. (3) Before evaluation (lines 263-272), swap model params with EMA shadow for eval, restore originals after. No configuration changes — all hyperparameters identical to EXP-009.

## Execution
Single run, completed normally. 93 epochs, 18064 steps in 300.0s. Per-step time 16-17ms throughout — zero overhead from EMA updates. No errors or retries.

## Results
- **Primary metric**: 94.98% (baseline: 95.39%, delta: -0.41pp, -0.43%)
- **Observations**: Two-phase accuracy trajectory due to BN running stats mismatch: (a) epochs 1-59: severe suppression at 80-90% because BatchNorm running_mean/running_var are buffers (not nn.Parameter), so they are NOT included in EMA shadow — the EMA-averaged conv/fc weights are evaluated with BN statistics computed from non-EMA forward passes; (b) epochs 60-93: rapid recovery after second LR drop (75% mark, LR→0.002) as SGD weights stabilize and BN stats become more compatible with EMA weights, climbing to peak 94.98% at epoch 91.
- **Analysis**: The hypothesis that EMA provides free regularization was correct in principle — the late-training recovery confirms EMA works when SGD weights and EMA weights converge (after LR drops). However, the naive parameter-only implementation fundamentally cannot match baseline because it ignores BatchNorm buffers. The BN mismatch dominates early training and never fully resolves.
- **Key Learning**: Parameter-only EMA is insufficient for models with BatchNorm — must use full state_dict EMA (including buffers) or recalibrate BN statistics after swapping EMA weights.

## Verification
- **Conditions**: Condition 1 (primary metric > 95.49%) FAILED; Conditions 2-3 PASSED
- **Review Notes**: Results confirmed trustworthy — the accuracy suppression is a well-understood consequence of the BN mismatch, not an implementation bug. The metric value 94.98% is genuine.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 94.98% below threshold 95.49% (condition 1 failure)

## Unexplored Avenues
- **Full state_dict EMA (including BN buffers)**: The core EMA idea is sound — the failure was in the parameter-only implementation. Using `model.state_dict()` / `load_state_dict()` instead of `named_parameters()` would include BN running_mean/running_var in the shadow copy and swap. This should eliminate the BN mismatch entirely.
- **BN recalibration after EMA swap**: Instead of including BN buffers in EMA, swap only parameters then run a few forward passes through training data with BN in train mode to recalibrate running stats for the EMA weights. More complex but avoids the EMA-of-BN-stats question.
- **EMA with lower β (0.99 or 0.995)**: Lower β tracks the current weights more closely, reducing the BN mismatch gap at the cost of less smoothing. Could be a simpler fix that partially addresses the issue.

## Next Steps
1. **Mixup α=0.2 replacing RandomErasing** (medium confidence) — next idea from brainstorm. Cross-sample augmentation that smooths decision boundaries; replace rather than stack to avoid over-regularization (EXP-010 lesson). Requires soft-label loss and WD reduction to ~2e-4.
2. **Full state_dict EMA** (medium confidence) — the EMA idea itself is not discredited, only the parameter-only implementation. Full state_dict EMA is the correct fix but adds complexity (deep copy of full state_dict each step).
3. **Increased depth (NUM_BLOCKS=4 or 5)** (low confidence) — more layers within the same width; untried direction. Risk: per-step overhead reduces epoch count.

## Exit Action Results

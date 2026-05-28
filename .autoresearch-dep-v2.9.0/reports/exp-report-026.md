# Report EXP-026: Nesterov Momentum
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Log**: logs/exp-log-026.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Direction: higher is better. Threshold: >96.56% (baseline + 0.1pp).

## Idea & Hypothesis
Enable Nesterov momentum (`nesterov=True` in SGD) to improve gradient estimation quality via look-ahead gradients, particularly in the final cosine decay phase. The prior test of Nesterov (EXP-004) failed in a completely different context (no AMP, batch 128, step-decay LR, width 2x). Hypothesis: +0.1-0.2pp improvement with zero throughput cost.

## Approach
Single parameter change: added `nesterov=True` to `optim.SGD()`. No other changes.

## Execution
Single run, 96 epochs in 300s. 16ms/step confirmed — zero per-step overhead from Nesterov. Wider mid-training oscillations observed (dips to 73.47% at epoch 34, 86.39% at epoch 59) compared to baseline. Model converged well in the final phase. best=final epoch (96.52% at epoch 96).

## Results
- **Primary metric**: 96.52% (baseline: 96.46%, delta: +0.06pp, +0.06%)
- **Observations**: Nesterov produced +0.06pp, slightly better than GC's +0.03pp (EXP-025), but still below the 0.1pp threshold. Both EXP-025 and EXP-026 achieved 96 epochs (vs expected ~99) and landed near 96.5%. This convergence at ~96.5% across multiple optimizer-level interventions suggests the model is hitting a genuine capacity/training-budget ceiling at this accuracy level. The wider oscillations during training (characteristic of Nesterov's look-ahead behavior) did not prevent good final convergence but also didn't substantially improve it.
- **Analysis**: The hypothesis was partially validated — Nesterov did provide a marginal improvement (+0.06pp) with zero per-step overhead. However, the effect was too small to clear the 0.1pp threshold. Three consecutive optimizer-level experiments (BN bias LR, GC, Nesterov) have all failed to produce >0.1pp improvement, strongly suggesting that optimizer tricks alone cannot break through 96.5% with this model capacity and epoch count. The next breakthrough likely requires either more capacity (deeper/wider model) or fundamentally different training approaches (knowledge distillation, different architecture).
- **Key Learning**: Nesterov gives +0.06pp but the model is hitting a capacity ceiling at ~96.5% — three optimizer experiments converge to the same vicinity, suggesting capacity or training length, not optimization quality, is the binding constraint.

## Verification
- **Conditions**: Condition 1 FAILED (96.52% < 96.56% threshold). Conditions 2-3 PASSED.
- **Review Notes**: Results trustworthy — convergence trajectory plausible, consistent with EXP-025's result.
- **Verdict**: no-improvement
- **Verdict Basis**: Verification condition 1 failed — +0.06pp below the 0.1pp threshold.

## Unexplored Avenues
- **Nesterov + shortened warmup (3 epochs)**: Combine Nesterov with reduced warmup to free 2 more epochs of productive training. Zero throughput cost combination.
- **Nesterov + GC combined**: Stack both optimizer tricks. Risk: GC costs 3 epochs from Python-level overhead, potentially negating benefit.
- **Higher learning rate (LR=0.25 or 0.3) with Nesterov**: Nesterov's better gradient estimates might tolerate higher LR, improving exploration.

## Next Steps
- **Combined Nesterov + shortened warmup** (medium confidence): Stack two near-miss zero-cost changes. Nesterov (+0.06pp) and shortened warmup could compound.
- **Deeper architecture NUM_BLOCKS=4** (medium confidence): The ~96.5% ceiling from three optimizer experiments suggests capacity is the binding constraint. ResNet-26 adds ~33% params but risks throughput regression.
- **Nesterov + alternating flip** (low-medium confidence): Stack Nesterov with alternating flip augmentation. Both are zero-cost but augmentation swaps have been at noise floor.

## Exit Action Results
- Log cleanup: Cleaned .log files from repo root.

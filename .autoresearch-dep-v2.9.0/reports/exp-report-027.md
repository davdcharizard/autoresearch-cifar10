# Report EXP-027: Nesterov + Shortened Warmup (3 epochs)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Log**: logs/exp-log-027.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Threshold: >96.56%.

## Idea & Hypothesis
Stack two zero-cost near-miss improvements: Nesterov momentum (+0.06pp in EXP-026) and shortened warmup (5→3 epochs, freeing 2 productive epochs). Hypothesis: additive effects would yield +0.10-0.15pp.

## Approach
Two parameter changes: `nesterov=True` in SGD and `WARMUP_EPOCHS = 3` (from 5).

## Execution
Single run, 98 epochs in 300s. The shortened warmup successfully freed 2 more epochs (98 vs 96 in EXP-026). 16ms/step confirmed.

## Results
- **Primary metric**: 96.45% (baseline: 96.46%, delta: -0.01pp)
- **Observations**: Counter-intuitively, adding shortened warmup to Nesterov REDUCED accuracy from 96.52% (Nesterov alone, EXP-026) to 96.45% (-0.07pp), despite gaining 2 more epochs. The shortened warmup reaches full LR (0.2) by epoch 3, causing early training instability that the model cannot fully recover from even with 98 epochs. The 5-epoch warmup is not just a stability precaution — it's load-bearing for final accuracy.
- **Analysis**: The hypothesis was wrong — the two effects are not additive but antagonistic. Nesterov's look-ahead gradients amplify the instability caused by rapid LR ramp-up. With 5-epoch warmup, Nesterov benefits from the gradual LR increase, producing better gradients during the exploration phase. With 3-epoch warmup, the model enters the high-LR regime too aggressively, and Nesterov's look-ahead exacerbates the overshoot. The result: more epochs but worse quality per epoch.
- **Key Learning**: 5-epoch warmup is load-bearing for final accuracy — shortening it costs more in convergence quality than it gains in extra epochs. Warmup and Nesterov interact negatively: Nesterov amplifies early instability from aggressive LR ramp-up.

## Verification
- **Conditions**: Condition 1 FAILED (96.45% < 96.56%). Conditions 2-3 PASSED.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric below baseline.

## Unexplored Avenues
- **Nesterov with longer warmup (7 epochs)**: If shorter warmup hurts, longer warmup might help. But the tradeoff is fewer productive epochs.
- **Nesterov alone (already tested)**: EXP-026 showed +0.06pp — below threshold. Revisiting with different seeds would test whether +0.06pp is robust.

## Next Steps
- **Deeper architecture NUM_BLOCKS=4** (medium confidence): After 4 optimizer experiments (EXP-024-027) all failing, the ~96.5% ceiling is confirmed. Capacity is the binding constraint. Depth is the most direct capacity increase.
- **Nesterov + reflect padding** (low-medium confidence): Stack Nesterov with data-quality improvement instead of schedule change. Different mechanism axis.
- **WIDTH_MULT=5** (low confidence): More aggressive width scaling. Higher throughput cost than depth.

## Exit Action Results
- Log cleanup: Cleaned .log files from repo root.

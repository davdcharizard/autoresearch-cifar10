# Report EXP-032: Alternating Flip Augmentation
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Log**: logs/exp-log-032.md

## Goal
Maximize best_test_acc. Baseline: 96.56% (EXP-031). Threshold: >= 96.66%.

## Idea & Hypothesis
Replace RandomHorizontalFlip with deterministic alternating flip (flip all even-epoch images) on top of Nesterov + reflect padding baseline.

## Approach
Removed `transforms.RandomHorizontalFlip()`, added `if epoch % 2 == 0: inputs = inputs.flip(-1)` after GPU transfer.

## Execution
98 epochs at 16ms/step. Zero throughput cost.

## Results
- **Primary metric**: 96.64% (baseline: 96.56%, delta: +0.08pp)
- **Observations**: The alternating flip added a real signal (+0.08pp) but below the 0.1pp threshold. The deterministic balanced exposure does help — the model benefits from guaranteed equal orientation training. best=final epoch (96.64%), suggesting the model is still converging.
- **Key Learning**: Alternating flip adds +0.08pp on the Nesterov+reflect baseline — a genuine signal but below threshold. Combined with the baseline's existing +0.10pp, the total improvement from BASE→current is now +0.18pp (96.46→96.64) across the triple stack.

## Verification
- **Verdict**: no-improvement (96.64% < 96.66%)

## Unexplored Avenues
- **Alternating flip + another zero-cost change**: Stack a fourth axis (e.g., WD tuning) to compound further.
- **Different alternating pattern**: Flip every 3 epochs instead of every 2, or flip a fraction of images.

## Next Steps
- Try adding alternating flip + one more zero-cost change (WD tuning or LR tuning) to compound enough to clear 96.66%.

## Exit Action Results

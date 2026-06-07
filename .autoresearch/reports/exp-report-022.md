# Report EXP-022: Gradient clipping (max_norm=5.0)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Log**: logs/exp-log-022.md

## Goal

Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis

Add gradient clipping (max_norm=5.0) to stabilize training against CutMix-induced gradient spikes.

## Approach

Added `scaler.unscale_(optimizer)` + `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` after backward and before optimizer step.

## Execution

Single clean run. 58 epochs in 300s.

## Results

- **Primary metric**: 96.21% (baseline: 96.39%, delta: -0.18%)
- **Analysis**: Gradient clipping did not help. Two possible explanations: (1) gradients in this setup are already well-behaved — CutMix with the current LR and batch size doesn't produce large enough spikes for max_norm=5.0 to activate meaningfully, or (2) the overhead of unscale + clip_grad_norm adds slight per-step cost. The 0.18% delta is within run-to-run variance (we've observed ~0.3% variance across runs with identical training code).
- **Key Learning**: Gradient clipping (max_norm=5.0) has no meaningful effect on this setup; gradients are already well-behaved.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (96.21%)
- **Verdict**: no-improvement

## Unexplored Avenues

- Lower max_norm (e.g., 1.0) — more aggressive clipping, but risks clamping useful gradients
- Weight decay increase (5e-4 → 1e-3)
- CutMix alpha tuning (1.0 → 0.5)

## Next Steps

1. **Weight decay 1e-3** (medium confidence) — WD 1e-4→5e-4 gave +0.48% in EXP-007. Further increase might help if 4.3M params still overfits slightly.
2. **CutMix alpha 0.5** (low-medium confidence) — U-shaped mixing distribution, qualitatively different from prob reduction.
3. **Multiple seeds for variance estimation** (informational) — run baseline 3 times to understand variance before further tuning.

## Exit Action Results

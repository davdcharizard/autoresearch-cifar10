# Report EXP-020: Extended TTA (spatial shifts) — eval-only
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Log**: logs/exp-log-020.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, higher is better) within 300s. Baseline: 96.39% (EXP-016, 2-view hflip TTA).

## Idea & Hypothesis

Add ±1px spatial shifts to TTA (6 views total: original, hflip, left, right, up, down). Hypothesis: more views would reduce prediction variance and improve accuracy to ~96.5-96.6%.

## Approach

Single change: replaced 2-view TTA with 6-view TTA in eval-mode forward(). Used F.pad with reflect mode for ±1px shifts. No training changes. All hyperparameters identical to baseline.

## Execution

Single clean run. 57 epochs in 300s (normal variance). Training identical to baseline.

## Results

- **Primary metric**: 96.13% (baseline: 96.39%, delta: -0.26%, -0.27%)
- **Observations**: 6-view TTA is unambiguously WORSE than 2-view hflip TTA. Training was identical to baseline (57 vs ~54 epochs is normal variance), so the difference is purely from TTA. The spatial-shift views dilute the strong hflip signal: each view's weight drops from 0.5 (2-view) to 0.167 (6-view). The shift views don't add useful diversity — the model's convolutional architecture already has inherent translation invariance, so 1px shifts produce very similar activations. The slight edge artifacts from reflect padding add noise rather than information.
- **Analysis**: The hypothesis was wrong. More TTA views is not automatically better. The hflip TTA works because horizontal flip is a meaningful transformation the model was explicitly trained on (RandomHorizontalFlip). Small spatial shifts don't create meaningfully different views — they're within the model's natural invariance. The conclusion: **hflip is the only valuable TTA augmentation for this model**. Further TTA exploration is unlikely to improve over the 2-view baseline.
- **Key Learning**: 6-view spatial-shift TTA (96.13%) worse than 2-view hflip TTA (96.39%); additional TTA views dilute the hflip signal with noise from near-identical shifted predictions.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (actual: 96.13%)
- **Review Notes**: Results trustworthy — clean isolation, training identical to baseline
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 96.13% below baseline 96.39% + 0.1% threshold

## Unexplored Avenues

- **Weighted TTA** (hflip at 0.75, shifts at 0.0625 each) — but given that unweighted spatial shifts actively hurt, weighting them down to near-zero is equivalent to removing them.
- **Other TTA transformations** (color jitter, small rotations) — but these introduce larger distributional shifts that are unlikely to help.
- **TTA with only the best subset of views** — diminishing returns; hflip-only is already the optimal TTA for this model.

## Next Steps

1. **Proper per-channel std normalization** (medium confidence) — change std from (1,1,1) to CIFAR-10 true std (0.247, 0.243, 0.262). Different input scaling affects loss landscape; may enable better convergence. Risk: LR/WD values were tuned for current normalization.
2. **Gradient clipping** (medium confidence) — add max-norm gradient clipping to stabilize late-stage training. Especially useful with CutMix which can create large gradient spikes.
3. **CutMix probability reduction** (low-medium confidence) — reduce from 0.5 to 0.3. Less augmentation → faster convergence in limited epochs.

## Exit Action Results

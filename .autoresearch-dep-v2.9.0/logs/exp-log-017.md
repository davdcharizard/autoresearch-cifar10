# EXP-017: Mixup α=0.2 Replacing RandomErasing

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented Mixup α=0.2 as a replacement for RandomErasing in four code edits to train.py: (1) Added `MIXUP_ALPHA = 0.2` hyperparameter constant; (2) Removed `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` from the augmentation pipeline; (3) Created `beta_dist = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA)` before the training loop for reuse; (4) Added batch-level mixup logic in the training loop — sample λ from Beta(0.2, 0.2), clamp max(λ, 1-λ), permute batch, mix inputs, construct one-hot labels with label smoothing 0.2 baked in, mix soft targets, and compute manual soft-target cross-entropy loss replacing `F.cross_entropy`.

### Surprises & Discoveries

None — implementation followed the plan straightforwardly. The existing label smoothing parameter (0.2) was hardcoded inline in the loss computation, which was replaced by explicit one-hot + smoothing + mixing.

### Decisions

- Pre-created `beta_dist` outside the loop rather than constructing a new `torch.distributions.Beta` each iteration. Minor optimization to avoid repeated object allocation (~19K iterations).
- Label smoothing value 0.2 is hardcoded inline in the mixup target construction (matching the prior `label_smoothing=0.2` in `F.cross_entropy`) rather than extracted to a named constant — consistent with the plan's minimal-change approach.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running `uv run train.py > run.log 2>&1` locally on single H20 GPU. This is the Mixup α=0.2 experiment replacing RandomErasing. Expect ~98 epochs in 300s budget at ~16ms/step. Target: best_test_acc > 95.67% (baseline 95.57% + 0.1pp threshold).

Observations:
- Training ran 96 epochs in 300.0s budget, 18620 steps — throughput unchanged from baseline (~16ms/step)
- Loss converged smoothly; no NaN/inf or divergence observed
- Test accuracy peaked at 95.53% (epoch ~mid-training) then oscillated in late epochs (95.26-95.49%)
- Final test accuracy 95.33% with test loss 0.4466
- Peak VRAM 865.2 MB — consistent with prior experiments

Key Metrics:
- best_test_acc: 95.53%
- final_test_acc: 95.33%
- final_test_loss: 0.4466
- training_seconds: 300.0
- total_seconds: 408.9
- startup_seconds: 1.2
- peak_vram_mb: 865.2
- num_epochs: 96
- num_steps: 18620
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **best_test_acc > 95.67%**: **FAILED** — Actual: 95.53%. Below the 95.67% threshold (0.14pp short) and 0.04pp below the 95.57% baseline.
2. **Full summary block present**: **PASSED** — All 6 fields present (best_test_acc, final_test_acc, training_seconds, total_seconds, num_epochs, num_steps).
3. **Eval count ≤ num_epochs**: **PASSED** — 96 eval lines = 96 epochs.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> (autopilot session — no human notes)

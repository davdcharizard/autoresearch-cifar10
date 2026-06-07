# EXP-019: Channels_last + T_max=49 + extended TTA (spatial shifts)

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-019
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Three changes to train.py: (1) channels_last memory format on model before EMA deepcopy + on training inputs, (2) extended TTA in forward() eval branch — 6 views (original, hflip, ±1px left/right/up/down shifts using reflect padding), (3) COSINE_T_MAX kept at 49 (unchanged). This combines the EXP-018 speedup with the proven TTA approach from EXP-016.

### Surprises & Discoveries

None — straightforward implementation of well-understood changes.

### Decisions

Used `F.pad(..., mode="reflect")` for spatial shifts to avoid introducing artificial zero boundaries. Each shift is exactly 1 pixel, maintaining 32×32 resolution.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-29
- **Ended**: 2026-05-29

Description:
- Running ResNet-20 k=4 with channels_last (NHWC) memory format, original T_max=49, and 6-view TTA (original + hflip + 4 spatial shifts). Expecting ~59 epochs with near-zero LR refinement + improved eval.

Observations:
- 64 epochs in 300s — channels_last gave ~18% speedup (much more than EXP-018's 59 epochs; variance or torch.compile differences)
- CosineAnnealingLR RESTARTS after T_max=49 — LR increases from 0 back toward 0.1 starting at epoch 55
- best/final gap of 0.90% (96.28% vs 95.38%) confirms model degradation from LR restart
- The assumption that "extra epochs at near-zero LR provide free refinement" was WRONG — CosineAnnealingLR is periodic, not clamped at minimum
- 6-view TTA impact cannot be isolated due to confounding with LR restart degradation

Key Metrics:
- best_test_acc: 96.28% (source: run.log)
- final_test_acc: 95.38% (source: run.log)
- training_seconds: 300.0s (source: run.log)
- num_epochs: 64 (source: run.log)
- peak_vram_mb: 511.3 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.28%, 0.11% below baseline (96.39%). (source: run.log)
2. Remaining conditions skipped.

### Informational Metrics

<!-- Not collected — necessary condition failed. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

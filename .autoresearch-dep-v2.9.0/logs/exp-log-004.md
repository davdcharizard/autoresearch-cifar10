# EXP-004: Nesterov Momentum + Label Smoothing 0.1

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-004
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Two keyword argument changes to train.py: added `nesterov=True` to the SGD optimizer constructor, and added `label_smoothing=0.1` to the `F.cross_entropy()` call. No other changes.

### Surprises & Discoveries
None — trivial changes.

### Decisions
No deviations from plan.

## Experimental Adjustments
(none)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27T15:43:00Z
- **Ended**: 2026-05-27T15:49:00Z

Description:
- Running width-2x ResNet-20 with TrivialAugmentWide + RandomErasing + WD=5e-4 + Nesterov + label_smoothing=0.1. Expected: 93.5-93.8%, recipe polish experiment.

Observations:
- 65 epochs (vs EXP-003's 69) — 4 fewer epochs from higher per-step cost (~11.9ms vs 11.3ms)
- Pre-LR-drop best 89.69% at epoch 34 — significantly better than EXP-003's 81.15% (Nesterov helps high-LR phase)
- Post-second-drop convergence: 93.28% peak at epoch 60, then declined to 93.11% by epoch 65
- Label smoothing increased training loss (expected cosmetic effect) but the model peaked earlier and lower

Key Metrics:
- best_test_acc: 93.28% (baseline: 93.33%, delta: -0.05pp)
- final_test_acc: 93.11%, final_test_loss: 0.2727
- training_seconds: 300.0, total_seconds: 363.6, peak_vram_mb: 598.7
- num_epochs: 65, num_steps: 25126, num_params: 1,073,962

## Verification Results
### Conditions Checked
- **Condition 1**: FAIL — best_test_acc 93.28% < 93.43% threshold (also below baseline 93.33%)
- Condition 2: skipped
- Condition 3: skipped

### Informational Metrics
(not collected — condition 1 failed)

## Errors & Dead Ends

## Human Notes
> (autopilot)

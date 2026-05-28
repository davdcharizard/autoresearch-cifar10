# EXP-003: Weight Decay 5e-4 on Width-2x Augmented Baseline

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-003
- **Commit**: a3e19f8
- **PR**: (skipped — known token permissions issue; user can create from autoresearch/exp-003 → main)
- **Outcome**: completed

## Implementation Notes

### Summary
Changed `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4` in train.py hyperparameters block. Single constant change. All other settings unchanged from EXP-002.

### Surprises & Discoveries
None — trivial one-constant change.

### Decisions
No deviations from plan.

## Experimental Adjustments
(none)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27T15:35:00Z
- **Ended**: 2026-05-27T15:41:00Z

Description:
- Running width-2x ResNet-20 with TrivialAugmentWide + RandomErasing and WD=5e-4 (up from 1e-4). Expected: best_test_acc 93.1-93.5%, aligning recipe with WRN paper's standard. Same wall-clock-fractional schedule, same architecture, same augmentation.

Observations:
- Params 1,073,962 (unchanged), 69 epochs, 11ms/step (same throughput as EXP-002)
- Higher initial loss than EXP-002 (2.62 vs 2.57) — expected with 5x stronger WD
- Pre-LR-drop best: 81.15% at epoch 34 (vs EXP-002's 86.45%) — WD makes training harder
- Post-first-drop (epoch 34-50): rapid convergence to 91.53% — slower catch-up than EXP-002
- Post-second-drop (epoch 52+): strong convergence from 91.53% to 93.33% — second drop delivered +1.8pp
- Final accuracy climbing steadily in last epochs: 92.99→93.08→93.10→93.12→93.33
- The second LR drop contributed even more than EXP-002 (+1.8pp vs +0.52pp)

Key Metrics:
- best_test_acc: 93.33% (source: run.log summary block)
- final_test_acc: 93.33% (source: run.log summary block)
- final_test_loss: 0.1976 (source: run.log summary block)
- training_seconds: 300.0
- total_seconds: 355.4
- peak_vram_mb: 598.7
- num_epochs: 69
- num_steps: 26608
- num_params: 1,073,962

## Verification Results

### Conditions Checked

- **Condition 1**: PASS — best_test_acc 93.33% > 93.02% threshold
- **Condition 2**: PASS — summary block complete
- **Condition 3**: PASS — eval_count (69) == num_epochs (69)

### Informational Metrics

- training_seconds: 300.0, total_seconds: 355.4, peak_vram_mb: 598.7
- num_epochs: 69, num_steps: 26608, num_params: 1,073,962

## Errors & Dead Ends

## Human Notes
> (autopilot)

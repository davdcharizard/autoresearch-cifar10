# EXP-005: AMP (torch.cuda.amp) with GradScaler

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-005
- **Commit**: b934204
- **PR**: (skipped — known token permissions issue)
- **Outcome**: completed

## Implementation Notes

### Summary
Added AMP (FP16 autocast + GradScaler) and channels_last memory format. Model created with channels_last, GradScaler created after optimizer, forward+loss wrapped in autocast context, backward uses scaler.scale(), optimizer step uses scaler.step() + scaler.update(). Input tensors converted to channels_last.

### Surprises & Discoveries
Ruff required reformatting the multi-line .to() call for the model — cosmetic only.

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
- **Started**: 2026-05-27T15:52:00Z
- **Ended**: 2026-05-27T15:59:00Z

Description:
- Running width-2x ResNet-20 with full recipe (aug + WD=5e-4) plus AMP (FP16 + GradScaler + channels_last). Expected: ~100+ epochs in 300s due to throughput improvement, best_test_acc 93.8-94.5%.

Observations:
- 106 epochs (vs 69 baseline) — AMP delivered 1.54x epoch throughput
- Peak VRAM dropped from 598.7 MB to 266.1 MB — FP16 halves activation memory
- Per-step time ~7.3ms (300s / 41179 steps) vs baseline's ~11.3ms — 1.55x speedup
- FP16 caused severe instability in the 0.01 LR phase (epochs 34-52): accuracy oscillated 68-82%, never recovering past pre-drop best of 82.40%
- Second LR drop at epoch ~52 stabilized training: jumped from 82.40% to 90.95% immediately
- Extended 0.001 LR phase (epochs 52-106) delivered massive convergence: 90.95% → 94.44%
- Best accuracy at epoch 101 (94.44%), still climbing — model hit 94.44% at epoch 101, then plateau

Key Metrics:
- best_test_acc: 94.44%, final_test_acc: 94.10%, final_test_loss: 0.1721
- training_seconds: 300.0, total_seconds: 405.8, startup_seconds: 1.1
- peak_vram_mb: 266.1, num_epochs: 106, num_steps: 41179, num_params: 1,073,962

## Verification Results
### Conditions Checked
- **Condition 1**: PASS — 94.44% > 93.43% threshold
- **Condition 2**: PASS — summary block complete
- **Condition 3**: PASS — eval_count (106) == num_epochs (106)

### Informational Metrics
- training_seconds: 300.0, total_seconds: 405.8, peak_vram_mb: 266.1
- num_epochs: 106, num_steps: 41179, num_params: 1,073,962

## Errors & Dead Ends

## Human Notes
> (autopilot)

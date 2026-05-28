# EXP-015: Label Smoothing 0.2 (Standalone)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-015
- **Commit**: 626e9d1
- **PR**: (failed — token lacks createPullRequest permission; branch autoresearch/exp-015 pushed to origin for manual PR if desired)
- **Outcome**: completed

## Implementation Notes

### Summary

Single-line change to train.py line 220: added `label_smoothing=0.2` parameter to the `F.cross_entropy(outputs, targets)` call. No other files or hyperparameters were modified. The change maps directly to Plan Milestone 1 — the `label_smoothing` kwarg is natively supported by PyTorch's `F.cross_entropy` with zero computational overhead.

### Surprises & Discoveries

None — the change was a straightforward single-parameter addition as planned.

### Decisions

No deviations from the plan were necessary.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local process
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run-015.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the full training pipeline with label_smoothing=0.2 added to the cross-entropy loss. This is the sole change from the baseline (EXP-009, 95.39%). The experiment tests whether output distribution regularization via soft targets improves generalization on the width-4x ResNet-20. Expected runtime ~310-320s (300s training budget + startup/eval overhead). We expect ~98 epochs to complete with loss values slightly higher than baseline due to the smoothed targets.

Observations:
- Training completed normally with 98 epochs in 300.0s — identical epoch count to baseline, confirming zero throughput cost of label smoothing
- Loss values slightly higher than baseline due to smoothed targets (expected behavior with label_smoothing=0.2)
- Throughput steady at ~16,300 img/s throughout training
- best_test_acc = final_test_acc = 95.57%, indicating the model peaked at the final epoch
- +0.18pp over baseline (95.39%), consistent with the 0.1-0.3pp hypothesis range

Key Metrics:
- best_test_acc: 95.57%
- final_test_acc: 95.57%
- final_test_loss: 0.3067
- training_seconds: 300.0
- total_seconds: 413.0
- peak_vram_mb: 864.6
- num_epochs: 98
- num_steps: 19100
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **best_test_acc > 95.49%**: PASS — 95.57% > 95.49%
   - Source: `grep "^best_test_acc:" run-015.log` → `best_test_acc:    95.57%`

2. **Full summary block present (count >= 4)**: PASS — count = 4
   - Source: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^peak_vram_mb:" run-015.log` → `4`

3. **Validation runs at most once per epoch (eval count <= num_epochs)**: PASS — 98 eval runs = 98 epochs
   - Source: `grep -c "eval ep" run-015.log` → `98`, `grep "^num_epochs:" run-015.log` → `98`

### Informational Metrics
- training_seconds: 300.0
- peak_vram_mb: 864.6
- final_test_acc: 95.57%
- final_test_loss: 0.3067
- num_epochs: 98
- num_steps: 19100
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> 

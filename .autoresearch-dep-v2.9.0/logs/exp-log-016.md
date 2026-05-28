# EXP-016: Higher BN Momentum (0.5)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-016
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Added a 3-line loop in train.py after model creation (line 149) and before the `num_params` calculation (line 150). The loop iterates `model.modules()`, checks `isinstance(m, nn.BatchNorm2d)`, and sets `m.momentum = 0.5`. This overrides the PyTorch default of 0.1 on all 13 BatchNorm2d layers in the width-4x ResNet-20 (1 stem + 2 per block × 6 blocks). No other files or hyperparameters were modified.

### Surprises & Discoveries

None — the change was a straightforward 3-line addition as planned.

### Decisions

No deviations from the plan were necessary.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local process
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run-016.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the full training pipeline with BN momentum set to 0.5 (up from PyTorch default 0.1) on all BatchNorm2d layers. This is the sole change from the baseline (EXP-015, 95.57%). The experiment tests whether faster BN running statistic convergence reduces the train-eval distribution mismatch in short-budget training (~98 epochs). Expected runtime ~320s (300s training budget + ~20s startup/eval overhead). Zero throughput cost expected — epoch count should remain ~98.

Observations:
- Training completed normally with 98 epochs, 19007 steps in 300.0s — identical epoch count to baseline
- Throughput ~16,300 img/s throughout, no degradation from BN momentum change
- Test accuracy oscillated more during warmup phase (81-86% range at epochs 20-35) compared to typical baseline behavior
- Best accuracy 95.59% achieved at/near final epoch, only +0.02pp over baseline 95.57%

Key Metrics:
- best_test_acc: 95.59%
- final_test_acc: 95.59%
- training_seconds: 300.0
- peak_vram_mb: 864.6
- num_epochs: 98
- num_steps: 19007

## Verification Results

### Conditions Checked

**Condition 1 — best_test_acc > 95.67%**: **FAIL**
- Command: `grep "^best_test_acc:" run-016.log`
- Result: best_test_acc = 95.59%
- 95.59% < 95.67% threshold (baseline 95.57% + 0.1pp)

**Condition 2 — Full summary block present (count >= 4)**: **PASS**
- Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^peak_vram_mb:" run-016.log`
- Result: count = 4

**Condition 3 — Validation runs at most once per epoch**: **PASS**
- Command: eval_count=$(grep -c "eval ep" run-016.log), num_epochs=$(grep "^num_epochs:" run-016.log | awk '{print $2}')
- Result: eval_count=98, num_epochs=98 → 98 <= 98

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 864.6
- final_test_acc: 95.59%
- num_epochs: 98
- num_steps: 19007
- num_params: 4,327,370

## Errors & Dead Ends

## Human Notes

> 

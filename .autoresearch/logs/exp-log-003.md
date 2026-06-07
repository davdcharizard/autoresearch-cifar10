# EXP-003: k=3 + T_max=57 + CutMix

## Execution

Overall Status & Info:
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-003
- **Commit**: 6e52ca5
- **PR**: https://github.com/davdcharizard/autoresearch-cifar10/pull/3
- **Outcome**: completed

## Implementation Notes

### Summary
Retry of EXP-002 with correct static T_max=57. WIDTH_MULT=3 ({48,96,192}), CutMix(alpha=1.0, p=0.5) replacing CutOut, all other settings from EXP-001.

### Surprises & Discoveries
Got 65 epochs (vs EXP-002's 62) — slight variation in epoch count between runs is normal.

### Decisions
Used static T_max=57 instead of dynamic calibration to avoid the torch.compile JIT timing issue from EXP-002.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- k=3 ResNet-20 with T_max=57, CutMix, AMP, torch.compile, Nesterov. Expected ~62 epochs, got 65.

Observations:
- best_test_acc == final_test_acc = 94.80% — perfect T_max alignment
- 65 epochs, 2.4M params, 425MB VRAM

Key Metrics:
- best_test_acc: 94.80% (source: run.log)
- final_test_acc: 94.80% (source: run.log)
- num_epochs: 65 (source: run.log)
- num_params: 2,436,346 (source: run.log)
- peak_vram_mb: 425.5 (source: run.log)

## Verification Results

### Conditions Checked
1. Run completion — **PASS**
2. Time budget (300.0s) — **PASS**
3. Accuracy (94.80% >= 94.13%) — **PASS**
4. Eval frequency (65 = 65) — **PASS**

### Informational Metrics
- final_test_loss: 0.2729
- training_seconds: 300.0
- total_seconds: 380.3
- startup_seconds: 5.8
- num_steps: 25182

## Errors & Dead Ends

## Human Notes

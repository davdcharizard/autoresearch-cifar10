# EXP-018: Channels_last (NHWC) memory format

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Three changes to train.py: (1) Added `model.to(memory_format=torch.channels_last)` after model creation and before EMA deepcopy, ensuring both models use NHWC. (2) Added `memory_format=torch.channels_last` to training input tensor conversion. (3) Changed COSINE_T_MAX from 49 to 55 to exploit expected additional epochs from speedup. All changes are minimal and follow the plan exactly.

### Surprises & Discoveries

The channels_last conversion is placed before `ema_model = copy.deepcopy(model)`, so the EMA model inherits NHWC format automatically. This avoids any format mismatch during the EMA parameter update (`p_ema.data.mul_(...).add_(p.data, ...)`).

### Decisions

No deviations from plan. T_max=55 is conservative (assumes 15% speedup → ~62 epochs, cosine spans 60).

## Experimental Adjustments

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
- Running ResNet-20 k=4 with channels_last (NHWC) memory format and COSINE_T_MAX=55. Expecting faster per-epoch training via cuDNN NHWC optimization with AMP, yielding more epochs in the 300s budget. Target: best_test_acc >= 96.49%.

Observations:
- Channels_last provided ~9% speedup: 59 epochs vs 54 baseline (~5.08s/ep vs ~5.56s/ep)
- Despite more epochs, accuracy is 96.11% — below 96.39% baseline
- T_max=55 (slower LR decay) may have hurt convergence compared to T_max=49
- best==final (96.11%) confirms T_max alignment is good, but the LR schedule shape matters
- num_params=4,327,754 (slightly different from expected 4,301,898 — investigating)

Key Metrics:
- best_test_acc: 96.11% @ epoch 59 (source: run.log)
- final_test_acc: 96.11% (source: run.log)
- training_seconds: 300.0s (source: run.log)
- num_epochs: 59 (source: run.log) — 9% more than baseline's 54
- peak_vram_mb: 553.0 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.11%, 0.28% below baseline (96.39%). (source: run.log)
2. **Training within 300s budget**: skipped — aborted after prior failure
3. **Eval called at most once per epoch**: skipped — aborted after prior failure

### Informational Metrics

<!-- Not collected — necessary condition failed. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

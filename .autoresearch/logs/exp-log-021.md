# EXP-021: Sparse Post-Drop Weight Averaging

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed - no-improvement; `best_test_acc=91.85%` did not reach the required `93.33%` threshold

## Implementation Notes

### Summary

EXP-021 keeps the proven `STAGE_WIDTHS = (28, 56, 112)` and `LR_MILESTONES = [21000, 64000]` anchor, then adds sparse post-drop weight averaging in `train.py`. The implementation imports `AveragedModel`, creates a raw model reference before optional `torch.compile`, trains the compiled wrapper while optimizing the raw model parameters, and updates an averaged copy from the raw model once per epoch after step 21000. The epoch-level evaluator still has one call site and evaluates either the live compiled model before averaging starts or the averaged model after averaging starts.

### Surprises & Discoveries

The installed PyTorch version is `2.11.0+cu130`, and `AveragedModel` supports `use_buffers`, so BatchNorm running statistics can be averaged with parameters. Avoiding a deepcopy of the compiled wrapper is cleaner: the compiled model is used for training forward passes, while the uncompiled raw model remains the source for optimizer parameters and sparse averaging.

### Decisions

The optimizer now receives `raw_model.parameters()` instead of `model.parameters()` after compile so the training state and averaging source are explicit and identical. The log line was extended to include `eval live` or `eval avg` plus `n_avg`, which provides a direct run-log check that averaging activated without adding another validation pass.
- **Retry adjustment**: After Run 1 crashed on the second averaged update, `AVG_USE_BUFFERS` was changed from `True` to `False`. In this PyTorch version, `AveragedModel(use_buffers=False)` averages parameters and then synchronizes buffers by copying them from the source model, avoiding unsupported integer-buffer averaging while keeping BatchNorm buffers current.

## Experimental Adjustments

- **Run 2 switches `AVG_USE_BUFFERS` to `False`**: Run 1 showed the default `use_buffers=True` averaging path crashes on integer BatchNorm buffers. PyTorch's `AveragedModel(use_buffers=False)` averages parameters and copies buffers from the source model after each update, preserving current BatchNorm state without integer averaging. (ref: Run 1 error; local `AveragedModel.update_parameters` source inspection)

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 12981; shell PID 1131578; uv PID 1131582; main Python PID 1131592
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/.autoresearch/logs/exp-021-run1-crash.log
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-06-08 17:50 UTC
- **Ended**: 2026-06-08 17:54 UTC

Description:
- Run the 28/56/112 ResNet-20 anchor with the 21k first LR drop and sparse post-drop averaged-model evaluation. This tests whether smoothing late low-LR weights can improve `best_test_acc` without increasing model capacity or adding per-step EMA overhead. The run must stay on one GPU, preserve one validation call per epoch, show `eval avg` after the first LR drop, and reach at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=22`, and `improvements=6`; EXP-021 threshold is 93.33%.
- GPU check showed physical GPU 1 idle while GPU 0 had active memory/utilization, so the run was launched with `CUDA_VISIBLE_DEVICES=1`.
- Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no traceback/OOM/NaN patterns found. Pre-drop evaluations are labeled `eval live n_avg 00`, preserving one evaluation per epoch, and reached `best_test_acc=84.97%` by epoch 15. (source: run.log L6-L46)
- Run 1 reached the planned first LR drop at step 21000 and activated averaged evaluation at epoch 54 with `eval avg n_avg 01`, reaching 91.49%. It then crashed during the second averaged update because PyTorch attempted integer `addcdiv` on BatchNorm bookkeeping buffers. (source: run.log L111-L140)

Key Metrics:

### Run 2

Metadata:
- **Job ID**: local session 83647; shell PID 1250240; uv PID 1250245; main Python PID 1250256
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 17:55 UTC
- **Ended**: 2026-06-08 18:03 UTC

Description:
- Retry the same sparse post-drop averaging experiment after changing `AVG_USE_BUFFERS` to `False`, which avoids integer-buffer averaging while keeping buffers synchronized from the source model. This retry should again reach the 21k first LR drop, activate `eval avg`, and then continue beyond the second averaged update without the Run 1 `addcdiv` crash.

Observations:
- Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early Run 2 training is healthy with no traceback/OOM/NaN patterns found. Pre-drop evaluations are labeled `eval live n_avg 00` and reached `best_test_acc=84.90%` by epoch 11. (source: run.log L6-L26)
- Run 2 reached the first LR drop at step 21000 during epoch 54, activated averaged evaluation, and passed the previous crash point with `eval avg n_avg 07` by epoch 60. Best averaged accuracy reached 91.85% at epoch 59. (source: run.log L111-L125)
- Run 2 completed without traceback, OOM, NaN, or Inf patterns. The averaged model degraded sharply after its early peak: late averaged evaluations fell to 58.31% at epoch 103 and 51.61% at epoch 110 while the recorded best remained 91.85%. This indicates naive equal post-drop averaging is incompatible with this recipe as implemented. (source: run.log final summary and late `eval avg` lines)

Key Metrics:
- `best_test_acc`: 91.85%
- `final_test_acc`: 51.61%
- `final_test_loss`: 2.9816
- `training_seconds`: 300.0
- `total_seconds`: 394.1
- `startup_seconds`: 2.2
- `peak_vram_mb`: 663.5
- `num_epochs`: 110
- `num_steps`: 42583
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: pass for measurement context; baseline query reported `baseline=93.23`, so the required EXP-021 improvement threshold was `93.33%`.
- Scope: pass; `git diff --name-only` reported only `train.py` among tracked files, with `data/` still untracked.
- Syntax and lint: pass before launch; `python3 -m py_compile train.py` and `uv run ruff check train.py` exited 0.
- Validation cadence: pass; `train.py` retained one `Eval()` construction and one `evaluator.evaluate(...)` call site.
- Sparse averaging activation: pass; Run 2 emitted `eval avg` lines from epoch 54 through epoch 110.
- Experiment completion: pass; Run 2 exited 0 and reported numeric summary metrics.
- Metric improvement: fail; `best_test_acc=91.85%` is below both the 93.23% baseline and the 93.33% improvement threshold.
- Schedule and hard constraints: pass; the first LR drop was reached at step 21000, training used the fixed 300.0 second budget, total wall-clock was 394.1 seconds, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 51.61%
- `final_test_loss`: 2.9816
- `training_seconds`: 300.0
- `total_seconds`: 394.1
- `peak_vram_mb`: 663.5
- `num_epochs`: 110
- `num_steps`: 42583
- `num_params`: 822,790

## Errors & Dead Ends

### 2026-06-08 — AveragedModel integer-buffer averaging crash
- Error: `RuntimeError: Integer division with addcdiv is no longer supported`
- Root cause: `AveragedModel(use_buffers=True)` included integer BatchNorm buffers such as `num_batches_tracked` in the default SWA averaging update.
- Source: run.log L113-L140
- Do NOT retry: do not use `AveragedModel(..., use_buffers=True)` with the default averaging function in this PyTorch version; use `use_buffers=False` so parameters are averaged and buffers are copied from the source model.

## Human Notes

> No human intervention during autopilot execution.

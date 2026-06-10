# EXP-042: Mild Mixup Alpha 0.1

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-042.md
- **Plan**: plans/plan-042.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-042
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - skipped if no remote exists)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the planned mild mixup path in `train.py`. Added `MIXUP_ALPHA = 0.1`, printed the active mixup setting during setup, created a `torch.distributions.Beta` object once before training, sampled `lam` per batch, mixed device-resident inputs with a device-local permutation, and replaced the single-target loss with a weighted two-target cross entropy while preserving `label_smoothing=0.05`.

### Surprises & Discoveries

The only implementation refinement was to create the Beta distribution once before the training loop rather than instantiating it every step. The sampling still happens per batch, matching the experimental intent while reducing avoidable Python overhead.

### Decisions

The evaluation path was left untouched: `Eval.evaluate(model, device)` still runs once per epoch on the model trained with mixup. The implementation keeps the existing optimizer, LR schedule, model shape, augmentation, seed, and fixed training budget unchanged so the result isolates mild mixup.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2605741 (`uv run train.py`; Python worker PID 2605744)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: failed/interrupted
- **Started**: 2026-06-09 08:53 UTC
- **Ended**: 2026-06-09 08:55 UTC

Description:
- Run the current CIFAR-10 training harness locally on a single selected GPU with mild mixup enabled at `MIXUP_ALPHA = 0.1`. This tests whether data and label interpolation improves generalization over the current `2e-4` label-smoothed reflection anchor. The success threshold is `best_test_acc >= 94.07%`, because the active baseline is 93.97% and the goal requires a +0.10 percentage-point improvement.

Observations:
- Startup is clean: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, `Batches per epoch: 390`, and `Mixup alpha: 0.1`. (source: `run.log` L1-L5)
- The run reached epoch 11 with `best: 83.49%`, then stopped after a partial epoch-12 progress line at step 4400. No `best_test_acc` summary, traceback, runtime error, `Killed`, `nan`, or `inf` string was present in the log. (source: `run.log` L7-L28; grep at 2026-06-09 08:57 UTC)
- Throughput was materially slower than the anchor path, with many later steps at 19-23 ms and only 30.2% of the training budget consumed by step 4400; this trajectory made reaching the first LR drop at step 21000 unlikely even if the run had finished. (source: `run.log` L28)

Key Metrics:
- Run 1 partial best validation accuracy: 83.49% at epoch 9.
- Final `best_test_acc`: unavailable because the process ended before the summary line.

### Run 2

Metadata:
- **Job ID**: local session 46675; shell PID 2618844, `uv run train.py` PID 2618845, Python worker PID 2618848
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: failed/interrupted
- **Started**: 2026-06-09 08:59 UTC
- **Ended**: 2026-06-09 09:01 UTC

Description:
- Retry the same EXP-042 code unchanged after Run 1 ended without a final summary or traceback. This retry is for run-control reliability only and does not change the experimental hypothesis, implementation, metric, or `94.07%` success threshold.

Observations:
- Startup is clean on GPU 1: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, `Batches per epoch: 390`, and `Mixup alpha: 0.1`. Early step times are mostly 7-10 ms, much healthier than Run 1. (source: `run.log` L1-L20)
- The foreground session handle returned code `-1`, after which the Python worker for this repo was gone, GPU 1 was idle, and `run.log` stopped advancing at step 6100 in epoch 16. No `best_test_acc` summary, traceback, runtime error, `Killed`, `nan`, or `inf` string was present in the log. (source: `run.log` L7-L36; process/log check at 2026-06-09 09:03 UTC)

Key Metrics:
- Run 2 partial best validation accuracy: 86.16% at epoch 14.
- Final `best_test_acc`: unavailable because the process ended before the summary line.

## Verification Results

### Conditions Checked

- **Single GPU selected**: passed. Run 2 used `CUDA_VISIBLE_DEVICES=1`; Run 1 used `CUDA_VISIBLE_DEVICES=0`.
- **Training completed without crashing/interruption**: failed. Both attempts stopped before the final summary; neither produced a numeric `best_test_acc`.
- **Numeric `best_test_acc` reported**: failed. No final `best_test_acc:` line was present in either log.
- **Improvement threshold met**: not evaluated. The active baseline is 93.97%, so EXP-042 needed `best_test_acc >= 94.07%`, but no final metric was produced.
- **Hard constraints respected by implementation**: passed for code scope. The experiment modified only `train.py`; `prepare.py`, dependency files, and evaluation harness were untouched.

### Informational Metrics

- Run 1 partial best: 83.49% at epoch 9; stopped after step 4400 / epoch 12.
- Run 2 partial best: 86.16% at epoch 14; stopped after step 6100 / epoch 16.
- Neither attempt reached the first LR drop at step 21000.

## Errors & Dead Ends

- **Run 1 interruption**: The interactive execution session returned code `-1`, the `uv run train.py` and Python worker processes were no longer visible, and `run.log` had not been modified since `2026-06-09 08:55:08 UTC`. Because there was no traceback or final metric, this is treated as an interrupted local run rather than an analyzed experimental result.
- **Run 2 interruption**: A retry with unchanged EXP-042 code also lost the foreground session handle with code `-1`; follow-up process inspection showed no Python worker for this repo and a stale `run.log` last modified at `2026-06-09 09:01:23 UTC`. Detached background launches were tested but were immediately reaped with empty logs in this environment, so no third attempt was launched.

## Human Notes

> No human notes yet.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

# EXP-045: Sparse Late EMA After First LR Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md
- **Plan**: plans/plan-045.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-045
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - skipped if no remote exists)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned bounded late EMA path in `train.py`. The code now keeps `base_model` as the optimizer and EMA source model, uses `train_model` as the possibly compiled training-forward wrapper, initializes a PyTorch `AveragedModel` only after the first LR drop boundary, updates it every 100 optimizer steps, and evaluates exactly one model per epoch: raw before EMA activation and EMA afterward.

### Surprises & Discoveries

No implementation surprises. The local PyTorch install exposes both `AveragedModel` and `get_ema_multi_avg_fn`; `python3 -m py_compile train.py` and `uv run ruff check train.py` both passed after the change.

### Decisions

Used `use_buffers=False` in `AveragedModel` to avoid the integer BatchNorm buffer averaging crash observed in EXP-021. Added `eval_model=raw|ema` to epoch evaluation lines and `ema_updates` to the final summary so the analysis phase can verify activation without extra validation passes.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 66673; shell PID 2977363, `uv run train.py` PID 2977364, Python worker PID 2977367
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 12:54 UTC
- **Ended**: 2026-06-09 13:00 UTC

Description:
- Run the current CIFAR-10 training harness locally on one selected GPU with sparse late EMA enabled after the first LR drop. This tests whether a low-overhead EMA evaluation path can smooth post-drop weights enough to improve over the 93.97% baseline. The success threshold is `best_test_acc >= 94.07%`, because the active goal requires a +0.10 percentage-point improvement.

Observations:
- Startup is clean on GPU 1: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. The foreground session is attached as local session 66673. (source: `run.log` L1-L4)
- Early training is healthy: first progress lines show `lr: 0.1000`, evaluation is tagged `eval_model=raw`, validation is once per epoch, and no traceback/OOM/NaN/Inf patterns were present in the initial log scan. By epoch 6, raw validation reached 81.37%. (source: `run.log` L5-L17)
- GPU contention reduced pre-drop coverage; the run reached only 88.71% raw best by epoch 53. The first LR drop still fired at step 21000 with `lr: 0.0100`, and EMA evaluation activated at epoch 54 with `eval_model=ema` and 89.07% best. (source: `run.log` L107-L112)
- EMA then degraded rapidly: later EMA evaluations fell to 84.59%, 83.18%, 81.42%, and finally 79.45%. Final `best_test_acc` was 89.07%, well below the 94.07% threshold. (source: `run.log` L112-L138)
- No second LR drop occurred: `grep -n "step 64000" run.log` returned no matches, as expected because the run completed at 23,495 steps. (source: `run.log` summary and grep exit 1/no output)

Key Metrics:
- `best_test_acc`: 89.07%
- `final_test_acc`: 79.45%
- `final_test_loss`: 0.6958
- `training_seconds`: 300.0
- `total_seconds`: 377.8
- `startup_seconds`: 3.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 61
- `num_steps`: 23,495
- `num_params`: 822,790
- `ema_updates`: 25

## Verification Results

### Conditions Checked
- Baseline and threshold: baseline is 93.97%, so EXP-045 required `best_test_acc >= 94.07%` to count as improvement. Result did not meet threshold. (source: active goal/index baseline and `run.log` summary)
- Scope: tracked source diff during the run was only the planned `train.py` EMA implementation. (source: `git diff -- train.py`)
- API/syntax/lint preflight: `AveragedModel` and `get_ema_multi_avg_fn` were available, `python3 -m py_compile train.py` passed, and `uv run ruff check train.py` passed before launch.
- EMA implementation and anchors: EMA constants and implementation were present, current anchor settings were preserved, and validation remained one `evaluator.evaluate(...)` call per epoch.
- Schedule and EMA activation: `Batches per epoch: 390`, initial progress used `lr: 0.1000`, step 21000 changed to `lr: 0.0100`, EMA eval appeared after activation, step 64000 was absent, `num_params` remained 822,790, and `ema_updates=25`. (source: `run.log`)
- Completion: process exited 0; total wall-clock was 377.8 seconds, below the 10-minute cap, and a numeric `best_test_acc` was printed. (source: `run.log` summary)
- Improvement rule: `best_test_acc=89.07%` is below `94.07%`, so verdict is no-improvement under the +0.10 percentage-point rule.

### Informational Metrics
- EMA behavior: late EMA activated but collapsed from 89.07% to a final 79.45%, indicating serious averaged-parameter / BatchNorm-state mismatch or too-short low-LR window under this configuration.
- Runtime: the run completed 23,495 steps in the fixed 300.0s training budget and 377.8s total wall-clock.

## Errors & Dead Ends

## Human Notes

> No human notes yet.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

# Plan EXP-004: EMA Evaluation Weights
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md

## Milestones

### Milestone 1: Implement EMA without changing the benchmark path
- [x] Add PyTorch SWA/EMA utility imports and EMA configuration constants.
- [x] Keep the base `ResNet` module as the optimizer-owned model, compile only the training forward wrapper, and maintain an EMA copy from the base module.
- [x] Update EMA once after every optimizer step and evaluate the EMA model once per epoch.
- [x] Run static checks to confirm `train.py` parses and the diff only touches `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root.
- [x] Monitor the log for startup, compile, CUDA, or EMA-related failures.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract `best_test_acc`, `peak_vram_mb`, and other final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current `91.95%` baseline with the required +0.10 percentage point margin; EXP-004 needs `best_test_acc >= 92.05%` to count as improvement.
- [x] Preserve enough log evidence for analysis, then remove temporary `run.log` after the result is recorded.

## Code Changes
- **train.py**: import `AveragedModel` and `get_ema_multi_avg_fn` from `torch.optim.swa_utils`.
- **train.py**: add `USE_EMA = True` and `EMA_DECAY = 0.999` near the existing hyperparameters. The decay gives roughly a 1000-step smoothing window, short enough to track late LR phases while still smoothing SGD noise.
- **train.py**: instantiate `base_model = ResNet(...).to(device)`, apply channels-last to `base_model`, count parameters from `base_model`, create `ema_model = AveragedModel(base_model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)`, and then set `model = torch.compile(base_model)` for the training forward path when enabled.
- **train.py**: pass `base_model.parameters()` to SGD so the optimizer owns the real module parameters used by the compiled wrapper.
- **train.py**: call `ema_model.update_parameters(base_model)` immediately after `optimizer.step()` and before scheduler stepping/metric logging.
- **train.py**: evaluate `ema_model` once per epoch when EMA is enabled; this preserves the once-per-epoch validation constraint and leaves `prepare.py` untouched.

## Configuration Changes
- `USE_EMA`: new `True` flag to make the intervention explicit and reversible.
- `EMA_DECAY`: new `0.999` decay value. Higher values risk lagging too much over a 300 second run; lower values may not smooth enough to improve `best_test_acc`.
- No changes to architecture, augmentation, optimizer type, LR milestones, precision, batch size, seed, data path, or evaluation harness.

## Execution Environment
- Method: local command from the project root: `uv run train.py > run.log 2>&1`.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, matching EXP-002 plus minimal EMA update overhead; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` failure that prevents training, or EMA utility API error.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.05%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold before execution:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=91.95`; the improvement threshold is `92.05`.
2. Confirm a single GPU is selected:
   `python - <<'PY'
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY`
   Pass condition: CUDA is available and the visible device count is at least 1; the run must use only one GPU.
3. Remove stale logs if present, then run:
   `uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
4. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
5. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.05`.
   If the value is below `92.05`, classify EXP-004 as `no-improvement` even if it is above `91.95`.
6. Scope review:
   `git diff -- train.py`
   Pass condition: only `train.py` has code changes for the experiment; `prepare.py`, dependency files, and the evaluation harness are unchanged.
7. Validation cadence review:
   inspect `train.py` and `run.log` to confirm exactly one `evaluator.evaluate(...)` call per epoch and no additional validation loop.

### Informational Metrics (Optional)
- final_test_acc: final summary line `final_test_acc:` in `run.log`.
- final_test_loss: final summary line `final_test_loss:` in `run.log`.
- training_seconds: final summary line `training_seconds:` in `run.log`.
- total_seconds: final summary line `total_seconds:` in `run.log`.
- peak_vram_mb: final summary line `peak_vram_mb:` in `run.log`.
- num_epochs: final summary line `num_epochs:` in `run.log`.
- num_steps: final summary line `num_steps:` in `run.log`.
- num_params: final summary line `num_params:` in `run.log`.

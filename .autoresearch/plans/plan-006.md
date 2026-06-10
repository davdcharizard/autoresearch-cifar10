# Plan EXP-006: EMA weight averaging for evaluation
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only: import `AveragedModel` + `get_ema_multi_avg_fn` from `torch.optim.swa_utils`; add `EMA_DECAY = 0.999`; build `ema_model` after the model is on device; call `ema_model.update_parameters(model)` each step after `optimizer.step()`; evaluate `ema_model` (not `model`) in the per-epoch eval
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] `uv run ruff check train.py` passes
- [x] Sanity: prints `4299866` without error (EMA wraps model, params unchanged)

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, `num_params 4,299,866` (UNCHANGED), decreasing loss 2.03→1.40, dt ~11ms/step (throughput-neutral, no NaN)

### Milestone 3: Run completes within budget and summary emitted
- [x] `run.log` contains `best_test_acc:` summary (95.97%); `total_seconds` < 600 (377.7)
- [x] `num_epochs`=70 (within the 65–77 noise band; throughput ~neutral at dt ~10–11ms — EMA update cost negligible)

### Milestone 4: Verification verdict
- [ ] FAIL: `best_test_acc` 95.97 < 96.10 (and < 96.00 baseline) → no-improvement
- [~] skipped (aborted after metric failure)

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The k=4 WideResNet, Cutout,
and the full training recipe (bf16, channels_last, time-fraction cosine, Nesterov, label smoothing, batch 128,
WD 1e-4, PEAK_LR 0.2, seed 42) are held FIXED — the ONLY change is *which weights are evaluated*: an EMA copy
instead of the raw SGD iterate. Strong single-variable attribution.

- **train.py — import (top, with the other torch imports)**: add
  `from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`. *Why*: both are in core torch
  (verified: torch 2.9.1+cu128, `use_buffers` param present) — **no new dependency**.

- **train.py — hyperparameter**: add `EMA_DECAY = 0.999` to the hyperparameter block. *Why*: at ~390 steps/epoch ×
  ~77 epochs ≈ 30k steps, decay 0.999 gives an effective window ~1000 steps (~2.5 epochs) so the average tracks
  the cosine-annealed low-LR tail (where weights settle) rather than stale high-LR weights.

- **train.py — build the EMA model** (right after `model = ResNet(...).to(device, channels_last)` and the
  `num_params` print): 
  ```python
  ema_model = AveragedModel(
      model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True
  )
  ```
  *Why* `use_buffers=True`: averages BN running stats alongside parameters so the EMA weights are evaluated with
  matching BN statistics — the standard practical alternative to a separate (budget-costing) BN-recompute pass.
  AveragedModel deep-copies the on-device channels_last model, so `ema_model` inherits device + memory format.

- **train.py — update EMA each step**: immediately after `optimizer.step()` (inside the timed region, before
  `torch.cuda.synchronize()`), add `ema_model.update_parameters(model)`. *Why inside the timed region*: the EMA
  lerp is a real per-step cost and must be charged to the 300s budget honestly (it is negligible — one in-place
  lerp over 4.3M params, no host sync). The first call copies the live weights (n_averaged=0→1); subsequent calls
  apply the EMA recursion.

- **train.py — evaluate the EMA model**: change the per-epoch eval call from
  `evaluator.evaluate(model, device)` to `evaluator.evaluate(ema_model, device)`. *Why this tests the hypothesis*:
  reports `best_test_acc` from the trajectory-averaged weights (flatter minimum) instead of the final iterate.
  `evaluate` calls `model.eval()` then `model(inputs)`; `AveragedModel` is a transparent drop-in (delegates
  forward to `.module`, supports `.eval()`). This remains exactly **one eval per epoch** — constraint preserved.

## Configuration Changes
- EMA_DECAY: (new) `0.999` — EMA decay for the evaluated weight average
- Evaluated weights: raw SGD iterate (`model`) → EMA copy (`ema_model`)
- ALL else UNCHANGED: WIDTH_MULT 4, Cutout(16) via cutout_batch, PEAK_LR 0.2, WD 1e-4, label smoothing 0.1,
  batch 128, bf16, channels_last, Nesterov, cosine schedule, MAX_STEPS 10_000_000, seed 42, eval frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM ≈ EXP-003 + one extra model copy (≈ +5 MB; ~500 MB total,
  far below the 98 GB ceiling — VRAM is free per project-insights).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. EMA is throughput-neutral → expect epochs
  ≈ EXP-003 (~77).
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:|^final_test_loss:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging (rising) for sustained steps after warmup → kill, treat as crash.
- Python traceback in `run.log` (empty `best_test_acc:` at end) → crash; inspect `tail -n 50 run.log`.
- No new output in `run.log` for > 2 min while training should be active → kill (hang).
- Total wall-clock > 10 min → kill, treat as failure.
- num_epochs collapses well below ~70 (toward EXP-002's 54) → not an abort per se, but record it: would mean the
  EMA update unexpectedly added cost (note for analysis).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (necessary condition: runs cleanly in budget):
   - `grep -aE "^best_test_acc:|^total_seconds:" run.log`; `tail -n 50 run.log` for tracebacks.
   - PASS if `best_test_acc:` present (non-empty), `total_seconds < 600`, no traceback. Timeout: 10 min.
2. **Metric improvement** (necessary condition: `best_test_acc` ≥ baseline + 0.1):
   - Parse `best_test_acc`. PASS if `best_test_acc >= 96.10`. FAIL (→ no-improvement) otherwise.
3. **No constraint violations** (necessary condition: no constraint violations):
   - `git diff --name-only autoresearch/dev` shows only `train.py`; no diff on `pyproject.toml`/`uv.lock`
     (no new deps — swa_utils is core torch); eval-line count == num_epochs (eval once/epoch); seed unchanged
     (`grep manual_seed train.py` → 42).
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirm EMA is throughput-neutral
  (expect ~75–77 as EXP-003, NOT a drop)
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — EMA often lowers eval loss (flatter minimum);
  compare to EXP-003's 0.204
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — expect ~500 MB (one extra model copy vs EXP-003)
- num_params: `grep -aE "^num_params:" run.log` — must be **4,299,866 (unchanged)**, confirming EMA adds no model
  capacity (pure weight averaging, not a capacity change)

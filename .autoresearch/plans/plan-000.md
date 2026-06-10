# Plan EXP-000: Modern training recipe — bf16 AMP + channels_last + budget-matched cosine schedule
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only (per scope) with the recipe changes below
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes (syntax)
- [x] `uv run ruff check train.py` passes (pre-commit style; auto-fix if needed)

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s of training start, `run.log` shows `Device: cuda`, the param count, and
      step/loss progress lines (loss decreasing, not NaN)

### Milestone 3: Run completes within budget and summary emitted
- [x] `run.log` contains the `best_test_acc:` summary block (92.06%)
- [x] `training_seconds` ≈ 300 (300.0) and `total_seconds` < 600 (388.6)
- [x] Run did not crash (non-empty `best_test_acc`)

### Milestone 4: Verification verdict
- [x] `best_test_acc >= 91.83` (92.06 ≥ 91.83)
- [x] No constraint violations (only train.py changed, eval once/epoch, no new deps, no seed hacking)

## Code Changes
All changes are confined to `train.py` (only editable file; `prepare.py` is frozen/hook-protected).

- **train.py — precision (bf16 AMP)**: Wrap the forward pass + loss computation in
  `with torch.autocast(device_type="cuda", dtype=torch.bfloat16):`. No `GradScaler` is needed for
  bf16 (wide exponent range, no underflow). `backward()` and `optimizer.step()` stay in fp32 as usual.
  *Tests hypothesis*: raises steps/sec → more epochs in the fixed 300s budget. H20 has native bf16.

- **train.py — memory format (channels_last)**: Convert the model with
  `model = model.to(device, memory_format=torch.channels_last)` and each input batch with
  `inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)`.
  *Tests hypothesis*: channels_last is the throughput-friendly layout for conv nets; compounds with AMP.

- **train.py — budget-matched LR schedule**: Remove `MultiStepLR`. Replace with a manual schedule set
  every step from the **elapsed-time fraction** `frac = total_training_time / TIME_BUDGET_S`:
  linear warmup over the first `WARMUP_FRAC` of the budget up to `PEAK_LR`, then cosine anneal to ~0 by
  `frac = 1.0`. Set `optimizer.param_groups[0]["lr"]` before `optimizer.step()` each iteration.
  *Tests hypothesis*: the baseline's step-space `MultiStepLR([32000,48000])` never anneals within the
  ~35k steps that fit in 300s (2nd drop never fires). Time-driven cosine guarantees a full anneal-to-zero
  regardless of realized throughput — the single highest-confidence lever. Driving by *time* (not step
  count) keeps the schedule correct even though AMP changes how many steps fit.

- **train.py — optimizer**: Add `nesterov=True` to the SGD optimizer (momentum already 0.9).
  *Tests hypothesis*: Nesterov momentum is a consistently-positive, zero-risk tweak for ResNet SGD.

- **train.py — loss**: Training loss becomes `F.cross_entropy(outputs, targets, label_smoothing=0.1)`.
  *Tests hypothesis*: label smoothing improves generalization/calibration. (Eval loss/accuracy in
  `prepare.py` is untouched — accuracy is argmax-based, unaffected by training-side label smoothing.)

- **train.py — step cap**: Raise `MAX_STEPS` to a large value (`10_000_000`) so the **time budget is the
  sole limiter**. This is required for the time-fraction schedule to anneal fully: if AMP pushes step
  throughput high enough to hit the old `MAX_STEPS=64000` before 300s elapse, training would stop with
  the LR not yet annealed to zero. The budget is defined by `TIME_BUDGET_S` (frozen in `prepare.py`);
  `MAX_STEPS` is a train.py hyperparameter, so this is in-scope and does not change the time budget.

## Configuration Changes
- LR scheme: `MultiStepLR(milestones=[32000,48000], gamma=0.1)` → time-fraction warmup+cosine to 0
- PEAK_LR: (effective const) `0.1` → `0.2` (one-cycle-style peak; warmup mitigates instability, BN tolerant)
- WARMUP_FRAC: new → `0.05` (≈first 5% of the 300s budget linearly warms 0→PEAK_LR)
- MOMENTUM: `0.9` → `0.9` (unchanged) + `nesterov=True` (new)
- WEIGHT_DECAY: `1e-4` → `1e-4` (unchanged — isolate the recipe; revisit later)
- BATCH_SIZE: `128` → `128` (unchanged — isolate precision/layout/schedule; batch scaling is a follow-up)
- label_smoothing: none → `0.1`
- precision: fp32 → bf16 autocast (forward+loss)
- memory_format: contiguous → channels_last
- MAX_STEPS: `64000` → `10_000_000` (make time the sole limiter)
- Seed: unchanged (`torch.manual_seed(42)` / `cuda.manual_seed(42)`) — no seed hacking

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background; do not tee/flood context)
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM need tiny (baseline 330 MB; bf16/channels_last similar).
- Estimated runtime: ~300s training + ~60–70s startup/eval ≈ **6–7 min** total wall-clock.
- Log output: all stdout/stderr → `run.log` in project root. Primary source of truth.
  Extract via `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^training_seconds:|^num_epochs:|^num_steps:" run.log`.
- Tool skill: none (local run; no job platform).

## Abort Criteria
- Loss is `NaN`/`inf` or diverging (rising, not falling) for sustained steps after warmup → kill, treat as crash.
- No new output in `run.log` for > 2 min while training should be active (hang) → kill.
- Total wall-clock exceeds **10 min** (per TASK.md) → kill, treat as failure.
- Python traceback in `run.log` (empty `best_test_acc:` at end) → run crashed; inspect `tail -n 50 run.log`.
- Peak VRAM approaches the 98 GB ceiling (not expected; model is tiny) → kill.

## Verification Protocol

### Verification Procedure
Run from project root after the run completes. Baseline = **91.73** (from `exp-index.sh baseline`); success bar = **91.83**.

1. **Confirm clean completion within budget** (necessary condition 2):
   - `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:" run.log`
   - PASS if `best_test_acc:` line is present (non-empty), `total_seconds < 600`, and no Python traceback
     appears in `tail -n 50 run.log`. FAIL (→ crash/no-improvement) otherwise. Timeout: 10 min wall-clock.
2. **Confirm metric improvement** (necessary condition 1):
   - Parse `best_test_acc` value from the summary line.
   - PASS if `best_test_acc >= 91.83`. FAIL (→ no-improvement) otherwise.
3. **Confirm no constraint violations** (necessary condition 3):
   - `git diff --name-only` (on experiment branch) shows only `train.py` modified.
   - Confirm no new entries in `pyproject.toml` (no deps added), `evaluate()` called once per epoch
     (unchanged loop structure — one eval call per epoch iteration), seed unchanged.
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; evaluation stops at the first failure.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — VRAM headroom used (soft-constraint awareness)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — how much training fit in 300s
  (compare to baseline 90 epochs / 34,861 steps to quantify the AMP throughput gain)
- img/s throughput: from the step progress lines (`img/s:` field) — efficiency vs. baseline fp32

# Plan EXP-005: Weight decay 1e-4 → 5e-4 (k=4 + Cutout)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md

## Milestones

### Milestone 1: Code change implemented and parse-clean
- [x] Edit `train.py` only: `WEIGHT_DECAY = 1e-4` → `5e-4`
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes
- [x] Confirm `grep -nE "^WEIGHT_DECAY" train.py` shows 5e-4 and WIDTH_MULT still 4

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, params 4,299,866, decreasing loss (no NaN)

### Milestone 3: Run completes within budget
- [x] `run.log` has `best_test_acc:` summary (96.05); `total_seconds` < 600 (376.4)
- [x] num_epochs=65 (lower than expected ~77 — transient ~17ms/step; WD is compute-neutral)

### Milestone 4: Verification verdict
- [x] FAIL: best_test_acc 96.05 < 96.10 → no-improvement
- [~] skipped (aborted after metric failure)

## Code Changes
Single change in `train.py` (only editable file). Model (k=4), Cutout, schedule, batch, LR, label smoothing,
seed all fixed for clean attribution.

- **train.py — `WEIGHT_DECAY` 1e-4 → 5e-4**: passed to `optim.SGD(..., weight_decay=...)`.
  *Why*: 5e-4 is the standard WRN weight decay; our 1e-4 is a leftover from the ResNet-20 (k=1) era and the
  4.3M-param model is likely under-regularized on the L2 axis. Throughput-neutral (epochs stay ~77), so it
  isolates the regularization effect on the proven-sweet-spot k=4 model.

## Configuration Changes
- WEIGHT_DECAY: `1e-4` → `5e-4`
- ALL else UNCHANGED: WIDTH_MULT 4, Cutout(16) via cutout_batch, PEAK_LR 0.2, label smoothing 0.1, batch 128,
  bf16, channels_last, Nesterov, cosine schedule, MAX_STEPS 10_000_000, seed 42, eval frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0). VRAM ~492 MB (unchanged).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. Epochs ~77 (WD is free).
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, crash.
- Python traceback / empty `best_test_acc:` at end → crash; inspect `tail -n 50 run.log`.
- No new log output > 2 min while training → kill (hang).
- Total wall-clock > 10 min → kill, failure.

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (cond 2): `grep -aE "^best_test_acc:|^total_seconds:" run.log`;
   `tail -n 50 run.log` for tracebacks. PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback. Timeout 10 min.
2. **Metric improvement** (cond 1): parse `best_test_acc`. PASS if `>= 96.10`, else no-improvement.
3. **No constraint violations** (cond 3): `git diff --name-only autoresearch/dev` = only `train.py`; no
   pyproject/uv.lock diff; eval-line count == num_epochs; seed unchanged (42). PASS if all hold, else invalid.
   Stop at first failure.

### Informational Metrics (Optional)
- final_test_loss: overfit check vs EXP-003's 0.204 (stronger WD should reduce train/test gap)
- num_epochs: confirm ~77 (throughput-neutral)
- peak_vram_mb: confirm unchanged

# Plan EXP-004: Increase width to k=6
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md

## Milestones

### Milestone 1: Code change implemented and parse-clean
- [x] Edit `train.py` only: `WIDTH_MULT = 4` → `6`
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes
- [x] Sanity: `uv run python -c "import train; m=train.ResNet(train.NUM_BLOCKS, width_mult=6); print(sum(p.numel() for p in m.parameters()))"` prints ~9–10M

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, the new (~9.7M) param count, decreasing loss (no NaN)

### Milestone 3: Run completes within budget; epoch count acceptable
- [x] `run.log` has `best_test_acc:` summary (95.26); `total_seconds` < 600 (356.0)
- [x] `num_epochs`=35 (BELOW ~40 — underfit, as flagged; k=6 compute-bound)

### Milestone 4: Verification verdict
- [x] FAIL: best_test_acc 95.26 < 96.10 → no-improvement
- [~] skipped (aborted after metric failure)

## Code Changes
Single change in `train.py` (only editable file). Everything else (Cutout, recipe, projection shortcuts, seed)
held fixed for clean attribution.

- **train.py — `WIDTH_MULT` 4 → 6**: stage widths {64,128,256} → {96,192,384}; param count ~4.3M → ~9.7M.
  *Why*: width has been the dominant accuracy lever (EXP-001: +2.84) and per-step time barely grows with width
  on the H20 (memory/launch-bound: k=1→k=4 only ~16% slower for 16× FLOPs, exp-report-003), so k=6 adds
  capacity at low wall-clock cost while Cutout regularizes the larger model.

## Configuration Changes
- WIDTH_MULT: `4` → `6` (stages {96,192,384}, ~9.7M params)
- ALL else UNCHANGED: NUM_BLOCKS 3, Cutout(16) via cutout_batch, PEAK_LR 0.2, WD 1e-4, label smoothing 0.1,
  batch 128, bf16, channels_last, Nesterov, cosine schedule, MAX_STEPS 10_000_000, seed 42, eval frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0). VRAM expected well under ceiling (~1–1.5 GB; k=4 was 492 MB).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. Expect epochs ~55–70 (per the
  near-flat width/throughput relationship); a larger drop would be informative.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, crash.
- Python traceback / empty `best_test_acc:` at end → crash; inspect `tail -n 50 run.log`.
- No new log output > 2 min while training → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- OOM/CUDA memory error → kill (not expected; would indicate a config issue).

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
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (capacity/epoch trade-off vs k=4's 77)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (confirm width stays cheap on VRAM)
- num_params: `grep -aE "^num_params:" run.log` (confirm ~9.7M)
- final_test_loss: overfit check vs EXP-003's 0.204

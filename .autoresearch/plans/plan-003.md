# Plan EXP-003: Vectorized GPU Cutout (recover throughput)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only: replace per-sample CPU `Cutout` with a batched GPU `cutout_batch`, applied in the loop
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] `uv run ruff check train.py` passes
- [x] Sanity: `uv run python -c "import torch,train; x=torch.ones(4,3,32,32); y=train.cutout_batch(x.clone(),16); print('per-img zeroed:', [int((y[i]==0).any(0).sum()) for i in range(4)])"` prints four counts each ≤256 and > 0

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, params 4,299,866, decreasing loss (no NaN)

### Milestone 3: Run completes within budget and throughput recovered
- [ ] `run.log` contains `best_test_acc:` summary; `total_seconds` < 600
- [ ] `num_epochs` recovered toward EXP-001 level (~75–79, well above EXP-002's 54) — confirms the bottleneck fix

### Milestone 4: Verification verdict
- [ ] `best_test_acc >= 95.52` (baseline 95.42 + 0.1 pp)
- [ ] No constraint violations (only train.py changed, eval ≤once/epoch, no new deps, no seed hacking)

## Code Changes
All changes confined to `train.py`. Model (k=4), recipe, and Cutout *semantics* (one random ≤16×16 hole per
image, zeroed in normalized space) are unchanged — only Cutout's implementation/location changes.

- **train.py — remove CPU Cutout from the pipeline**: Delete the `Cutout(CUTOUT_SIZE)` entry from the training
  `transforms.Compose` and remove the per-sample `Cutout` class (or leave unused — prefer removing to keep the
  file clean). Keep the `CUTOUT_SIZE = 16` constant.

- **train.py — add `cutout_batch` (batched GPU op)**: Add a module-level function:
  ```python
  def cutout_batch(x, size):
      # x: (B,C,H,W) on device; zero one random size×size window per image (clipped to border)
      b, _, h, w = x.shape
      cy = torch.randint(0, h, (b,), device=x.device)
      cx = torch.randint(0, w, (b,), device=x.device)
      y0 = (cy - size // 2).view(b, 1, 1)
      x0 = (cx - size // 2).view(b, 1, 1)
      yy = torch.arange(h, device=x.device).view(1, h, 1)
      xx = torch.arange(w, device=x.device).view(1, 1, w)
      hole = (yy >= y0) & (yy < y0 + size) & (xx >= x0) & (xx < x0 + size)  # (B,H,W)
      return x.masked_fill(hole.unsqueeze(1), 0.0)
  ```
  Negative/over-range `y0/x0` are handled naturally by the arange comparison (equivalent to clipping, matching
  EXP-002's `max(0,·)/min(h,·)` window).

- **train.py — apply in the training loop**: Immediately after `inputs = inputs.to(device, …channels_last)`
  (and before the autocast forward), add `inputs = cutout_batch(inputs, CUTOUT_SIZE)`. Applies to training
  batches only; the eval path (`prepare.py`) is untouched.
  *Why this tests the hypothesis*: identical regularization at ~zero cost (batched GPU mask, no `.item()`
  host-sync, no per-sample Python loop) → restores the ~79 epochs the 300s budget allows → more training of the
  already-better regularized model → higher acc.

## Configuration Changes
- Cutout location: CPU `train_tf` transform (per-sample) → batched GPU op in the training loop (per-batch)
- CUTOUT_SIZE: `16` (unchanged); semantics unchanged (one ≤16×16 zeroed hole per image)
- ALL else UNCHANGED: WIDTH_MULT 4, PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128, bf16, channels_last,
  Nesterov, MAX_STEPS 10_000_000, seed 42, eval frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0). VRAM ~same (~490 MB; the mask is a transient (B,H,W) bool).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. Expect epoch count to recover to ~75–79.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, treat as crash.
- Python traceback / empty `best_test_acc:` at end → crash; inspect `tail -n 50 run.log`.
- No new log output for > 2 min while training → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- num_epochs does NOT recover (still ~54) → not an abort per se, but record it: the masking op may itself be
  costly or a sync remains (note for analysis).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **95.42** (`exp-index.sh baseline`); success bar = **95.52**.

1. **Clean completion within budget** (necessary condition 2):
   - `grep -aE "^best_test_acc:|^total_seconds:" run.log`; `tail -n 50 run.log` for tracebacks.
   - PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback. Timeout: 10 min.
2. **Metric improvement** (necessary condition 1):
   - Parse `best_test_acc`. PASS if `best_test_acc >= 95.52`. FAIL (→ no-improvement) otherwise.
3. **No constraint violations** (necessary condition 3):
   - `git diff --name-only autoresearch/dev` = only `train.py`; no pyproject/uv.lock diff; eval-line count ==
     num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42).
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — KEY signal: expect recovery to ~75–79
  (vs EXP-002's 54), confirming the throughput fix
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- final_test_loss: from summary (compare overfitting vs EXP-002's 0.217)

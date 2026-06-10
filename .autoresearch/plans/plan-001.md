# Plan EXP-001: Widen the ResNet (WideResNet-style, k=4) + projection shortcuts
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only: add `WIDTH_MULT`, widen stages, switch to projection shortcuts
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] `uv run ruff check train.py` passes
- [x] Sanity: `python -c "import train; m=train.ResNet(train.NUM_BLOCKS, width_mult=train.WIDTH_MULT); print(sum(p.numel() for p in m.parameters()))"` prints a param count ~16× the 269,722 baseline (≈4–5M) without error

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, the new (larger) param count, decreasing loss (no NaN)

### Milestone 3: Run completes within budget and summary emitted
- [x] `run.log` contains `best_test_acc:` summary (94.90%); `total_seconds` < 600 (385.7)
- [x] `num_epochs` large enough (79 epochs fit — ample to converge)

### Milestone 4: Verification verdict
- [x] `best_test_acc >= 92.16` (94.90 ≥ 92.16)
- [x] No constraint violations (only train.py changed, eval once/epoch, no new deps, no seed hacking)

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). Architecture-only —
the EXP-000 training recipe (bf16, channels_last, time-fraction cosine, Nesterov, label smoothing, batch 128,
WD 1e-4, PEAK_LR 0.2, seed 42) is held FIXED for clean attribution.

- **train.py — `BasicBlock` projection shortcuts**: Replace the channel-padding identity downsample with a
  standard projection shortcut. When `need_pad` (stride≠1 or in≠out channels), build
  `self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))`;
  otherwise `self.shortcut = nn.Identity()`. In `forward`, `shortcut = self.shortcut(x)`. Removes the
  `pad_channels`/slicing logic. *Why*: 1×1 projection shortcuts suit wider stages and are the standard
  WRN/ResNet-B downsample; the 1×1 conv is learnable capacity vs. lossy zero-pad. (Gets kaiming init via
  the existing `_weights_init`, which now must also cover the shortcut conv — it does, since it matches `nn.Conv2d`.)

- **train.py — `ResNet` width multiplier**: Add `width_mult` arg (default 1). Keep the stem at 16 channels
  (`conv1`: 3→16, `bn1`(16)). Widen the three stages by `k=width_mult`:
  `layer1: 16→16k (stride1)`, `layer2: 16k→32k (stride2)`, `layer3: 32k→64k (stride2)`, `fc: 64k→num_classes`.
  With k=1 this is identical to the current net (16→16, 16→32, 32→64); with k=4 it is {64,128,256}.
  *Why*: more channels = more capacity, the binding ceiling per EXP-000; WRN shows width is the most
  compute-efficient capacity knob under a wall-clock budget.

- **train.py — hyperparameter + call site**: Add `WIDTH_MULT = 4` to the hyperparameter block. Update the
  model construction to `ResNet(NUM_BLOCKS, NUM_CLASSES, width_mult=WIDTH_MULT)`. Print includes the new
  param count (existing print already reports `num_params`). *Why*: k=4 ({64,128,256}) is a well-known
  strong CIFAR width with a large capacity jump; VRAM is free (EXP-000 used 164 MB / 98 GB).

## Configuration Changes
- WIDTH_MULT: (new) `4` — stage widths {16,32,64} → {64,128,256}; ~16× params (≈4–5M, still tiny absolutely)
- Shortcut type: channel-padding identity → 1×1-conv projection + BN (on downsample blocks only)
- All EXP-000 recipe knobs UNCHANGED: PEAK_LR 0.2, WARMUP_FRAC 0.05, MOMENTUM 0.9 + nesterov,
  WEIGHT_DECAY 1e-4, LABEL_SMOOTHING 0.1, BATCH_SIZE 128, bf16 autocast, channels_last, MAX_STEPS 10_000_000, seed 42
- NUM_BLOCKS: `3` (unchanged — depth stays 20; widen, don't deepen)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM expected to stay far below the 98 GB ceiling
  (baseline 164 MB; ~16× activations/params still well within budget).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. Per-step time higher than baseline
  (more FLOPs) → fewer epochs; expected to still fit tens of epochs on H20.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging (rising) for sustained steps after warmup → kill, treat as crash.
- Python traceback in `run.log` (empty `best_test_acc:` at end) → crash; inspect `tail -n 50 run.log`.
- No new output in `run.log` for > 2 min while training should be active → kill (hang).
- Total wall-clock > 10 min → kill, treat as failure.
- OOM / CUDA memory error → kill (unexpected given headroom; would indicate a config bug).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **92.06** (`exp-index.sh baseline`); success bar = **92.16**.

1. **Clean completion within budget** (necessary condition 2):
   - `grep -aE "^best_test_acc:|^total_seconds:" run.log` and `tail -n 50 run.log` for tracebacks.
   - PASS if `best_test_acc:` present (non-empty), `total_seconds < 600`, no traceback. Timeout: 10 min.
2. **Metric improvement** (necessary condition 1):
   - Parse `best_test_acc`. PASS if `best_test_acc >= 92.16`. FAIL (→ no-improvement) otherwise.
3. **No constraint violations** (necessary condition 3):
   - `git diff --name-only autoresearch/dev` shows only `train.py`; no diff on `pyproject.toml`/`uv.lock`;
     eval-line count == num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42).
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (VRAM headroom used by the wider model)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (capacity/throughput tradeoff vs
  EXP-000's 109 / 42,156 — key signal for whether width was worth the epoch cost)
- num_params: `grep -aE "^num_params:" run.log` (confirm the ~16× capacity increase)

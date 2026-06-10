# Plan EXP-002: Cutout augmentation (16×16) on the k=4 WideResNet
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only: add a `Cutout` transform + `CUTOUT_SIZE`, append to `train_tf`
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] `uv run ruff check train.py` passes
- [x] Sanity: `uv run python -c "import torch,train; t=train.Cutout(16); x=torch.ones(3,32,32); y=t(x.clone()); print('zeroed px:', int((y==0).any(0).sum().item()))"` prints a nonzero count ≤ 256 (a ≤16×16 region zeroed)

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Within ~60s: `run.log` shows `Device: cuda`, params 4,299,866 (unchanged), decreasing loss (no NaN)

### Milestone 3: Run completes within budget and summary emitted
- [x] `run.log` contains `best_test_acc:` summary (95.42%); `total_seconds` < 600 (367.6)
- [x] `num_epochs`=54 (LOWER than expected ~79 — Cutout impl became a dataloader CPU bottleneck; still improved)

### Milestone 4: Verification verdict
- [x] `best_test_acc >= 95.00` (95.42 ≥ 95.00)
- [x] No constraint violations (only train.py changed, eval once/epoch, no new deps, no seed hacking)

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The k=4 WideResNet
(EXP-001) and the full training recipe (bf16, channels_last, cosine, Nesterov, label smoothing, batch 128,
WD 1e-4, PEAK_LR 0.2, WIDTH_MULT 4, seed 42) are held FIXED — only the training augmentation changes.

- **train.py — `Cutout` transform**: Add a small callable class that, given a normalized CxHxW tensor, zeros
  one random square of side `CUTOUT_SIZE` (centered at a random pixel, clipped to the image border). Use
  `torch.randint` for the center coordinates so the randomness is covered by the existing `torch.manual_seed(42)`
  (deterministic, not seed hacking). Zeroing in normalized space (std=1, mean subtracted) sets the region to the
  dataset mean — the standard Cutout fill.
  ```python
  class Cutout:
      def __init__(self, size):
          self.size = size
      def __call__(self, img):  # img: CxHxW tensor
          _, h, w = img.shape
          s = self.size
          cy = int(torch.randint(0, h, (1,)).item())
          cx = int(torch.randint(0, w, (1,)).item())
          y1, y2 = max(0, cy - s // 2), min(h, cy + s // 2)
          x1, x2 = max(0, cx - s // 2), min(w, cx + s // 2)
          img[:, y1:y2, x1:x2] = 0.0
          return img
  ```
- **train.py — append to `train_tf`**: Add `Cutout(CUTOUT_SIZE)` as the LAST element of the training
  `transforms.Compose` (after `ToTensor` + `Normalize`, so it operates on the normalized tensor). The eval
  transform in `prepare.py` is untouched (frozen) — Cutout applies to training only.
  *Why this tests the hypothesis*: input-space regularization reduces overfitting of the high-capacity k=4
  model, the now-likely ceiling; it adds ~0 compute and 0 params so the epoch budget is preserved.

- **train.py — hyperparameter**: Add `CUTOUT_SIZE = 16` to the hyperparameter block (standard CIFAR-10 hole size).

## Configuration Changes
- CUTOUT_SIZE: (new) `16` — one 16×16 zeroed square per training image (DeVries & Taylor CIFAR-10 default)
- Training augmentation: RandomCrop(4)+HFlip+Normalize → + Cutout(16) appended
- ALL else UNCHANGED: WIDTH_MULT 4, PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128, bf16, channels_last,
  Nesterov, MAX_STEPS 10_000_000, seed 42, eval transform frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0). VRAM ~same as EXP-001 (~490 MB).
- Estimated runtime: ~300s training + ~60–80s startup/eval ≈ 6–8 min. Cutout is a cheap CPU tensor op in the
  dataloader workers (NUM_WORKERS=8) — negligible impact; epoch count expected ~same as EXP-001 (~79).
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging (rising) for sustained steps after warmup → kill, treat as crash.
- Python traceback in `run.log` (empty `best_test_acc:` at end) → crash; inspect `tail -n 50 run.log`.
- No new output in `run.log` for > 2 min while training should be active → kill (hang).
- Total wall-clock > 10 min → kill, treat as failure.

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **94.90** (`exp-index.sh baseline`); success bar = **95.00**.

1. **Clean completion within budget** (necessary condition 2):
   - `grep -aE "^best_test_acc:|^total_seconds:" run.log` and `tail -n 50 run.log` for tracebacks.
   - PASS if `best_test_acc:` present (non-empty), `total_seconds < 600`, no traceback. Timeout: 10 min.
2. **Metric improvement** (necessary condition 1):
   - Parse `best_test_acc`. PASS if `best_test_acc >= 95.00`. FAIL (→ no-improvement) otherwise.
3. **No constraint violations** (necessary condition 3):
   - `git diff --name-only autoresearch/dev` shows only `train.py`; no diff on `pyproject.toml`/`uv.lock`;
     eval-line count == num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42).
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (confirm Cutout didn't cut throughput;
  expect ~79 epochs as EXP-001)
- final_test_loss: from the summary (Cutout should reduce overfitting; compare to EXP-001's 0.249)

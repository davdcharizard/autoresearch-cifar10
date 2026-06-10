# Plan EXP-048: GridMask occlusion — swap Cutout's single hole for a distributed grid (matched strength)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md

## Milestones

### Milestone 1: gridmask_batch implemented, matched-strength, dt-safe by construction
- [ ] Add a `gridmask_batch(x, d_min, d_max, mask_ratio)` GPU-vectorized function mirroring `cutout_batch`'s style (coordinate-grid mask + `masked_fill`, seeded torch RNG on GPU, no `.item()` syncs).
- [ ] Add hyperparameters `GRIDMASK_D_MIN = 8`, `GRIDMASK_D_MAX = 16`, `GRIDMASK_RATIO = 0.5` (removed-square side = ratio×period → removed-area ≈ ratio² ≈ 0.25, matching Cutout-16's ~25% so this isolates PATTERN from strength).
- [ ] Swap the training-loop call `cutout_batch(inputs, CUTOUT_SIZE)` → `gridmask_batch(inputs, GRIDMASK_D_MIN, GRIDMASK_D_MAX, GRIDMASK_RATIO)`. Leave `cutout_batch` defined (now unused) for baseline reference.
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` parses; confirm the swap is the only training-path change, recipe/optimizer/schedule/seed/compile-mode unchanged.

### Milestone 2: Run launched on idle GPU and confirmed healthy
- [ ] `nvidia-smi` → pick an idle GPU (util ~0%, mem <700MiB); shared node — foreign jobs intermittently occupy a GPU.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in background.
- [ ] Confirm within ~90s: banner `ResNet-20 | params: 4,299,866`, loss falling normally (ep1 ~40-46%), no traceback.

### Milestone 3: dt / epoch verified — throughput-neutrality
- [ ] Extract dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`. Confirm steady **8ms** (GridMask is the same vectorized-mask op class as Cutout → expect no dt change) and `num_epochs` ≈ ~91. A dt rise would indicate an unexpected sync/graph issue → discard as confounded.

### Milestone 4: Accuracy verified against baseline
- [ ] Extract `best_test_acc`; compare to bar 96.32 (baseline 96.22 + 0.1).

## Code Changes
- **train.py — new `gridmask_batch` (inserted near `cutout_batch`, ~L58)**:
  ```python
  def gridmask_batch(x, d_min, d_max, mask_ratio):
      """GridMask (Chen et al. 2020), vectorized on the GPU batch: delete a periodic
      grid of squares per image — one random grid period d and (offset_y,offset_x) per
      image, with removed-square side = round(mask_ratio*d) so removed-area ≈ mask_ratio²
      (=~25% at 0.5, matched to Cutout-16). Train-only regularizer using the seeded torch
      RNG; no per-sample CPU `.item()` syncs (cf. EXP-002), so it does not throttle the
      dataloader. Axis-aligned (no rotation) to stay a single static-shape vectorized op."""
      b, _, h, w = x.shape
      d = torch.randint(d_min, d_max + 1, (b,), device=x.device)            # (b,) period
      lm = (d.float() * mask_ratio).round().long().clamp(min=1)             # removed-square side
      oy = (torch.rand(b, device=x.device) * d.float()).long()             # per-image grid offset
      ox = (torch.rand(b, device=x.device) * d.float()).long()
      yy = torch.arange(h, device=x.device).view(1, h, 1)                  # (1,h,1)
      xx = torch.arange(w, device=x.device).view(1, 1, w)                  # (1,1,w)
      dv, lmv = d.view(b, 1, 1), lm.view(b, 1, 1)
      oyv, oxv = oy.view(b, 1, 1), ox.view(b, 1, 1)
      row_in = ((yy - oyv) % dv) < lmv                                     # (b,h,1)
      col_in = ((xx - oxv) % dv) < lmv                                     # (b,1,w)
      hole = row_in & col_in                                               # (b,h,w) grid of squares
      return x.masked_fill(hole.unsqueeze(1), 0.0)
  ```
  - **Why it tests the hypothesis**: replaces single-hole occlusion (Cutout-16) with distributed grid-of-squares occlusion at matched ~25% removed-area → isolates occlusion PATTERN (the one untested augmentation sub-lever) from strength. Mirrors the only plateau-breaker's mechanism (substitute a more effective augmentation at zero convergence cost).
  - **Risks/edge cases**: `torch.remainder` (`%`) on int tensors is non-negative for positive divisor (sign-of-divisor), so `(yy-oyv) % dv ∈ [0,dv)` even when the offset exceeds the coordinate — correct. `mask_ratio=0.5` → removed-side = d/2 → removed-area ≈ 25% (infinite-periodic limit; finite 32×32 with offset ≈ 20-28%, a fair Cutout match). Static shapes (batch 128 fixed) → CUDA-graph-safe; same op class as `cutout_batch` → dt expected 8ms (verify M3). Seed-clean: uses the existing seeded GPU RNG (same as Cutout); the RNG-stream change vs Cutout is the intended augmentation change, not seed hacking.
- **train.py — hyperparameters**: add `GRIDMASK_D_MIN = 8`, `GRIDMASK_D_MAX = 16`, `GRIDMASK_RATIO = 0.5` near `CUTOUT_SIZE`.
- **train.py — training-loop call**: swap `inputs = cutout_batch(inputs, CUTOUT_SIZE)` → `inputs = gridmask_batch(inputs, GRIDMASK_D_MIN, GRIDMASK_D_MAX, GRIDMASK_RATIO)`.
- **NO other changes** — TA + crop + flip, optimizer (SGD+Nesterov), schedule (time-fraction cosine peak 0.2), LS, WD, seed 42, batch 128, `mode="reduce-overhead"` all UNCHANGED. Param count unchanged (4,299,866).

## Configuration Changes
- New augmentation params `GRIDMASK_D_MIN=8`, `GRIDMASK_D_MAX=16`, `GRIDMASK_RATIO=0.5` (rationale: d∈[8,16] gives distributed 4-8px square holes with 2-4 periods across 32px; ratio 0.5 → ~25% removed-area, matched to Cutout-16 to isolate pattern from strength).
- Cutout effectively removed from the train path (replaced by GridMask). No other recipe/hyperparameter changes.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background; harness re-invokes on completion.
- Resources: single idle NVIDIA H20 (shared node, idx 0/1; a foreign job is intermittently on GPU 0 — pick the idle GPU). VRAM trivial. Fixed 300s training budget.
- Estimated runtime: ~6-7 min wall (compile ~5s startup + 300s training + per-epoch evals). Must be < 10 min.
- Log output: `run.log` in project root. dt lines use `\r` — extract via `tr '\r' '\n'`.
- Tool skill: none (local).

## Abort Criteria
- Loss diverges (NaN/inf) or fails to fall below ~1.0 in the first few epochs.
- Traceback / shape error (esp. broadcasting error in the mask construction) → fix (code error, single retry).
- dt steady-state ≫ 8ms while GPU idle → unexpected sync/graph issue → discard as confounded, debug the mask op.
- GPU contention mid-run (dt ≫ 8ms with a foreign process co-resident, per infra-errors.md) → discard as contention-confounded, rerun on idle GPU.
- No `dt:`/epoch-eval output after ~120s (silent hang).
- Total wall-clock approaches 10 min without summary.

## Verification Protocol

### Verification Procedure
Baseline (from experiment index) = **96.22%**; bar = **96.32%** (baseline + 0.1).

1. **Run completes cleanly within budget** — `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:|^num_epochs:|^num_steps:|^peak_vram_mb:|^num_params:" run.log`. Pass: `best_test_acc` present/non-empty, `total_seconds` < 600, `training_seconds` ≈ 300, `num_params` = 4,299,866 (unchanged). Empty `best_test_acc` ⇒ crash (`tail -n 50 run.log`). Run timeout: 600s wall.
2. **Throughput-neutrality** — dt distribution via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`; confirm steady 8ms and `num_epochs` ≈ ~91 (a clean fair test vs the 91-ep baseline).
3. **Primary necessary condition** — `grep -aE "^best_test_acc:" run.log`. Pass iff `best_test_acc ≥ 96.32`.
4. **No hard-constraint violations** — `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (GridMask uses only torch); seed 42 unchanged; no seed hacking (deterministic mask math on the existing seeded RNG).
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirms throughput-neutrality (~91 ep).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (expect ≈ baseline ~491).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — does the occlusion-pattern change move loss even if top-1 is flat?

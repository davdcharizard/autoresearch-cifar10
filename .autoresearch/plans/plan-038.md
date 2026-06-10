# Plan EXP-038
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md
**Brainstorm**: brainstorm/brainstorm-038.md

## Summary
Compute-neutral **fat-head width reallocation**: change the per-stage channel widths from the
uniform `{w1,w2,w3} = {64,128,256}` (k=4) to **`{44,128,320}`**, narrowing the spatially-expensive
stage1 (32×32) to fund a wider spatially-cheap discriminative stage3 (8×8) at ≈constant total FLOPs.
This adds net effective capacity (448→492 channels, concentrated in the stage feeding the
classifier) WITHOUT the epoch-wall under-training that killed uniform widening (EXP-004/009). Tested
against the bar **96.32** (baseline 96.22 + 0.1).

## Baseline (from experiment index)
- best_test_acc baseline = **96.22%** (commit 6c417a4, EXP-012); bar = **96.32**.
- Reference run shape: ~91 epochs, dt ~8ms/step, final_test_loss ~0.195, ~393–421s wall,
  params 4,299,866.

## Hypothesis
Reallocating capacity from stage1 (64→44 ch) to the discriminative stage3 (256→320 ch) at constant
FLOPs (≈+0.4%, per-stage FLOPs ∝ w²·area for areas {1024,256,64}) adds net capacity where it most
serves classification, lifting best_test_acc above 96.32 WITHOUT the epoch-wall regression of uniform
widening — at throughput-neutral ~91 ep. Honest most-likely: within-noise (~96.0–96.3) if the
capacity bound is global rather than stage3-local, or mild regression if narrowing stage1 starves
early features. Clean compute-neutral failure mode.

## FLOP / compute-neutrality check (why this dodges the High compute wall)
Dominant 3×3 conv FLOPs per stage ≈ (5·w² + in_prev·w)·area, areas S1=1024, S2=256, S3=64:
- Baseline {64,128,256}: 22.0 + 23.1 + 23.1 = **68.2** units
- Fat-head {44,128,320}: 10.6 + 22.4 + 35.4 = **68.4** units → **+0.4%** (effectively neutral)

This plan does NOT contradict the project-insights HIGH compute-wall entry: that entry concerns
FLOP-**adding** changes (uniform k=5/k=6 added +56–125% FLOPs → 41–65 ep → under-train). This change
is FLOP-**neutral** by construction and pursues the Medium-importance prescription ("top-1 gains here
require capacity, not polish") via the only un-walled route. The realized epoch count is the
empirical check (Milestone 3 / abort criteria): must stay ≈baseline (≥~88 ep); a material drop means
the FLOP estimate was wrong and the result is compute-confounded.

## Milestones

### Milestone 1 — Code change implemented and passing local checks
- [ ] Edit `train.py` L100-101: replace `k = width_mult` / `w1, w2, w3 = 16*k, 32*k, 64*k` with
      explicit `w1, w2, w3 = 44, 128, 320` (+ explanatory comment).
- [ ] AST check: `uv run python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
- [ ] Model instantiates + report params/widths: `uv run python -c "import train; m=train.ResNet(3,10);
      print('params', sum(p.numel() for p in m.parameters())); print('w3 fc', m.fc.in_features)"`
      → expect fc.in_features == 320, params != 4,299,866 (will increase).
- [ ] Diff scope check: `git diff --name-only` lists **only** `train.py`.

### Milestone 2 — Experiment running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background.
- [ ] Confirm `run.log` shows `Device: cuda`, a `params:` line (new count), first eval line appears.

### Milestone 3 — Run completed and verified
- [ ] Run exits 0, prints full summary block (`best_test_acc:` … `num_params:`).
- [ ] **Compute-neutrality check**: `num_epochs >= ~88` and dt ~8ms (confirms FLOP-neutral; if
      epochs dropped materially the result is compute-confounded — note in analysis).
- [ ] Extract metrics, compare to bar 96.32 / baseline 96.22.
- [ ] Confirm clean completion (<600s wall, eval_count == num_epochs, only train.py changed, seed 42).

## Code Changes

**File: `train.py` (L100-101, the ONLY change)**
- **What**: replace the uniform width derivation
  ```python
  # before
  k = width_mult
  w1, w2, w3 = 16 * k, 32 * k, 64 * k
  # after
  # Fat-head width reallocation (EXP-038): narrow the spatially-expensive stage1 (32×32) and
  # widen the spatially-cheap discriminative stage3 (8×8) at ~constant FLOPs (per-stage FLOPs
  # ∝ w²·area). {64,128,256}→{44,128,320}: +44 net channels, ≈+0.4% FLOPs, ~91 ep expected.
  w1, w2, w3 = 44, 128, 320
  ```
- **Why it tests the hypothesis**: isolates capacity *placement* — total FLOPs (hence epochs) held
  constant, only the channel distribution changes, so any metric delta is attributable to where
  capacity sits, not to under-training. The `_make_layer`/`BasicBlock` projection shortcuts adapt to
  arbitrary widths (train.py L80-84, L104-107) and `fc = nn.Linear(w3, num_classes)` (L107) +
  `adaptive_avg_pool2d` (L131) handle the wider stage3 with no other edit.
- **Risks/edge cases**: (a) narrowing stage1 64→44 (−31% early-feature width, still 2.75× the
  original ResNet-20's 16ch) could starve early feature extraction → within-noise null or mild
  regression; (b) `width_mult`/`WIDTH_MULT` becomes unused — harmless (no lint gate in this repo;
  the `ResNet(..., width_mult=WIDTH_MULT)` call still passes, arg simply ignored). No new deps, no
  eval-side change (prepare.py untouched), seed unchanged.

## Configuration Changes
Widths only (`{64,128,256}→{44,128,320}`). All other hyperparameters unchanged (PEAK_LR 0.2, batch
128, WD 1e-4, label smoothing 0.1, Cutout 16, TrivialAugmentWide + reflect-default crop, cosine-to-0
LR, Nesterov m0.9, seed 42, 300s budget, torch.compile reduce-overhead). Widths chosen to hold FLOPs
within ~±1% of baseline (see FLOP check) so the test is compute-neutral.

## Execution Environment
- **Method**: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, `run_in_background: true`.
- **Resources**: single NVIDIA H20; fixed `TIME_BUDGET_S=300` training compute.
- **Estimated runtime**: ~393–421s wall (≈6.5–7 min), well under the 10-min ceiling.
- **Log output**: stdout+stderr → `run.log` at project root (sole source of truth). Per-step lines
  use `\r`; extract dt via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms"`.
- **Monitoring**: `Monitor` on `tail -f run.log` filtered to errors + final summary block only (not
  per-epoch evals) to limit notification noise; background task notifies on exit.

## Abort Criteria
- Any Python traceback / non-zero exit, or NaN/inf in `loss:` → kill, mark failed.
- No `run.log` growth or no first eval line within ~120s of launch → kill, inspect.
- Total wall-clock approaching 10 min (600s) → kill, treat as failure.
- dt materially > ~9.5ms / epoch count trending well below ~85 → NOT an abort (let it finish), but
  flag in analysis as compute-confounded (the FLOP estimate was wrong).

## Verification Protocol

### Verification Procedure
Run after the experiment completes. Baseline = 96.22 (from `exp-index.sh baseline`).

1. **Primary metric clears the bar** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:" run.log`
   - Pass iff `best_test_acc >= 96.32`. Else no-improvement.
2. **Clean completion within budget** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_params:" run.log`
   - Pass iff summary block present, `total_seconds < 600`, run exited 0.
3. **No hard-constraint violations** (NECESSARY):
   - `git diff --name-only` shows only `train.py`; eval-line count == `num_epochs:` (≤1 eval/epoch);
     no new imports/deps; seed unchanged (42); prepare.py/eval untouched. (num_params is EXPECTED to
     change — capacity is not a constraint; only train.py-only + budget + ≤1 eval/epoch are.)
   - Timeout per command: 30s. Overall run timeout: 600s wall.

### Informational Metrics (Optional)
- `peak_vram_mb:` — VRAM (expect modest increase from wider stage3; soft constraint, fine).
- `num_epochs:` / `num_steps:` — **compute-neutrality check** (expect ~88–91 ep / dt ~8ms; a drop to
  <~85 signals the reallocation was not FLOP-neutral → compute-confounded result).
- `final_test_loss:` — convergence check (expect ≈0.195 if converged; higher → under-train).

## Expected Outcome / Decision
- **If `best_test_acc >= 96.32`**: improvement — commit, merge to `autoresearch/dev`, PR to main.
- **If within-noise (~96.0–96.3) or below, at ~91 ep**: no-improvement — teaches that the capacity
  bound is GLOBAL (not reallocatable to stage3); closes the fat-head sub-lever. Discard, record.
- **If regressed with epochs dropped <~85**: compute-confounded — record that the widths were not
  FLOP-neutral; a gentler reallocation could be a follow-up (not auto-scheduled).

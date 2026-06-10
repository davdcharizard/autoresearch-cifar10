# Plan EXP-037
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md
**Brainstorm**: brainstorm/brainstorm-037.md

## Summary
Single-argument augmentation-quality change: switch the train-only `RandomCrop` from
default zero-padding (`padding_mode='constant'`) to `padding_mode='reflect'`. Reflect
the 4-px crop border instead of zero-filling it, so translated training crops carry
natural edge statistics rather than artificial black wedges. Compute-neutral (CPU PIL
op), params unchanged (4,299,866), train-only (eval/test_tf does not crop → no
train/eval mismatch). Tested against the bar **96.32** (baseline 96.22 + 0.1).

## Baseline (from experiment index)
- best_test_acc baseline = **96.22%** (commit 6c417a4, EXP-012); bar = **96.32**.
- Reference run shape: ~91 epochs, dt ~8ms/step, final_test_loss ~0.195, ~393–405s wall.

## Hypothesis
Reflect-padding removes the black-border artifact zero-padding injects into every
cropped training image, tightening the train/test distribution match → marginal
best_test_acc gain. Honest most-likely outcome: within-noise (~96.1–96.3), since the
net is regularization-saturated and BN may absorb the thin-border effect. Throughput-
neutral → ~91 epochs, params unchanged.

## Milestones

### Milestone 1 — Code change implemented and passing local checks
- [ ] Edit `train.py` L158: `transforms.RandomCrop(32, padding=4)` →
      `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`
- [ ] AST/import sanity: `uv run python -c "import ast; ast.parse(open('train.py').read())"`
- [ ] Diff scope check: `git diff --name-only` lists **only** `train.py`
- [ ] Confirm the diff is exactly the one-argument addition (no other lines touched)

### Milestone 2 — Experiment running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [ ] Confirm `run.log` shows `Device: cuda`, `params: 4,299,866`, first eval line appears

### Milestone 3 — Run completed and verified
- [ ] Run exits 0, prints summary block (`best_test_acc:` … `num_params:`)
- [ ] Extract metrics and compare to bar 96.32 / baseline 96.22
- [ ] Confirm clean completion (<600s wall, eval_count == num_epochs, params 4,299,866, seed 42)

## Code Changes

**File: `train.py` (L158, the ONLY change)**
- **What**: add `padding_mode="reflect"` argument to the existing `RandomCrop`.
  ```python
  # before
  transforms.RandomCrop(32, padding=4),
  # after
  transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
  ```
- **Why it tests the hypothesis**: this isolates augmentation *quality* (border
  statistics) from strength/policy — same crop size, same padding width, same op
  count. The only difference is what fills the 4-px border before the random crop.
- **Risks/edge cases**: none functional — `reflect` is a stock torchvision
  `RandomCrop` mode (32×32 image, 4-px pad is well within reflect's size limit, which
  requires pad < dimension). No GPU sync, no FLOP change, no new dependency. Eval is
  untouched (test_tf does not crop), so no train/eval normalization or shape mismatch.

## Configuration Changes
None. All hyperparameters (PEAK_LR 0.2, batch 128, WD 1e-4, label smoothing 0.1,
Cutout 16, TrivialAugmentWide, cosine-to-0 LR, Nesterov m0.9, seed 42, 300s budget,
`torch.compile` reduce-overhead) are unchanged.

## Execution Environment
- **Method**: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`,
  `run_in_background: true`.
- **Resources**: single NVIDIA H20; fixed `TIME_BUDGET_S=300` training compute.
- **Estimated runtime**: ~393–405s wall (≈6.5–7 min), well under the 10-min ceiling.
- **Log output**: stdout+stderr → `run.log` at project root (sole source of truth).
  Per-step lines use `\r`; extract dt via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms"`.
- **Monitoring**: `Monitor` tool on `tail -f run.log` filtered to errors + eval summary
  lines + final summary block (not every step) to limit notification noise.

## Abort Criteria
- Any Python traceback / non-zero exit, or NaN/inf in `loss:` → kill, mark failed.
- No `run.log` growth or no first eval line within ~120s of launch → kill, inspect.
- Total wall-clock approaching 10 min (600s) → kill, treat as failure.
- `params:` != 4,299,866 in the startup line → kill (would indicate an unintended edit).

## Verification Protocol

### Verification Procedure
Run after the experiment completes. Baseline = 96.22 (from `exp-index.sh baseline`).

1. **Primary metric clears the bar** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:" run.log`
   - Pass iff `best_test_acc >= 96.32` (baseline 96.22 + 0.1). Else no-improvement.
2. **Clean completion within budget** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_params:" run.log`
   - Pass iff summary block present, `total_seconds < 600`, run exited 0.
3. **No hard-constraint violations** (NECESSARY):
   - `git diff --name-only` shows only `train.py`.
   - `num_params:` == 4,299,866; eval-line count == `num_epochs:` (≤1 eval/epoch);
     no new imports/deps; seed unchanged (42). prepare.py/eval untouched.
   - Timeout per command: 30s. Overall run timeout: 600s wall.

### Informational Metrics (Optional)
- `peak_vram_mb:` — VRAM headroom (expect ≈ baseline; aug change is CPU-side).
- `num_epochs:` / `num_steps:` — throughput-neutrality check (expect ~91 ep / dt ~8ms;
  a drop would indicate an unexpected dataloader slowdown from reflect-padding).
- img/s from step lines — efficiency cross-check.

## Expected Outcome / Decision
- **If `best_test_acc >= 96.32`**: improvement — commit, merge to `autoresearch/dev`, PR to main.
- **If within-noise (~96.1–96.3) or below**: no-improvement — closes the crop-padding-mode
  sub-lever; discard changes, record in goal-learnings (augmentation-quality axis probed).
- Also record (analyze phase) the input-std-normalization infeasibility finding (frozen
  eval pins std=(1,1,1)) into goal-learnings Protocol Findings so it is not re-proposed.

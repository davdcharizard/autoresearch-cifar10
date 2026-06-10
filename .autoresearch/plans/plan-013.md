# Plan EXP-013: Reduce Cutout hole size 16→8px under the TA+compile recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Change `CUTOUT_SIZE = 16` → `CUTOUT_SIZE = 8` in train.py (single constant).
- [ ] `uv run ruff check train.py` clean; `git diff` shows ONLY the CUTOUT_SIZE line changed (TA + compile from
      EXP-012 baseline untouched).

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED), no traceback, no NaN, compile completes.
- [ ] Read steady-state `dt`/`img/s` from step ~50–500 — expect ~8ms/step (≈ EXP-012; Cutout size doesn't change cost).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params` from run.log.

## Code Changes
- **train.py** (the ONLY editable file): change the module-level constant `CUTOUT_SIZE = 16` to `CUTOUT_SIZE = 8`.
  - **Why this tests the hypothesis**: TrivialAugment (added EXP-012) raised total augmentation strength; the 16px
    Cutout hole was tuned WITHOUT auto-aug (EXP-002/003). A smaller 8px hole reduces total regularization while
    keeping orthogonal occlusion, testing whether the occlusion sweet spot shifted down under TA and whether less
    over-regularization improves convergence within the 300s budget.
  - **Risks/edge cases**: (a) two-sided — if 16px was already optimal under TA, 8px slightly under-regularizes
    (small loss); (b) the +0.1 bar over 96.22 (→ ≥96.32) is demanding and the delta may fall in the ~0.2pp noise
    band → corroborate any gain with final_test_loss (should drop below 0.195) and the late-eval cluster, not a lone
    best epoch. No throughput/param/scope risk (it's a single int; the cutout_batch op is unchanged in structure).

## Configuration Changes
- `CUTOUT_SIZE`: 16 → 8 (rationale: co-tune occlusion strength with the newly-added TrivialAugment; standard
  practice is that optimal Cutout size shrinks when combined with strong auto-augmentation — brainstorm-013 §
  Web/Lit Review, knowledge/papers/trivialaugment.md).
- No other change: k=4, batch 128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, seed 42, TrivialAugmentWide,
  torch.compile(reduce-overhead) — all inherited from the EXP-012 baseline (commit 6c417a4).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM; 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (< 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or divergence → kill.
- Traceback / process exit ≠ 0 at startup → kill, fix, single retry.
- `num_params` ≠ 4,299,866 at startup → unexpected (this change must not touch params) → kill, investigate.
- No log progress for > ~120s after startup → kill (silent hang).
- NOTE: epoch count is expected ~89–91 (≈ EXP-012; Cutout size doesn't affect throughput) — a low count would be a
  surprise to investigate, not a planned outcome here.

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (`exp-index.sh baseline`); success bar = **96.32** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a value
   AND `total_seconds < 600`, AND `grep -ac "Traceback" run.log` == 0. Pass = all hold. (Timeout: 600s.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.32` (baseline 96.22
   + 0.1). FAIL → verdict no-improvement. (Decisive condition.)
3. **Cond 3 — no constraint violations** (only if Cond 2 passes): `git diff --name-only` lists ONLY `train.py`;
   `num_params == 4,299,866` (unchanged); seed 42 intact; eval-line count == `num_epochs` (eval once/epoch).

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~89–91 ⇒ fair converged test).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — the key corroborator; should drop below EXP-012's 0.195
  if reduced occlusion improved the fit. If acc is flat/down AND loss ≈ 0.195+, the 16px hole was already optimal.
- `img/s` & `dt`: from step ~50–500 — confirm ~8ms/step (≈ EXP-012; rules out any throughput confound).

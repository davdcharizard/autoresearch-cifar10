# Plan EXP-017: LR-schedule micro-tuning — lower peak LR 0.2 → 0.15 (sign-corrected probe)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Change `PEAK_LR` from `0.2` to `0.15` in train.py (line 23). No other change.
- [ ] `uv run ruff check train.py` clean; `git diff` shows ONLY the `PEAK_LR` constant line changed.

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED — hyperparameter-only change), clean compile, no
      traceback, no NaN.
- [ ] Confirm warmup peaks at the new LR: the printed `lr:` should ramp toward ~0.15 during the first ~5% of the
      budget, then begin the cosine descent. Steady-state `dt` ~8ms/step (~84–91 epochs expected — compute-neutral,
      fair same-budget test).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params`, `peak_vram_mb`.

## Code Changes
- **train.py** (the ONLY editable file):
  1. Line 23: `PEAK_LR = 0.2` → `PEAK_LR = 0.15`. Update the inline comment to note the EXP-017 retune (down).
  - **Why this tests the hypothesis**: `PEAK_LR` feeds `lr_at_fraction` (L35-41, warmup+cosine schedule) and the
    optimizer's initial `lr` (L194). EXP-016 showed peak 0.3 regressed (optimum ≤ 0.2); 0.15 probes whether the
    optimum lies BELOW 0.2 — a gentler peak that may settle into a better-generalizing minimum within the budget.
    Schedule SHAPE (5% warmup → cosine-to-0) unchanged; only amplitude scales.
  - **Risks/edge cases**: (a) the optimum may simply BE 0.2 → null; (b) a lower peak could under-progress within
    the fixed ~84–91-epoch budget → slight regression. Both graceful (baseline 96.22 holds). Throughput/params
    unchanged (pure hyperparameter change) → fair test, no confound.

## Configuration Changes
- `PEAK_LR`: 0.2 → 0.15 (peak of the linear-warmup→cosine schedule). Rationale: EXP-016 established the LR optimum
  is ≤ 0.2 (peak 0.3 regressed to 95.77); 0.15 is the maximum-likelihood optimum location if it lies below 0.2 — a
  modest step from the known-good 0.2 toward the textbook batch-128 WRN peak (0.1). All else inherited from the
  EXP-012 baseline (k=4, batch 128, WARMUP_FRAC 0.05, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment,
  torch.compile, seed 42, commit 6c417a4).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM; 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (< 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:|^peak_vram_mb:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or sustained divergence → kill.
- Traceback / process exit ≠ 0 at startup → kill, fix, single retry.
- `num_params` ≠ 4,299,866 → unexpected for a hyperparameter-only change (signals an accidental structural edit) →
  kill, investigate.
- Early-accuracy sanity: if `test_acc` is still < ~90% past ~50% of the budget (≈ ep 45) → suspect a problem → note
  for analysis (let it finish unless NaN/stall).
- No log progress for > ~120s after startup → kill (silent hang).

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (`exp-index.sh baseline`); success bar = **96.32** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a
   value AND `total_seconds < 600`, AND `grep -ac "Traceback" run.log` == 0. Pass = all hold. (Timeout: 600s.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.32`. FAIL →
   verdict no-improvement. (Decisive condition.)
3. **Cond 3 — no constraint violations** (only if Cond 2 passes): `git diff --name-only` lists ONLY `train.py`;
   the diff is the single `PEAK_LR` line; seed 42 intact; eval-line count == `num_epochs` (eval once/epoch);
   `num_params` == 4,299,866 (unchanged); no new deps.

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~84–91; compute-neutral).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — corroborator vs EXP-012's 0.195 (< 0.195 with acc↑
  ⇒ lower LR improved generalization; ≈ 0.195 with flat acc ⇒ peak 0.2 already optimal).
- `peak_vram_mb`: `grep -a "^peak_vram_mb:" run.log` — soft-constraint awareness (expect ~454 MB, unchanged).
- `img/s` & `dt`: step ~400–500 — confirm ~8ms/step to rule out a throughput confound.

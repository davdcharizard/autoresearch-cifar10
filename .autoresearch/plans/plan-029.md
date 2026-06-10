# Plan EXP-029: SGDR — cosine annealing with warm restarts (2 cycles)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Add `N_CYCLES = 2` constant near the other hyperparameters (train.py ~L24, after WARMUP_FRAC).
- [ ] Rewrite `lr_at_fraction(frac)` (train.py L35-41) to a 2-cycle SGDR schedule: split [0,1] into N_CYCLES equal cycles; within the current cycle apply cosine PEAK_LR→~0; warmup over WARMUP_FRAC of the FIRST cycle only; restarts jump straight to PEAK_LR (no warmup).
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] Smoke check (`uv run python`): import `lr_at_fraction`, assert the schedule shape at key fractions — `lr(0.0)==0` (warmup start), `lr(0.025)≈PEAK` (end of first-cycle warmup = WARMUP_FRAC×cycle_len = 0.05×0.5), `lr(0.012)≈0.5*PEAK` (mid-warmup), `lr(0.499)≈0` (end of cycle 1), `lr(0.5)≈PEAK` (restart), `lr(1.0)≈0` (end of cycle 2); all values in [0, PEAK]. Also build `ResNet(3,10,4)` and assert params == 4,299,866 (unchanged — schedule-only change).

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile, step lines, no NaN. The printed `lr:` should rise during early warmup then fall (cosine) — and is expected to JUMP back up near the 50%-budget mark (the restart) — a live confirmation SGDR is active.

### Milestone 3: Run completes; throughput-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms` (schedule-only change adds zero compute; epochs SHOULD match baseline). If epochs deviate materially it indicates an unrelated environment issue, not the schedule.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — add `N_CYCLES = 2`** (after WARMUP_FRAC, ~L24):
  ```python
  N_CYCLES = 2  # SGDR warm restarts (EXP-029): number of equal cosine cycles over the time budget
  ```
- **train.py — rewrite `lr_at_fraction` (L35-41)**:
  ```python
  def lr_at_fraction(frac):
      """Budget-matched SGDR (Loshchilov & Hutter 2017): split the budget into N_CYCLES
      equal cosine cycles, each annealing PEAK_LR -> ~0; linear warmup over WARMUP_FRAC of
      the FIRST cycle only; each restart jumps straight back to PEAK_LR. N_CYCLES=1 reduces
      exactly to the previous single cosine-to-0 schedule."""
      frac = min(max(frac, 0.0), 1.0)
      cycle_len = 1.0 / N_CYCLES
      cycle_idx = min(int(frac / cycle_len), N_CYCLES - 1)   # clamp the frac==1.0 edge into the last cycle
      local = (frac - cycle_idx * cycle_len) / cycle_len      # position within the current cycle, in [0,1]
      if cycle_idx == 0 and local < WARMUP_FRAC:
          return PEAK_LR * local / WARMUP_FRAC                # warmup (first cycle only)
      if cycle_idx == 0:
          progress = (local - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
      else:
          progress = local                                    # restarts: no warmup
      return PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * progress))
  ```

  Why this tests the hypothesis: it replaces the single full-budget cosine with 2 warm-restart cycles, so the optimizer re-anneals from PEAK at the 50% mark — the restart can escape the basin the single cosine settles into and converge into a potentially flatter/better-generalizing minimum (SGDR's single-model re-exploration benefit). Schedule-only → compute/param-neutral (no epoch-wall risk).

  Risks/edge cases: (a) Budget-splitting — each cycle gets ~45 ep; the strong-aug recipe may under-converge per cycle → null/mild-regression (the falsifiable downside). (b) Warmup is now WARMUP_FRAC of the first CYCLE (~2.3 ep) vs ~4.5 ep of the full budget before — shorter, but warmup is second-order here (the 50% restart jumps to PEAK with no warmup anyway and trains stably). (c) The `int()` edge at frac==1.0 is handled by the `min(..., N_CYCLES-1)` clamp (verified in the smoke check). N_CYCLES=1 exactly reproduces the baseline schedule (clean fallback).

## Configuration Changes
- `N_CYCLES`: (new) = 2. Two cycles is the minimal SGDR that introduces one restart while leaving each cycle long enough (~45 ep) to anneal meaningfully at the 300s budget. (T_mult=1, equal-length cycles.)
- All else unchanged: PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, Cutout 16, TA, Nesterov, params 4,299,866, seed 42.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` watch on run.log for the summary + NaN/error.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (unchanged).
- Estimated runtime: ~380-405s total (schedule-only; same compute as baseline). Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **Loss not decreasing after the first warmup** → kill. (NOTE: a transient loss BUMP right after the 50%-budget restart is EXPECTED and not an abort condition — the LR jumps back to PEAK; loss should recover and re-descend.)
- **No output / hang**: no new step lines for >120s → kill.
- **Wall-clock runaway**: process past ~580s → kill.
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (pure arithmetic change); seed 42 unchanged.

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024):** record `num_epochs` and mean `dt`. SGDR is schedule-only (zero added compute), so epochs SHOULD be ~91 / dt ~8ms.
- epochs ~91 & dt ~8ms → throughput-neutral → the accuracy delta is a FAIR test of the warm-restart hypothesis.
- epochs materially off ~91 → an unrelated environment/throughput issue (NOT caused by the schedule) → note and treat with caution.
Also sanity-check the restart fired: the run.log `lr:` trace should show a jump back toward 0.2 near the 50%-budget mark.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (throughput-neutral check).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195.

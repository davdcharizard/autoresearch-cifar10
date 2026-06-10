# Plan EXP-016: LR-schedule micro-tuning — raise peak LR 0.2 → 0.3 on the TA+Cutout recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Change `PEAK_LR` from `0.2` to `0.3` in train.py (line 23). No other change.
- [ ] `uv run ruff check train.py` clean; `git diff` shows ONLY the `PEAK_LR` constant line changed.

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED — this is a hyperparameter-only change),
      clean compile, no traceback, no NaN.
- [ ] Confirm warmup peaks at the new LR: the printed `lr:` should ramp toward ~0.30 during the first ~5%
      of the budget (warmup), then begin the cosine descent. Steady-state `dt` ~8ms/step (~91 epochs expected —
      compute-neutral, so a fair same-budget test, unlike EXP-015).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params`, `peak_vram_mb`.

## Code Changes
- **train.py** (the ONLY editable file):
  1. Line 23: `PEAK_LR = 0.2` → `PEAK_LR = 0.3`. Update the inline comment to note the EXP-016 retune.
  - **Why this tests the hypothesis**: `PEAK_LR` is consumed by `lr_at_fraction` (L35-41, the warmup+cosine
    schedule) and as the optimizer's initial `lr` (L194). Raising the peak makes the optimizer explore a wider
    region under the heavy TA+Cutout regularization before the cosine anneals to 0 — testing whether a flatter,
    better-generalizing minimum is reachable. The schedule SHAPE (5% linear warmup → cosine-to-0) is unchanged;
    only its amplitude scales.
  - **Risks/edge cases**: (a) a higher peak could overshoot/underfit within the ~91-epoch budget → graceful
    no-improvement (baseline 96.22 holds); (b) early-epoch loss may be transiently higher/noisier during the
    higher-LR phase — this is EXPECTED and not an abort trigger as long as it is not NaN/inf and recovers as the
    cosine anneals. Throughput/params unchanged (pure hyperparameter change) → fair test, no confound.

## Configuration Changes
- `PEAK_LR`: 0.2 → 0.3 (peak of the linear-warmup→cosine schedule). Rationale: peak 0.2 was a pre-widen,
  pre-augmentation heuristic (EXP-000) and was never re-tuned after k=4 + Cutout + TrivialAugment added heavy
  regularization; strongly-augmented recipes tolerate/benefit from a more aggressive peak (Goyal 2017
  arXiv:1706.02677; Loshchilov 2017 arXiv:1608.03983). 0.2 already beat the textbook batch-128 peak (0.1) in
  EXP-000, evidence the model likes a higher LR; 0.3 is a modest further step. All else inherited from the
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
- Loss NaN/inf or sustained divergence (loss climbing for many steps without recovery, not just the expected
  higher-LR transient) → kill.
- Traceback / process exit ≠ 0 at startup → kill, fix, single retry.
- `num_params` ≠ 4,299,866 → unexpected (this is a hyperparameter-only change; any param change signals an
  accidental structural edit) → kill, investigate.
- Early-accuracy sanity: if `test_acc` is still < ~90% past ~50% of the budget (≈ ep 45) → suspect instability
  from the higher LR → note for analysis (let it finish unless NaN/stall).
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
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~88–91; compute-neutral, should match
  EXP-012; a large deviation would itself be a confound to flag).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — corroborator vs EXP-012's 0.195 (< 0.195 with acc↑
  ⇒ higher LR improved generalization; ≈ 0.195 with flat acc ⇒ peak already near-optimal).
- `peak_vram_mb`: `grep -a "^peak_vram_mb:" run.log` — soft-constraint awareness (expect ~450 MB, unchanged).
- `img/s` & `dt`: step ~400–500 — confirm ~8ms/step to rule out a throughput confound.

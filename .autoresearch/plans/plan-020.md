# Plan EXP-020: SWA with a lower constant-LR floor (SWA_LR 0.05 → 0.02)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Re-apply the EXP-019 SWA implementation to train.py (it was discarded after the no-improvement verdict — the working tree is back at the EXP-012 baseline). Six edits, identical to EXP-019 EXCEPT `SWA_LR = 0.02` (was 0.05): import `AveragedModel`; add `SWA_START_FRAC=0.75`, `SWA_LR=0.02`, `BN_RECOMPUTE_BATCHES=50`; rewrite `lr_at_fraction` (warmup → cosine PEAK_LR→SWA_LR over [WARMUP_FRAC, SWA_START_FRAC] → constant SWA_LR tail); add `recompute_bn()`; construct `swa_model = AveragedModel(model)`; branch the per-epoch eval (tail → update_parameters + recompute_bn + eval SWA model; main phase → eval raw model).
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.
- [ ] `git diff --name-only` shows only `train.py`.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED), clean compile, no traceback, no NaN.
- [ ] Confirm the LR floor: in the tail, logged `lr` reads `0.0200` (constant), not 0.05 and not decaying to 0.
- [ ] Confirm the tail SWA eval path runs (`[swa]` eval lines appear from ~ep 68; acc sensible).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints the summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `final_test_loss`, `num_epochs`, `total_seconds`, `peak_vram_mb` from run.log.

## Code Changes
- **train.py** (the ONLY file modified): the EXP-019 SWA code with the single hyperparameter change `SWA_LR = 0.02`
  (was 0.05). Concretely:
  - Hyperparameters: `SWA_START_FRAC = 0.75`, `SWA_LR = 0.02`, `BN_RECOMPUTE_BATCHES = 50`.
  - `lr_at_fraction`: warmup → cosine from `PEAK_LR` (0.2) down to `SWA_LR` (0.02) over [WARMUP_FRAC, SWA_START_FRAC]
    → constant `SWA_LR` for the tail [SWA_START_FRAC, 1]. Continuous at the join (cos(pi)=−1 ⇒ SWA_LR).
  - `recompute_bn(swa_model, loader, n_batches, device)`: reset BN stats (momentum=None, cumulative average) and
    forward `BN_RECOMPUTE_BATCHES` augmented training batches (channels_last + Cutout + bf16 autocast) through the
    SWA model in train mode under no_grad.
  - `swa_model = AveragedModel(model)` constructed once before the loop (eager, never compiled).
  - Per-epoch eval branch: tail (`total_training_time/TIME_BUDGET_S >= SWA_START_FRAC`) → `update_parameters` +
    `recompute_bn` + evaluate the SWA model; main phase → evaluate the raw model. Exactly one `evaluate()`/epoch.
  - **Why this tests the hypothesis**: a lower floor makes each averaged snapshot individually higher-top-1 (LR
    nearer the well-converged region) while retaining iterate movement for flat-region averaging — directly testing
    EXP-019's diagnosis that the 0.05 floor was too high (raw iterate cratered to 91.8%, capping the average).
  - **Risks/edge cases**: (a) if 0.02 is too low the iterate barely moves → average approaches a single
    constant-0.02 endpoint lacking cosine-to-0's final sharpening → likely ~96.0–96.1 (graceful no-improvement).
    (b) swa_model eager/separate → torch.compile graphs untouched (no recompiles). (c) +~17MB VRAM (trivial).
    (d) BN-recompute + eval run outside the per-step timer → training keeps the full 300s; total_seconds < 600.

## Configuration Changes
- SWA_LR: 0.05 (EXP-019) → **0.02** (the ONLY change vs EXP-019). Rationale: 0.05's raw iterate degraded to
  91.8% top-1 (snapshot quality cap); 0.02 (=10% of peak ≈ where baseline cosine sits ~80% through) keeps the
  snapshots higher-quality while still moving enough to average. Literature: SWA solution quality is set by the
  constant averaging LR — interior sweet spot (knowledge/papers/swa.md).
- Unchanged from EXP-019: SWA_START_FRAC 0.75 (keeps the ~24-snapshot tail that was still improving),
  BN_RECOMPUTE_BATCHES 50, and the full EXP-012 recipe (k=4 4.3M params, PEAK_LR 0.2, batch 128, Nesterov,
  WD 1e-4, LS 0.1, Cutout(16), TrivialAugment, torch.compile, bf16, channels_last, seed 42).

## Execution Environment
- Method: local — `cd <project-root> && CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background launch.
- Resources: 1× NVIDIA H20 (98GB); ~0.5GB VRAM expected.
- Estimated runtime: ~390–420s total wall-clock (300s training + eval + ~6s BN-recompute), well under 600s.
- Log output: all stdout/stderr → `run.log` (source of truth).
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf, or training diverges through the tail (debiased loss climbing steadily rather than holding/decreasing).
- Traceback / crash at startup — fix code error, counts as one retry.
- No new log output for > 3 minutes (silent hang).
- `params` ≠ 4,299,866 at startup (SWA must not change params).
- total wall-clock approaching 600s — kill and treat as failure.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22%**; success bar = **96.32%** (+0.1pp).

1. **Baseline**: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline "/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/.autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv"` → confirm 96.22.
2. **Cond 1 — primary metric clears bar**: `grep -aE "^best_test_acc:" run.log` → PASS iff `best_test_acc >= 96.32`. (Decisive; run finishes ≤ ~7 min.)
3. **Cond 2 — clean completion within budget**: `best_test_acc` and `total_seconds` present; `grep -ac "Traceback" run.log` == 0; `total_seconds < 600`.
4. **Cond 3 — no constraint violations**: `git diff --name-only` = train.py only; `num_params` == 4,299,866; eval-count (`grep -ac "eval ep" run.log`) == `num_epochs` (one evaluate()/epoch); no new deps (`swa_utils` is core torch); seed 42 intact.
5. Compare and render verdict. Empty `best_test_acc` ⇒ crashed (`tail -n 50 run.log`).
6. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log`
- final_test_loss: `grep -aE "^final_test_loss:" run.log` (compare to EXP-019's 0.1788 and baseline 0.195 — diagnostic for whether the lower floor moved the loss/flatness)
- Also note the SWA-eval trajectory tail (did the 0.02 average exceed EXP-019's 95.97? did it plateau or keep rising?).

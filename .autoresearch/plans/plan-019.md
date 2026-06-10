# Plan EXP-019: SWA with a constant-LR averaging tail (proper Stochastic Weight Averaging)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md

## Milestones

### Milestone 1: Code changes implemented and pass local checks
- [ ] Add SWA hyperparameters (`SWA_START_FRAC=0.75`, `SWA_LR=0.05`, `BN_RECOMPUTE_BATCHES=50`) to the hyperparameter block.
- [ ] Modify `lr_at_fraction` so the schedule is: linear warmup → cosine from `PEAK_LR` down to `SWA_LR` over `[WARMUP_FRAC, SWA_START_FRAC]` → constant `SWA_LR` for `[SWA_START_FRAC, 1.0]`. Confirm continuity at the join (cosine endpoint = `SWA_LR`).
- [ ] Add a `recompute_bn(swa_model, loader, n_batches, device)` helper that resets BN running stats (`momentum=None`, cumulative average) and runs `n_batches` augmented training batches forward through `swa_model` in train mode (channels_last + Cutout, under bf16 autocast — matching training-time input distribution).
- [ ] Construct `swa_model = torch.optim.swa_utils.AveragedModel(model)` before the loop (eager, never compiled).
- [ ] In the training loop, detect the SWA tail (`total_training_time / TIME_BUDGET_S >= SWA_START_FRAC`). At each epoch end: if in the tail, `swa_model.update_parameters(model)`, then `recompute_bn(...)`, then evaluate the **SWA model** (`evaluator.evaluate(swa_model, device)`); otherwise evaluate the **raw model** as before. Exactly one `evaluate()` call per epoch in both branches.
- [ ] Ensure `best_acc` is updated from whichever model was evaluated that epoch; keep the final summary block intact.
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED — SWA adds no model params), clean compile, no traceback, no NaN.
- [ ] Confirm the LR floor holds: in the tail, logged `lr` reads `0.0500` (constant), not decaying to 0.
- [ ] Confirm the tail SWA eval path runs (an `eval` line appears each tail epoch; acc is sensible, not degenerate).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints the summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `total_seconds`, `peak_vram_mb` from run.log.

## Code Changes
- **train.py** (the ONLY file modified):
  1. **Hyperparameters**: add `SWA_START_FRAC = 0.75` (fraction of budget after which the constant-LR averaging tail begins), `SWA_LR = 0.05` (constant tail LR — a moderate floor = peak/4, standard SWA range), `BN_RECOMPUTE_BATCHES = 50` (truncated BN-stat recompute pass; ~6.4k images is ample for stable BN estimates and keeps overhead ~0.3s/epoch). Rationale: tests the hypothesis that a flat-region weight average under a *moving* (constant-LR) iterate beats the single cosine-to-0 endpoint.
  2. **`lr_at_fraction(frac)`**: replace the single cosine-to-0 branch with: warmup (unchanged) → for `WARMUP_FRAC ≤ frac < SWA_START_FRAC`, `SWA_LR + (PEAK_LR - SWA_LR) * 0.5 * (1 + cos(pi * progress))` where `progress = (frac - WARMUP_FRAC)/(SWA_START_FRAC - WARMUP_FRAC)` → for `frac ≥ SWA_START_FRAC`, return `SWA_LR`. This supplies the terminal-LR floor that EXP-006 lacked (the documented precondition for weight averaging to help). Continuity holds: at `frac = SWA_START_FRAC`, `progress=1`, `cos(pi)=-1` → returns `SWA_LR`.
  3. **`recompute_bn` helper**: new module-level function. Resets `_BatchNorm` running stats and sets `momentum=None` (cumulative moving average, identical to `torch.optim.swa_utils.update_bn`), then forwards `BN_RECOMPUTE_BATCHES` training batches through `swa_model` in `.train()` mode with the same input pipeline as training (channels_last, `cutout_batch`, bf16 autocast), under `torch.no_grad()`. Needed because `AveragedModel(use_buffers=False)` does NOT average BN buffers — the averaged weights have stale BN stats that must be re-estimated before eval.
  4. **SWA model + loop branch**: construct `swa_model = AveragedModel(model)` once before the loop. In the per-epoch tail branch, accumulate the snapshot (`update_parameters`), recompute BN, and evaluate `swa_model`; the main-phase branch evaluates the raw `model` exactly as today. `best_acc = max(best_acc, test_acc)` over all epochs regardless of which model produced it.
  - **Risks/edge cases**: (a) constant-LR tail forgoes cosine-to-0 sharpening — if averaging under-compensates, mild regression (graceful no-improvement). (b) `swa_model` is eager and separate, so `torch.compile(model)` graphs are untouched — no recompiles. (c) extra model ≈ +17MB VRAM (trivial of 98GB). (d) BN-recompute + eval run outside the per-step timer (like the existing eval), so they don't perturb the LR/budget clock; their wall-clock (~0.3s × ~21 tail epochs ≈ 6s) keeps `total_seconds` well under 600.

## Configuration Changes
- LR schedule: cosine `0.2 → 0` over full budget  →  cosine `0.2 → 0.05` over `[5%, 75%]`, then constant `0.05` over `[75%, 100%]` (rationale: SWA requires a non-zero terminal LR so the iterate keeps moving through the flat region; Izmailov 2018, knowledge/papers/swa.md).
- New constants: `SWA_START_FRAC = 0.75`, `SWA_LR = 0.05`, `BN_RECOMPUTE_BATCHES = 50`.
- Eval target in the tail: raw `model` → BN-recomputed `swa_model` (still ≤ 1 `evaluate()` call/epoch).
- Unchanged: k=4 architecture (4,299,866 params), batch 128, Nesterov, WD 1e-4, label smoothing 0.1, Cutout(16), TrivialAugment, torch.compile, bf16, channels_last, seed 42, PEAK_LR 0.2, WARMUP_FRAC 0.05.

## Execution Environment
- Method: local — `cd <project-root> && uv run train.py > run.log 2>&1`, background launch (`run_in_background: true`), single GPU (`CUDA_VISIBLE_DEVICES=0` if needed).
- Resources: 1× NVIDIA H20 (98GB); ~0.5GB VRAM expected (baseline ~454MB + ~17MB SWA copy).
- Estimated runtime: ~300s training + startup + per-epoch eval overhead + ~6s BN-recompute ≈ ~390–410s total wall-clock (well under the 600s hard limit).
- Log output: all stdout/stderr → `run.log` (the source of truth). Monitor via tail/Monitor for the completion summary and the constant-LR-tail confirmation.
- Tool skill: none (local run).

## Abort Criteria
- Loss goes NaN/inf, or training diverges after the LR floor kicks in (debiased loss climbing steadily through the tail rather than holding/decreasing).
- Traceback / crash at startup (e.g., AveragedModel or BN-recompute error) — fix code error, counts as one retry.
- No new log output for > 3 minutes (silent hang).
- `params` ≠ 4,299,866 at startup (SWA must not change model params — would signal a wiring bug).
- total wall-clock approaching 600s — kill and treat as failure.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline` on `experiment-indices/improve-cifar10-test-accuracy.tsv`) = **96.22%**; success bar = **96.32%** (+0.1pp).

1. **Baseline**: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline "/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/.autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv"` → confirm baseline 96.22.
2. **Cond 1 — primary metric clears bar**: `grep -aE "^best_test_acc:" run.log` → parse the value; PASS iff `best_test_acc >= 96.32`. (Decisive condition; timeout: run should finish ≤ ~7 min.)
3. **Cond 2 — clean completion within budget**: `grep -acE "^best_test_acc:|^total_seconds:" run.log` present AND `grep -ac "Traceback" run.log` == 0 AND `total_seconds < 600` (from `grep -aE "^total_seconds:" run.log`).
4. **Cond 3 — no constraint violations**: `git diff --name-only` lists only `train.py`; `num_params` == 4,299,866 (unchanged); eval-count == num_epochs (one evaluate per epoch — verify by the count of `eval ep` lines == `num_epochs`); no new deps (only torch/math/torchvision, all already present — `swa_utils` is core torch); seed 42 intact.
5. Compare and render verdict. If `best_test_acc` empty ⇒ crashed (`tail -n 50 run.log`).
6. Remove `run.log` before the next experiment (housekeeping; `.autoresearch/` excluded).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (watch the tail epoch count — need enough snapshots, ~15–21, for averaging to matter)
- throughput (img/s): from the per-step log lines (`img/s:` field) — confirm ~15k (throughput-neutral vs compiled-k4).

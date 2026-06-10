# Plan EXP-021: Larger Cutout (CUTOUT_SIZE 16 → 20)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] On the EXP-012 baseline train.py (working tree = baseline), change the single constant `CUTOUT_SIZE = 16` → `CUTOUT_SIZE = 20`.
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.
- [ ] `git diff --name-only` shows only `train.py`; `git diff` shows only the CUTOUT_SIZE line changed.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED), clean compile, no traceback, no NaN.
- [ ] Confirm throughput-neutral (~8ms/step, ~15k img/s, ~91 epochs) — Cutout is a GPU op, size change adds no cost.

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `final_test_loss`, `num_epochs`, `total_seconds`, `peak_vram_mb` from run.log.

## Code Changes
- **train.py** (the ONLY file modified): line 28, `CUTOUT_SIZE = 16` → `CUTOUT_SIZE = 20`.
  - **Why this tests the hypothesis**: `cutout_batch` zeroes a `CUTOUT_SIZE`×`CUTOUT_SIZE` window per image; raising
    it to 20 (≈39% area vs 16's ≈25%) increases occlusion regularization. EXP-013 showed reducing it under-regularizes
    (loss rose, acc fell); this tests the indicated up-direction on a generalization-bound model.
  - **Risks/edge cases**: 20px may over-occlude (remove too much signal) → loss rises, acc drops (graceful
    no-improvement → Cutout optimum is ≤16). The vectorized `cutout_batch` clips the window to the border via the
    mask, so a larger size is handled correctly (no index errors). No compute/param change → fair test.

## Configuration Changes
- CUTOUT_SIZE: 16 → **20** (the ONLY change). Rationale: aug-strength UP-direction indicated by EXP-013
  (down-direction under-regularized); augmentation is the only recently-productive mechanism on this project.
- Unchanged: the full EXP-012 recipe (k=4 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov, WD 1e-4,
  LS 0.1, TrivialAugment, torch.compile, bf16, channels_last, seed 42).

## Execution Environment
- Method: local — `cd <project-root> && CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background launch.
- Resources: 1× NVIDIA H20 (98GB); ~0.45GB VRAM expected.
- Estimated runtime: ~390–420s total wall-clock (300s training + per-epoch eval), well under 600s.
- Log output: all stdout/stderr → `run.log` (source of truth).
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or diverging.
- Traceback / crash at startup — fix code error, counts as one retry.
- No new log output for > 3 minutes (silent hang).
- `params` ≠ 4,299,866 at startup (should be impossible — aug-only change).
- total wall-clock approaching 600s — kill and treat as failure.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22%**; success bar = **96.32%** (+0.1pp).

1. **Baseline**: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline "/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/.autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv"` → confirm 96.22.
2. **Cond 1 — primary metric clears bar**: `grep -aE "^best_test_acc:" run.log` → PASS iff `best_test_acc >= 96.32`. (Decisive; run finishes ≤ ~7 min.)
3. **Cond 2 — clean completion within budget**: `best_test_acc` and `total_seconds` present; `grep -ac "Traceback" run.log` == 0; `total_seconds < 600`.
4. **Cond 3 — no constraint violations**: `git diff --name-only` = train.py only; `num_params` == 4,299,866; eval-count (`grep -ac "eval ep" run.log`) == `num_epochs` (one evaluate()/epoch); no new deps; seed 42 intact.
5. Compare and render verdict. Empty `best_test_acc` ⇒ crashed (`tail -n 50 run.log`).
6. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log`
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — KEY diagnostic: vs baseline 0.195. Loss DOWN+acc UP = larger Cutout helps; loss UP = over-occlusion (axis closed at ≤16).

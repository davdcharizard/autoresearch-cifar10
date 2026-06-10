# Plan EXP-023: Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] On the EXP-012 baseline train.py, change the single constant `LABEL_SMOOTHING = 0.1` → `LABEL_SMOOTHING = 0.05` (L27).
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.
- [ ] `git diff --name-only` shows only `train.py`; `git diff` shows only the LABEL_SMOOTHING line changed.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED), clean compile, no traceback, no NaN.
- [ ] Confirm throughput-neutral (~8ms/step, ~15k img/s, ~91 epochs) — LS is a scalar in the loss, zero compute change.

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `final_test_loss`, `num_epochs`, `total_seconds`, `peak_vram_mb` from run.log.

## Code Changes
- **train.py** (the ONLY file modified): line 27, `LABEL_SMOOTHING = 0.1` → `LABEL_SMOOTHING = 0.05`.
  - **Why this tests the hypothesis**: `LABEL_SMOOTHING` is passed to `F.cross_entropy(..., label_smoothing=LABEL_SMOOTHING)`
    in the training loop. The project's strongest insight is that the recipe is convergence-bound (not overfit-bound),
    so REDUCING a regularizer is the indicated direction; LS is the one recipe regularizer never swept. Halving it lets
    the model commit to sharper targets within the budget while TA+Cutout still supply regularization.
  - **Risks/edge cases**: LS top-1 effects are usually small (mainly calibration), so the result may land within the
    ~0.2pp noise band (graceful no-improvement). If 0.1 was load-bearing, reducing could slightly hurt. No compute/param
    change → fully attributable fair test. NOTE on diagnostic: label smoothing inflates cross-entropy test loss by a
    fixed offset, so `final_test_loss` is NOT directly comparable across different LS values — judge primarily on
    best_test_acc; a LOWER LS naturally yields a lower reported test loss even with equal accuracy.

## Configuration Changes
- LABEL_SMOOTHING: 0.1 → **0.05** (the ONLY change). Rationale: the convergence-bound insight (EXP-005/011/018/022)
  prescribes reducing regularization; 0.05 is a single interior step (0.0 reserved as a follow-up if 0.05 helps).
- Unchanged: the full EXP-012 recipe (k=4 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov, WD 1e-4,
  TrivialAugment + Cutout(16), torch.compile, bf16, channels_last, seed 42).

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
- `params` ≠ 4,299,866 at startup (should be impossible — scalar-only change).
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
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — NOTE: not comparable across LS values (LS adds a fixed offset to CE loss); a drop here is expected from lower LS and is NOT itself a quality signal. Judge on best_test_acc.

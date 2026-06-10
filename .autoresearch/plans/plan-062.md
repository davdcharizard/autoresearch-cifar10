# Plan EXP-062: WARMUP_FRAC isolation (0.05 → 0.10)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md

## Closed-axis check
The LR-schedule axis has PEAK_LR (EXP-016/017) and schedule-SHAPE (EXP-029 SGDR) closed, but WARMUP_FRAC (the linear-warmup length) was never isolated — it is a distinct, untested scalar, not a retry of a closed sub-lever. Single-variable, throughput- and wall-neutral (no EXP-061-style wall-overrun risk: warmup only redistributes the existing time-fraction LR schedule, adds zero work).

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py L24: `WARMUP_FRAC = 0.05` → `WARMUP_FRAC = 0.10`. Update the inline comment to note EXP-062.
- [ ] Smoke: `ast.parse` OK; `git diff --name-only` == train.py only (one-line change); sanity-check `lr_at_fraction(0.0)=0`, `lr_at_fraction(0.10)≈PEAK_LR` (peak reached at frac=0.10), `lr_at_fraction(1.0)≈0` (still anneals to 0).

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (no structural change → identical throughput), no NaN, loss descending, LR ramp visibly longer (peak ~0.2 reached later, around frac 0.10 / ~ep9 vs ~ep4.5 baseline). Wall projects ~593s (same recipe; AugMix wall, gate per EXP-054/061 — if base wall projects > ~596s, note it).

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline.

## Code Changes
- **train.py (L24)**: `WARMUP_FRAC` 0.05 → 0.10. The `lr_at_fraction` function (L35-41) linearly warms 0→PEAK_LR over the first WARMUP_FRAC of the time budget, then cosine-anneals to 0. Doubling the warmup fraction lengthens the ramp (~4.5→~9 epochs) and slightly shortens the high-LR cosine phase. Tests whether a longer warmup stabilizes early training under noisy AugMix gradients at PEAK_LR=0.2. Risk/edge case: the LR regime is finely balanced (EXP-016/017: ±0.05 peak cost ~0.5pp), so a longer warmup may marginally under-train the mid-cosine phase → small regression possible; no structural/throughput/scope risk.

## Configuration Changes
- `WARMUP_FRAC`: 0.05 → 0.10. Rationale: the one untested LR-schedule scalar; longer warmup hypothesized to stabilize early high-LR training under strong augmentation. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2, Nesterov m0.9, WD1e-4, LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~593s (< 600s; same recipe as EXP-054).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or not descending by ep5.
- dt drifts ≫ 8ms (contention — should not happen, no structural change): kill, relaunch on clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:" run.log`; confirm summary printed, `total_seconds < 600`, total wall < 10 min, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (unchanged loop); no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics.
- Timeout: 10 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs/num_steps, final_test_loss: `grep -aE "^peak_vram_mb:|^num_epochs:|^num_steps:|^final_test_loss:" run.log` — confirm ~91 ep (throughput unchanged) and compare loss to EXP-054's 0.1968.

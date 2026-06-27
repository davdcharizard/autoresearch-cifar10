# Experiment Log: EXP-038 — BN running-stat momentum 0.1 → 0.02 (last unmeasured implicit constant)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-038.md
- **Plan**: plans/plan-038.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-038 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Plan Milestone 1 implemented in train.py only: new constant `BN_MOMENTUM = 0.02` in the hyperparameter block, passed to all three `nn.BatchNorm2d` construction sites (BasicBlock bn1/bn2, ResNet stem bn1) — covering all 19 BN layers (verified by module walk). Diff: 4 insertions / 3 deletions, train.py only. CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run`): 19 BN layers all report momentum 0.02; params 4,286,026 (unchanged — momentum is buffer-update arithmetic, not a parameter); forward (4,3,32,32)→(4,10) OK. Momentum is set at construction and never mutated at runtime, so the EXP-035 compile-guard caveat (runtime attribute toggling → recompile storm) does not apply.

### Surprises & Discoveries
- None — the smallest-diff experiment alongside EXP-036. Confirmed BN momentum was previously the PyTorch default everywhere (no explicit setting existed anywhere in train.py).

### Decisions
- Chose constructor-argument style (3 sites + 1 constant) over a post-construction `for m in model.modules()` loop — keeps the dial visible in the constants block like every other dosed hyperparameter, per repo convention.

## Run Log

### Run 1
- **Description**: Full composite gated run on GPU 0 of the baseline recipe with BN running-stat momentum 0.02 (~50-batch EMA horizon vs default ~10). Training path is byte-identical (weights/gradients/schedule/noise untouched; only the stat-buffer EMA coefficient changes), so expected signatures match baseline exactly: dt ≈22.4ms, ~139 epochs, params 4,286,026, VRAM 1613MB, total ~475–495s. Watchdog: baseline thresholds (contention 4×>27ms, STARTUP_KILL tick 10, NaN, divergence eval<15% after ep5, wall cap 600s). Predicted mechanism signature (diagnostic, not abort): hot-phase evals slightly below family (stat lag), converging to a plateau whose LEVEL and SCATTER answer the hypothesis vs bar 96.81.
- **Job ID / PID**: background task bq5ct1s5f (composite script /tmp/exp036_composite.sh — baseline-threshold variant, exactly plan-038's watchdog spec; train PID in LAUNCHED line)
- **Log file**: run.log (project root; deleted after analysis per goal constraints)
- **WandB**: n/a
- **Status**: completed (rc=0, watchdog never triggered)
- **Started**: 2026-06-10 19:41:52 (GATES_CLEAR poll 1: apps=0, load=9)
- **Ended**: 2026-06-10 ~19:50:11 (PROC_EXITED tick 34; total_seconds 499.3)
- **Observations**: Training-path signatures byte-identical to baseline as designed: 31 watchdog windows 21.7–22.8ms, fine profile 267 windows mean 22.3ms with 0 >27, 139 epochs / 13,429 steps, VRAM 1613.0MB, params 4,286,026. The eval-side, however, moved hard in the WRONG direction with the lag mechanism written all over it: ep5 eval 35.30 (family ~64; eval dipped below its own earlier epochs while train loss progressed normally), ep20 65.83 (family ~79) — the 50-batch stat horizon badly misaligns constants while weights move fast. Critically, the damage persists INTO the plateau: last-15 evals mean 96.022, spread 0.64 (family ~96.5, spread ~0.15), final_test_loss 0.1917 vs family ~0.185 — the cosine tail's continued weight drift (the EXP-033 endgame climb) keeps 50-batch-old constants stale even at "convergence". Prediction inverted: smoothing the estimator increased effective constants error; FRESHNESS (default 10-batch horizon) is what keeps BN constants matched to current weights.
- **Key Metrics**: best_test_acc 96.27 | final 96.20 | final_test_loss 0.1917 | 139 epochs / 13,429 steps | dt mean 22.3ms (267 win, 0 slow>27) | params 4,286,026 | VRAM 1613.0MB | startup 18.4s | total 499.3s

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
First-failure-stop protocol per plan-038 (bar = baseline 96.71 + 0.1 = 96.81).

1. **best_test_acc ≥ 96.81** — **FAIL**. `grep "^best_test_acc:" run.log` → **96.27%**. Pre-condition (profile) PASSED first: 267 windows, mean 22.3ms, 0 >27ms (quantization-safe threshold per EXP-037 protocol note — no 30.0 readings at all), 139 epochs within 139±4 → pristine, uncontaminated. Integrity: num_params 4,286,026 ✓; training_seconds 300.0 ✓; eval_lines 139 = num_epochs ✓. 96.27 < 96.81 → fails on merits, and sits BELOW the baseline band (96.4–96.7) — a genuine negative effect, not noise.
2. **Completes within budget** — skipped (aborted after prior failure). [Incidental: rc=0, total 499.3 ≤ 600.]
3. **Validation ≤ once/epoch** — skipped (aborted after prior failure). [Incidental: 139 = 139.]

**Verdict basis**: valid pristine run, condition 1 failed → no-improvement. Hypothesis INVERTED with full mechanism diagnostics: lag dominated variance at every phase (ep5 35.3 vs ~64; plateau mean 96.02 with spread 0.64 vs family ~96.5/±0.15; test_loss 0.1917 vs 0.185).

### Informational Metrics
- **Plateau statistics**: last-15 mean 96.022, min 95.63, max 96.27, spread 0.64 — vs baseline family ~96.5 mean, ~0.15 spread. Scatter INCREASED (prediction was reduction): stale constants under continued tail weight-drift add error rather than removing it.
- **Hot-phase lag signature**: ep1 31.42, ep2 39.58, ep5 35.30 (non-monotone — eval dropped while training progressed), ep10 67.03, ep20 65.83, ep30 79.32 — the misalignment is large and persistent through the hot phase, confirming the estimator-lag mechanism operated (with sign opposite the hypothesis).
- **dt / epochs / VRAM / params**: 22.3ms / 139 / 1613.0MB / 4,286,026 — baseline-identical, implementation pure.
- **final_test_loss**: 0.1917 vs family ~0.185 — consistent with systematically stale constants.

## Human Notes
(autopilot — none)

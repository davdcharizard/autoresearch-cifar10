# Experiment Log: EXP-039 — BN running-stat momentum 0.1 → 0.25

## Execution

- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-039.md
- **Plan**: plans/plan-039.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-039
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (run clean; verification Condition 1 failed on merits — 96.64 < 96.81)

## Implementation Notes

### Summary
Milestone 1 executed exactly as planned: on `autoresearch/exp-039` (cut from `autoresearch/dev` @ 1990397), added `BN_MOMENTUM = 0.25` to the constants block (after `LABEL_SMOOTHING`) and passed `momentum=BN_MOMENTUM` at the three `nn.BatchNorm2d` construction sites (BasicBlock bn1/bn2, ResNet stem bn1). `git diff --stat`: train.py only, 4 insertions / 3 deletions — byte-mirror of EXP-038's diff with 0.02 → 0.25. CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python`, module walk): all 19 BatchNorm2d layers report momentum 0.25; params 4,286,026 unchanged.

### Surprises & Discoveries
None at implementation time — the diff shape was validated by EXP-038 (identical mechanism, no compile-guard interaction since momentum is set at construction, not mutated at runtime).

### Decisions
- Dose chosen at 0.25 (not 0.2 or 0.5) per plan rationale: ~2.5× lag reduction while the per-estimate sample count stays ~4 batches × 512 = 2,048 — far from the noisy single-batch regime; 0.2 risks an unresolvable sub-noise read.
- Composite launcher reuses the baseline-threshold variant (contention > 27ms — off the 6ms quantization rungs at true dt 22.4) since the training path is signature-identical to baseline; no experiment-specific dt gate needed (unlike EXP-037's SE-dt gate).

## Run Log

### Run 1
- **Description**: Full budget-matched run of the m=0.25 variant on GPU 0 via composite gated launcher (`/tmp/exp039_composite.sh`): dual launch gates (GPU-0 zero compute apps AND 1-min load < 60), background `uv run train.py > run.log 2>&1`, 44×15s tick watchdog (CONTENTION_KILL 4 consecutive windows > 27ms; STARTUP_KILL tick 10; NaN guard; DIVERGENCE_KILL eval < 15% after ep5; WALL_CAP_KILL tick 44). Expected: baseline signatures (dt 22.3–22.4ms, ~139 epochs, VRAM 1613MB, total ~480–500s); hypothesis says plateau evals at-or-above family (~96.5) if residual lag at m=0.1 is real, baseline-band if 0.1 is already at the optimum. Hot-phase diagnostic: ep5/10/20 evals at-or-above family (~64/~75/~79) = lag reduced; below = variance cost.
- **Job ID / PID**: background task bjtexxz93 (`/tmp/exp039_composite.sh`); train PID printed in composite stdout
- **Log file**: run.log (project root); watchdog summary via composite script stdout (task output file)
- **WandB**: N/A
- **Status**: completed (rc=0, no watchdog trigger)
- **Started**: 2026-06-10 20:03:11 (gates clear at poll 1: GPU-0 apps=0, load=3.0)
- **Ended**: 2026-06-10 ~20:11:30 (PROC_EXITED at tick 33; total_seconds 490.5)
- **Observations**: Pristine run end-to-end. Watchdog windows 21.7–22.7ms across all 30 ticks, slow_streak never above 0 (composite stdout, task bjtexxz93). Coarse profile from run.log step lines: 267 windows mean 22.31ms, 0 > 27ms; 200-step quantization-safe windows: 66, mean 22.32, max 22.5. Startup 18.1s, VRAM 1613.0MB, 139 epochs / 13,446 steps — signatures byte-identical to baseline family as predicted.
- **Key Metrics**: best_test_acc 96.64 (ep133), final 96.54, final_test_loss 0.1859, training_seconds 300.0, total 490.5s. Hot phase: ep3 52.70 / ep5 62.23 (family ~64) / ep10 75.52 (~75) / ep20 82.01 (~79) — family-equal, no lag-reduction gain and no variance damage. Plateau (last 15): mean 96.449, min 96.15, max 96.64, spread 0.49 vs family ~96.5 ± ~0.15 — mean in noise-band but scatter ~3× family: the 4-batch-horizon variance cost is visible in eval draws while the max-statistic still harvests a top draw (96.64 ≈ family best-of-run).

## Experimental Adjustments

(none yet)

## Errors & Dead Ends

(none yet)

## Verification Results

### Conditions Checked

First-failure-stop per plan-039 § Verification Protocol; baseline at verification time 96.71 (bar 96.81).

- **Pre-condition — run integrity**: PASS. Profile pristine: 267 step-line windows mean 22.31ms, 0 > 27ms; 200-step quantization-safe windows (66) mean 22.32ms, max 22.5ms (no rung ambiguity — EXP-037 protocol note applied). num_epochs 139 ∈ [135, 143]. Integrity sub-checks: num_params 4,286,026 ✓ (capacity unchanged), training_seconds 300.0 ✓ (timer semantics untouched), eval-line count 139 == num_epochs ✓. Source: run.log greps recorded above; composite stdout (task bjtexxz93).
- **Condition 1 — best_test_acc ≥ 96.81**: **FAIL on merits.** `grep "^best_test_acc:" run.log` → 96.64 < 96.81. The result is judged on merits: clean profile, full epoch count, family signatures — no false-failure risk. Verification stopped here per first-failure-stop.
- **Condition 2 — completes within budget**: skipped per protocol (incidental: rc=0, total_seconds 490.5 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: skipped per protocol (incidental: 139 evals = 139 epochs).

**Diagnostics (dose-response read)**: hot phase family-equal (ep5 62.23 vs ~64, ep10 75.52 vs ~75, ep20 82.01 vs ~79) — no measurable lag reduction, i.e., m=0.1's constants are NOT meaningfully stale; plateau last-15 mean 96.449 (noise-band vs family ~96.5) with spread 0.49 ≈ 3× family (~0.15) — the variance cost of the ~4-batch horizon is real and visible; final_test_loss 0.1859 = family (~0.185); best 96.64 within the baseline band (96.4–96.7).

### Informational Metrics

- peak_vram_mb: 1613.0 (unchanged)
- num_epochs: 139 (baseline-equal)
- num_params: 4,286,026 (unchanged)

## Human Notes

(autopilot — none)

# EXP-058: Classifier weight decay ×4 (fc.weight WD 5e-4 → 2e-3)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-058
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Two-hunk diff to train.py exactly as planned (Milestone 1): added `FC_WEIGHT_DECAY = 2e-3` to the constants block, and split the optimizer into three param groups — conv weights at WEIGHT_DECAY 5e-4, `[base_model.fc.weight]` alone at FC_WEIGHT_DECAY 2e-3, BN/bias at 0. The in-loop lr assignment and the group-0 lr print are group-count-agnostic, so the printed lr remains the live schedule. CPU sanity (`/tmp/exp058_sanity.py`) passed all four checks: params 4,286,026 exact; fc.weight alone in the 2e-3 group; ledger conv 4,277,952 / fc 2,560 / no-decay 5,514 disjoint+exhaustive; 3-step smoke decreasing (5.19 → 2.46). No GPU probe per plan — EXP-057 validated this optimizer-only diff class in vivo (family signatures throughout).

### Surprises & Discoveries
- None. The three-group structure slotted in without touching any loop code, as predicted.

### Decisions
- fc.bias stays in the no-decay group (ndim ≤ 1), matching both the baseline and EXP-057 conventions — the dose targets only the weight matrix that sets logit scale.
- Sanity smoke starting loss differs run-to-run (model init precedes the manual_seed call in the sanity script); only the decreasing property is asserted, per the established pattern.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task b6v0w7dzn (composite), bcrwyfs08 (gate watcher)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (training); /tmp/exp058_composite_run1.log (gate/watchdog telemetry)
- **WandB**: N/A
- **Status**: completed-CONTAMINATED (relaunched as Run 2)
- **Started**: 2026-06-11 06:43
- **Ended**: 2026-06-11 (RC=0; integrity failed post-hoc)

Description:
- Single gated run of the fc-WD-×4 recipe on GPU 0 via the standing composite launcher (dual gates + watchdog). Expected: family signatures (D0 ∈ [21.5, 23.5]ms, 138–140 epochs, 13,400–13,515 steps, params 4,286,026, ~470–510s total). Decision read: best_test_acc vs pre-registered branches — ≥ 96.81 escalate to replicate-pair (MEAN); [96.41, 96.73] family-band (slope saturates at/before 5e-4, axis closed flat); (96.73, 96.81) no-improvement by protocol; < 96.41 over-constrained (optimum bracketed in (0, 2e-3), 5e-4 measured best, axis closed from above).

Observations:
- GATE_DECISION D0=22.5ms ∈ family band [21.5, 23.5], contention_thresh 28.1ms, projected 137 epochs — graph-unchanged expectation confirmed at launch (source: /tmp/exp058_composite_run1.log, GATE_DECISION line)
- **CONTAMINATED — integrity pre-condition FAIL**: mid-run contention episode at ticks 18–20 (windows 24.0/28.0/27.4ms, steps ~7100–7750) and tick 32 (30.0ms; raw dt samples 48/50/51/95ms in run.log) inflated charged step time → 12,916 steps / 134 epochs vs bands [13,300, 13,600] / [136, 141]. ~500 steps (~3.7% of budget) lost. Below the 4-window CONTENTION_KILL streak, so the watchdog correctly let it finish, but the integrity gate catches it post-hoc (source: /tmp/exp058_composite_run1.log ticks 18–20, 32; run.log dt histogram)
- Read NOT analyzable per protocol (infra-errors EXP-011/032: contaminated runs are relaunched byte-identically, never analyzed). For the record only: best 96.54 @ ep133 with a still-climbing tail at −500 steps — discarded as evidence.

Key Metrics:
- best_test_acc: 96.54 (CONTAMINATED — not a decision input) | 134 ep / 12,916 steps | 300.0s charged / 486.1s total | params 4,286,026 (source: run.log summary)

### Run 2

Metadata:
- **Job ID**: background task b4ak77ryd (composite), bmn82ub1a (gate watcher)
- **Log file(s)**: run.log (training); /tmp/exp058_composite_run2.log (gate/watchdog telemetry)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 (launched after ~64 min gate wait)
- **Ended**: 2026-06-11 (RC=0 at watchdog tick 34)

Description:
- Byte-identical relaunch of Run 1 (retry 1 of 2) after the Run 1 contention contamination. Same diff, same launcher, same pre-registered branches. run.log deleted before relaunch per protocol.

Observations:
- Gate held the launch ~64 min while a foreign compute app occupied GPU 0 at 99–100% util (the same job that contaminated Run 1 mid-flight; its memory footprint shrank 35.5→15.5GB over the wait until it exited). GATES_CLEAR ~poll 128; GATE_DECISION D0=22.5ms ∈ family band, contention_thresh 28.1ms, projected 137 epochs (source: /tmp/exp058_composite_run2.log)
- CLEAN run: one transient window 25.7ms at tick 12 (< 27 threshold; dt histogram shows 2 isolated blips of 35/60ms in 266 samples), all other windows ≤ 23.x, slow_streak 0 throughout, no kills, RC=0 (source: /tmp/exp058_composite_run2.log; run.log dt histogram)
- Trajectory family-shaped early (ep1 34.94 ≥ 30), converged plateau DEPRESSED: 96.18–96.24 over the last 8 evals at family-band test_loss (~0.187) — same level-depression-at-family-CE shape as EXP-057's relief direction (source: run.log eval lines)

Key Metrics:
- best_test_acc: 96.24 @ ep136 (source: run.log summary; branch (iv) — mean−2.1σ, well below family floor 96.41)
- final_test_acc: 96.23 / final_test_loss: 0.1874 (family band) @ ep138 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 497.2 | num_epochs: 138 | num_steps: 13,322 | num_params: 4,286,026 | peak_vram_mb: 1,613 (source: run.log summary)

## Verification Results

(Conditions evaluated on Run 2 — the only integrity-clean run. Run 1 is contaminated and excluded per protocol.)

### Conditions Checked

- **Integrity pre-condition** (plan-058 step 0): PASS on Run 2 — RC=0; D0 22.5 ∈ [21.5, 23.5]; no kill markers; max window 25.7 < 27 (single transient, slow_streak 0); `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds: 497.2` ≤ 600; 138 evals ≤ 138 epochs; epochs 138 ∈ [136, 141], steps 13,322 ∈ [13,300, 13,600]; ep1 34.94 ≥ 30; no NaN. (Run 1 FAILED integrity: steps 12,916 < 13,300, contention windows 28.0/27.4/30.0ms — relaunched, never analyzed.)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 96.24 (`grep "^best_test_acc:" run.log`). Pre-registered branch (iv): < 96.41 → over-constrained; optimum bracketed in (0, 2e-3), 5e-4 the measured best of three points (0 → 96.36; 5e-4 → family mean; 2e-3 → 96.24). First-failure-stop: no escalation.
- **Condition 2 — completes within budget**: PASS informationally (497.2 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (138/138, structural).

### Informational Metrics

- peak_vram_mb: 1,613.0 (family) | num_epochs: 138 | num_params: 4,286,026 (informational; Condition 1 failed)

## Errors & Dead Ends

### 2026-06-11 — Run 1 contaminated by foreign GPU-0 job below the kill threshold
- Error: `no error — RC=0, sane metric (96.54); steps 12,916 vs band ≥13,300; contention windows 28.0/27.4/30.0ms never reached the 4-consecutive CONTENTION_KILL streak`
- Root cause: a foreign compute app landed on GPU 0 mid-run (later observed directly at 99–100% util during the Run 2 gate wait) and time-sliced intermittently — episodes too short for the watchdog streak, long enough to cost ~500 steps (~3.7% of budget).
- Source: /tmp/exp058_composite_run1.log ticks 18–20 and 32; run.log dt histogram (48/50/51/95ms samples)
- Do NOT retry: never analyze a run whose step count is below the family band even when windows stay under the kill streak — the post-hoc step-ledger integrity check (infra-errors EXP-011, Protocol Findings step-count instrument) is the binding gate; relaunch byte-identically.

## Human Notes

> {Researcher can add comments, corrections, or context here}

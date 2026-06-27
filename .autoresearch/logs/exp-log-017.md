# EXP-017: Per-stage depth redistribution [3,3,3] → [2,3,4] at constant FLOPs

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (run completed cleanly; necessary condition 1 not met — 96.43 < 96.81)

## Implementation Notes

### Summary

Exactly the plan's three edits, nothing else (5-line diff). `NUM_BLOCKS = 3` became the tuple `(2, 3, 4)`; the three `_make_layer` calls in `ResNet.__init__` now index it (`num_blocks[0]/[1]/[2]`); the startup print's depth formula became `2 + 2 * sum(NUM_BLOCKS)` (still prints "ResNet-20"). `_make_layer` already supported arbitrary counts via its `strides` list construction, so no other code paths are touched. Syntax verified via `ast.parse` (deliberately not importing train.py — module level instantiates `Eval()`). All training constants and the compile/warmup/eval paths are byte-identical to baseline @ 1990397, so any metric delta is attributable to the allocation change alone.

### Surprises & Discoveries

- None at implementation time — the class was already parameterized in the right shape; the tuple change is the entire intervention. Expected params: 5,392,714 (4,286,026 − 73,984 stage-1 block + 1,180,672 stage-3 block), to be confirmed against the run.log params line within ~15s of launch.

### Decisions

- The plan's informational dt-gate (Milestone 3) deliberately does NOT kill on architecture-priced slowdowns — only the contention watchdog kills (4 consecutive >30ms windows). Rationale recorded in plan § Abort Criteria: a completed 510s run with a real metric beats a verdict-less early kill, per the EXP-007 precedent (even a 55-epoch run yielded a clean no-improvement verdict).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: b5k4crkc2 (composite background script: pre-check + train + inline watchdog)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 09:52:12 (GPU 0 confirmed free at launch by the script's pre-check)
- **Ended**: 2026-06-10 10:01:12 (TRAIN_EXIT rc=0)

Description:
- Single 300s-budget training run of the [2,3,4] depth-redistributed ResNet-20 4x on GPU 0 via the standard composite launcher (pre-launch GPU-0 free check + inline contention watchdog, auto-kill on 4 consecutive >30ms windows). Tests whether +1.11M params moved to stage 3 at provably-equal FLOPs lifts best_test_acc ≥ 96.81. Expected signatures if FLOPs-neutrality holds: params 5,392,714, windowed dt ≈ 22.4ms, ~135–139 epochs, total ~510s, VRAM ~1.6–1.8GB.

Observations:
- Pristine execution: watchdog emitted zero SLOW events (task b5k4crkc2 output — only LAUNCH/TRAIN_EXIT lines); post-hoc windowed profile 0 of 278 windows >30ms, mean 21.5ms — the run was FASTER than the 22.4ms baseline, confirming the FLOPs-neutral (indeed activation-traffic-reducing) prediction: 144 epochs / 13,950 steps vs baseline ~139/~13,400 (source: run.log summary block; profile command output).
- params line exactly as computed: `ResNet-20 (4x wide) | params: 5,392,714` — layer wiring correct (source: run.log head).
- startup_seconds 23.7 vs ~13 warm-cache baseline: compile cache miss from the new graph shape (expected, one-time); total 515.1s still well under the 600s cap.
- VRAM 1427.7MB vs baseline 1613.0MB — removing a stage-1 block (64ch@32×32 activations) saved more than the stage-3 block added, as predicted (source: run.log summary).
- Trajectory CONVERGED with a proper plateau: final eight evals flat in 96.2–96.43, best at ep 138 of 144, final ≈ best (Δ0.08) — this is a convergence-level deficit, NOT epoch starvation; the EXP-008 diagnostic (plateau-at-end = architecture problem) applies (source: run.log eval trail, eps 133–144).
- Note: the early-signal Monitor (b1o0kidjj) armed in a separate turn first-polled near run END (windows 13800–13950) — the known turn-scheduling pattern from infra-errors; harmless here since the inline watchdog was the active protection.

Key Metrics:
- best_test_acc: 96.43% @ ep 138/144 (source: run.log summary + eval trail) — baseline 96.71, bar 96.81: −0.28pp
- total_seconds: 515.1 | training_seconds: 300.0 | startup_seconds: 23.7 (source: run.log summary)
- num_epochs: 144 | num_steps: 13,950 | windowed dt mean 21.5ms, 0/278 >30ms (source: run.log + profile)
- peak_vram_mb: 1427.7 | num_params: 5,392,714 (source: run.log summary)
- final_test_acc: 96.35% | final_test_loss: 0.1917 (source: run.log summary)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition (contention sanity)**: num_epochs 144 within ~10% of clean projection (~139–144 at ≤22.4ms; run measured 21.5ms so 144 is exactly on-model) AND post-hoc windowed profile 0 of 278 windows >30ms — CLEAN, run is analyzable (source: profile command output; run.log).
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: `grep "^best_test_acc:" run.log` → 96.43%. **FAILED** (−0.28pp vs baseline, −0.38pp vs bar).
- **Condition 2 — total ≤ 600s, rc=0**: skipped — aborted after prior failure (informally observed: 515.1s, rc=0 — would have passed).
- **Condition 3 — eval at most once per epoch**: skipped — aborted after prior failure (informally observed: 144 eval lines = 144 epochs — would have passed).

### Informational Metrics

Not collected per protocol (necessary condition failed). Informal values from run.log for the report: peak_vram_mb 1427.7; num_epochs 144; num_params 5,392,714.

## Errors & Dead Ends

<!-- Append only. Never delete. Agent reads this before proposing any next action.
     Include source pointers for traceability. -->

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

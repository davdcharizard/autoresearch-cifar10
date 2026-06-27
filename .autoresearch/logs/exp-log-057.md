# EXP-057: Decouple the classifier from weight decay (fc.weight WD 5e-4 → 0)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-057
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-hunk diff to train.py exactly as planned (Milestone 1): the optimizer param-group construction now captures `fc_weight_id = id(base_model.fc.weight)` and excludes that one tensor from `decay_params`, adding it to `no_decay_params`. Optimizer call, schedule, warmup, loop all byte-identical. CPU sanity (`/tmp/exp057_sanity.py`, CUDA_VISIBLE_DEVICES="") passed all four checks: (a) total params 4,286,026 exact; (b) fc.weight in the WD=0 group only; (c) group numel ledger exact — decay 4,277,952 / no-decay 8,074 (= 5,514 BN+bias baseline + 2,560 fc.weight); (d) 3-step smoke at lr 0.01, losses 3.0979 → 1.7845 decreasing. No GPU probe needed per plan: the compiled graph is untouched (optimizer is eager; same two-group foreach SGD structure), so training signatures are expected byte-identical to the family.

### Surprises & Discoveries
- None at implementation time. The `id()`-membership pattern against `base_model` (EXP-055-validated) worked unchanged; the compiled wrapper shares the underlying Parameters.

### Decisions
- Sanity script instantiates the model directly and replicates the group construction rather than importing `main()` (which would download data / require CUDA paths) — same pattern as EXP-055/056 sanity scripts.
- No extra in-run instrumentation (e.g., printing fc weight norm): keeping the loop byte-identical preserves the attribution guarantee; mechanism engagement is structural (group membership asserted in sanity) rather than dynamic, so the EXP-055 physical-signature rule is satisfied by the EXPECTED ABSENCE of any dt change.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task bqj1jlg2t (composite), b8e7g51t1 (gate watcher)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (training); /tmp/exp057_composite_run1.log (gate/watchdog telemetry)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 06:25
- **Ended**: 2026-06-11 (RC=0 at watchdog tick 33, ~486s after launch)

Description:
- Single gated run of the fc-WD-decoupled recipe on GPU 0 via the standing composite launcher (dual gates: zero GPU-0 compute apps AND host load < 60; watchdog: GATE_KILL D0 > 26ms, CONTENTION_KILL 4 windows > max(26, D0×1.25), STARTUP_KILL by tick 12, wall cap 660s). Expected: family signatures (D0 ∈ [21.5, 23.5]ms with NO probe offset since the graph is unchanged, 138–140 epochs, 13,400–13,515 steps, params 4,286,026, ~470–510s total). Decision read: best_test_acc vs pre-registered branches — ≥ 96.81 escalate to replicate-pair (MEAN); [96.41, 96.73] family-band null (corner closed redundant); (96.73, 96.81) no-improvement by protocol; < 96.41 sign-down (fc WD load-bearing, closed from below).

Observations:
- Gates passed on first poll (apps=0, load=11); GATE_DECISION D0=22.7ms ∈ family band [21.5, 23.5] with contention_thresh 28.4ms, projected 136 epochs — graph-unchanged expectation confirmed at launch (source: /tmp/exp057_composite_run1.log, GATE_DECISION line)
- PRISTINE run: all watchdog windows 21.7–22.7ms, slow_streak 0 throughout, no kill markers, PROC_EXITED RC=0 at tick 33 (source: /tmp/exp057_composite_run1.log ticks 3–32)
- Trajectory family-shaped: ep1 36.31 (≥ 30 tripwire), converged plateau ~96.22–96.36 over the final 8 evals — plateau LEVEL sits ~0.2 below the family plateau (~96.5–96.7) at family-band test_loss, i.e., a level depression, not a trajectory anomaly (source: run.log eval lines ep1–3, ep133–140)
- Mechanism attribution clean by construction: dt, epochs (140), steps (13,511), startup (11.9s), VRAM (1,613MB) all family-identical; the only changed quantity was fc.weight decay pressure (source: run.log summary block)

Key Metrics:
- best_test_acc: 96.36 @ ep136 (source: run.log summary; branch (iv) — below family floor 96.41)
- final_test_acc: 96.34 / final_test_loss: 0.1905 (family band) @ ep140 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 485.7 | num_epochs: 140 | num_steps: 13,511 | num_params: 4,286,026 | peak_vram_mb: 1,613 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Integrity pre-condition** (plan-057 step 0): PASS — RC=0; D0 22.7 ∈ [21.5, 23.5]; no GATE_KILL/CONTENTION_KILL/STARTUP_KILL, no window > 27ms (max 22.7); `num_params: 4,286,026` exact; `training_seconds: 300.0`; `total_seconds: 485.7` ≤ 600; 140 eval lines ≤ 140 epochs (once-per-epoch ceiling); epochs 140 ∈ [136, 141], steps 13,511 ∈ [13,300, 13,600]; ep1 36.31 ≥ 30; zero NaN matches. (source: run.log + /tmp/exp057_composite_run1.log)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 96.36 (`grep "^best_test_acc:" run.log`). Pre-registered branch (iv): < 96.41 → sign-down, fc WD's margin cap is load-bearing regularization; corner closed from below. First-failure-stop: no escalation.
- **Condition 2 — completes within budget**: PASS informationally (total_seconds 485.7 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (140 evals / 140 epochs, structural).

### Informational Metrics

- peak_vram_mb: 1,613.0 (family) | num_epochs: 140 | num_params: 4,286,026 (collected informationally; Condition 1 failed so these do not gate anything)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

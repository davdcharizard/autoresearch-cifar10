# EXP-018: Zero-init residual — γ=0 in each BasicBlock's final BN (bn2)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (run completed cleanly; necessary condition 1 not met — 95.72 < 96.81)

## Implementation Notes

### Summary

Exactly the plan's single addition: after `self.apply(self._weights_init)` in `ResNet.__init__`, a 5-line block (2 comment + 3 code) zeroes `bn2.weight` for every `BasicBlock` via `init.zeros_()`. The `init` module (`torch.nn.init`) was already imported at top of file — no import change needed. Diff is +5 lines, nothing else; all training constants, architecture, schedule, augmentation, and compile paths byte-identical to baseline @ 1990397. Syntax verified via `ast.parse` (not importing train.py — module level instantiates `Eval()`).

### Surprises & Discoveries

- None at implementation time. Confirmed the plan's WD subtlety holds in the actual code: `bn2.weight` has ndim=1 so it lands in the `no_decay_params` group (weight_decay=0.0) — the zeroed γs are free to grow via gradients and are not pinned at 0 by decay.

### Decisions

- Placed the zeroing loop inside `ResNet.__init__` (immediately after `self.apply(self._weights_init)`) rather than in `main()`, so the init scheme is a property of the model class and runs before `.to(device)`/compile wrapping — guaranteeing the eager `base_model` and compiled `model` share the same zero-γ weights (they share storage anyway, but init-before-everything removes any ordering question).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: b4lrsii9b (composite background script: pre-check + train + inline watchdog)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 10:35:05 (GPU 0 confirmed free at launch by the script's pre-check)
- **Ended**: 2026-06-10 10:43:35 (TRAIN_EXIT rc=0)

Description:
- Single 300s-budget run of the baseline recipe with zero-γ init in all nine bn2 layers, on GPU 0 via the standard composite launcher (pre-launch free check + inline contention watchdog). Tests whether identity-at-init eases the early high-LR phase and lengthens the converged plateau, lifting best_test_acc ≥ 96.81. Expected signatures (must match baseline exactly — identical compiled graph): params 4,286,026, windowed dt ≈ 22.4ms, ~139 epochs, VRAM ≈ 1613MB, total ~480–515s. Early trail (eps 1–20) recorded for the faster-onset claim; nearest same-throughput comparator EXP-017 (63.76@5, 75.06@8).

Observations:
- Pristine execution: zero watchdog SLOW events; post-hoc windowed profile 0 of 268 windows >30ms, mean 22.3ms; 139 epochs / 13,475 steps; startup 13.2s (warm compile cache — graph identical to baseline as predicted); VRAM 1613.0MB and params 4,286,026 byte-identical to baseline. Attribution is perfect: the only delta is the init (source: task b4lrsii9b output; run.log summary; profile command output).
- **The faster-onset hypothesis INVERTED**: the early trail is dramatically SLOWER, not faster — ep1 18.37, ep5 35.26, ep8 55.16, ep10 68.01 vs the same-throughput comparator EXP-017's 63.76@5 / 75.06@8. Identity-at-init meant the network spent the hottest, most valuable part of the time-keyed schedule with near-zero residual expressivity while the nine γs grew from 0 — effectively a stem-only net during warmup/peak (source: run.log eval trail eps 1–10).
- Mid-run it was still far behind (78.91 @ ep60, 89.97 @ ep100 — baseline-family runs are ~6–8pp ahead at these marks) and converged to a FLAT plateau (final eight evals 95.63–95.72, final ≈ best −0.05) a full −0.99pp below baseline: a convergence-level deficit by the EXP-008 diagnostic, the early heat was simply spent without learning (source: run.log eval trail).
- Mechanism reading: literature gains for zero-γ are measured at fixed EPOCHS on long ImageNet schedules where the turn-on cost amortizes and early stability enables bigger peak LRs; under a fixed WALL CLOCK with the peak LR already certified optimal, zero-γ is a pure DEFERRAL of effective capacity — the same mechanism that closed hot-LR, capacity, and heavy-regularization moves (project-insights Medium).

Key Metrics:
- best_test_acc: 95.72% @ ep 135/139 (source: run.log summary + eval trail) — baseline 96.71, bar 96.81: −0.99pp
- total_seconds: 489.7 | training_seconds: 300.0 | startup_seconds: 13.2 (source: run.log summary)
- num_epochs: 139 | num_steps: 13,475 | windowed dt mean 22.3ms, 0/268 >30ms (source: run.log + profile)
- peak_vram_mb: 1613.0 | num_params: 4,286,026 — both byte-identical to baseline (source: run.log summary)
- final_test_acc: 95.67% | final_test_loss: 0.2028 (source: run.log summary)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition (contention sanity)**: num_epochs 139 = clean projection exactly; 0 of 268 windows >30ms — CLEAN, run is analyzable (source: profile command output).
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: `grep "^best_test_acc:" run.log` → 95.72%. **FAILED** (−0.99pp vs baseline; largest deficit since EXP-008).
- **Condition 2 — total ≤ 600s, rc=0**: skipped — aborted after prior failure (informally: 489.7s, rc=0 — would have passed).
- **Condition 3 — eval at most once per epoch**: skipped — aborted after prior failure (informally: 139 eval lines = 139 epochs — would have passed).

### Informational Metrics

Not collected per protocol (necessary condition failed). Informal values from run.log for the report: peak_vram_mb 1613.0; num_epochs 139; num_params 4,286,026 — all identical to baseline, as the identical-graph claim requires.

## Errors & Dead Ends

<!-- Append only. Never delete. Agent reads this before proposing any next action.
     Include source pointers for traceability. -->

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

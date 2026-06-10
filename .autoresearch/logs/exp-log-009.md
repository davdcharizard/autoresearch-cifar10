# EXP-009: Compiled k=5 WideResNet (capacity, threading the k4–k6 gap)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-009
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Two edits to `train.py` only (Milestone 1): (1) `WIDTH_MULT` 4→5 (stages {80,160,320}); (2) added
`compiled_model = torch.compile(model, mode="reduce-overhead")` after the model is on device + num_params printed,
and routed the training forward through `compiled_model` (`outputs = compiled_model(inputs)`); eval unchanged on the
eager `model`. Parse-clean, ruff clean, diff is train.py-only (+8/-2), eval line still eager, seed still 42.
Param count to be confirmed at runtime (expected 6,712,314 — pre-computed offline from the architecture).

### Surprises & Discoveries
- (none at implementation time — this is the validated EXP-007/008 compile pattern applied to a wider net.)

### Decisions
- **k=5, not k=6**: deliberately the untested intermediate width. EXP-004's k=6 was compute-bound (eager 22ms →
  35 epochs → underfit). k=5 is ~1.56× k=4 FLOPs (vs k=6's 2.25×), so with the compile enabler it should keep a
  fair epoch count (~55–65) rather than starve. Single conceptual variable vs compiled-k4 (EXP-007): width.
- **Compile as enabler, attribution preserved**: EXP-007 showed compiled-k4 = 95.92 ≈ baseline (null standalone
  accuracy effect), so any gain over ~96.0 is attributable to the k=5 width, not the compile.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID; local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the EXP-003 recipe with width raised to k=5 (6.7M params) and the validated torch.compile(reduce-overhead)
  enabler on the training forward. Tests whether the project's dominant lever (capacity) — which stalled at k=6 only
  because of the epoch wall — pays off at an intermediate width now that compile lifts that wall ~30%. Expect
  ~6,712,314 params, clean compile, ~55–65 epochs, clean run < 600s. Success = best_test_acc ≥ 96.10.

Observations:
- Clean startup; params **6,712,314** (= k=5, +56.1% vs k=4), clean compile, no traceback (run.log head).
- **Steady-state dt = 18ms/step (~7,000 img/s)**, stable from step 50 (run.log L step00050–00450). This is
  notably worse than the ~11–12ms projected: k=5 is more compute-bound than expected, so reduce-overhead's
  CUDA-graph win (which targets launch overhead) helps less than it did on launch-bound k=4 (8ms). For reference
  compiled-k4 was 8ms, eager k=6 was 22ms — compiled-k5 at 18ms sits closer to the k=6 compute regime.
- Implication: ~7s/epoch → tracking toward **~40 epochs**, the EXP-004-style epoch-starvation zone the plan
  flagged (record, not abort). A no-improvement near/below 96.0 would therefore be partly epoch-masked.
- Loss decreasing normally (3.0→1.98 by step 450), no NaN; ep1 eval 28.74% (early, expected).

Key Metrics:
- best_test_acc: **94.21%** @ best (ep 39) — far BELOW baseline 96.00 and bar 96.10 (run.log summary).
- **num_epochs: 41** / num_steps 15,674 — ~half of k=4's 77 epochs. Severe under-training: the k=5 net did not
  converge in the budget. (eval count = 41 = num_epochs → eval once/epoch confirmed.)
- final_test_acc 94.12; **final_test_loss 0.2440** — well above k=4's converged 0.204 and compiled-k4's 0.208,
  i.e. still under-fit (loss was still falling: 0.246→0.244 over the last evals).
- dt 18ms/step steady (~7,000 img/s); num_params 6,712,314 (k=5 confirmed); peak_vram 599.9 MB; total 363.3s.
- Context: compiled-k5 (94.21 @ 41 ep) regressed below BOTH compiled-k4 (95.92 @ 89 ep) and even eager k=6
  (95.26 @ 35 ep, EXP-004). The k5/k6 inversion is volatile under-trained-regime noise; the unmistakable trend is
  width↑ → effective-epochs↓ → accuracy↓ at this budget. Compile's ~30% boost (which mostly helped launch-bound
  k=4) is far too small to make k≥5 trainable here — compiled-k5 18ms = 2.25× compiled-k4 8ms (FLOP ratio only 1.56×).

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 94.21% present, total_seconds 363.3 < 600, no
  traceback (run.log summary; tracebacks=0).
- **Cond 2 — metric ≥ 96.10**: **FAIL**. 94.21 < 96.10 (also far below 96.00 baseline, −1.79pp). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2 per protocol.

### Informational Metrics

- num_epochs 41 / num_steps 15,674 (KEY confound: epoch-starved as the plan flagged — only ~half of k=4's 77).
- final_test_loss 0.2440 (under-fit). peak_vram_mb 599.9. img/s ~7,000 (dt 18ms). num_params 6,712,314.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

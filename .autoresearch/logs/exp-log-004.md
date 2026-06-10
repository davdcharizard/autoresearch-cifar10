# EXP-004: Increase width to k=6

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-004
- **Commit**: (none — no-improvement; changes discarded)
- **PR**: (n/a)
- **Outcome**: failed (no-improvement: 95.26 < baseline 96.00)

## Implementation Notes

### Summary
One-line change in `train.py`: `WIDTH_MULT` 4→6 (stages {96,192,384}). Param count 4.3M → **9,659,802**
(~2.25×). Everything else fixed (Cutout via cutout_batch, recipe, projection shortcuts, seed). Syntax + ruff pass.

### Surprises & Discoveries
- None at implementation. The width multiplier was already parametrized (EXP-001), so this is a trivial edit.

### Decisions
- Kept all recipe knobs fixed (PEAK_LR 0.2, WD 1e-4) to isolate the capacity effect; WD tuning for the larger
  model is a documented follow-up.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash (filled at launch), local GPU 0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-08
- **Ended**: pending

Description:
- Running the k=6 WideResNet (~9.66M params) + Cutout under the fixed 300s budget on GPU 0, recipe unchanged.
  Tests EXP-004 hypothesis that more capacity lifts best_test_acc above 96.00%. KEY signal: num_epochs (expect
  ~55–70 given the near-flat width/throughput relation; a large drop would flag underfit risk).

Observations:
- Started cleanly: 9,659,802 params (~2.25× k=4). **dt jumped to ~22ms/step** (vs k=4 ~10ms) → k=6 is
  compute-bound (unlike k=1→k=4 which stayed memory-bound). Projecting only **~33 epochs** in 300s — into
  underfit-risk territory for a 9.7M model. Loss ~1.07 @ ep4, healthy, no NaN (run.log early lines).

Key Metrics:
- best_test_acc: **95.26%** @ epoch 35 (baseline 96.00%, **−0.74 pp** — REGRESSION) (source: run.log summary)
- num_epochs: **35** | num_steps: 13,314 (vs k=4 EXP-003's 77 / 29,931 — k=6 is compute-bound, ~22ms/step)
- final_test_loss: 0.2234 (vs EXP-003 0.204) | peak_vram_mb: 729.8 | num_params: 9,659,802
- training_seconds: 300.0 | total_seconds: 356.0

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS. `best_test_acc:` present, total_seconds 356.0 < 600,
  no traceback (run.log summary).
- **Condition 2 — metric improvement (≥ 96.10)**: **FAIL**. best_test_acc = 95.26 < 96.10 (below baseline 96.00
  by 0.74 pp). → no-improvement; remaining condition not evaluated.
- **Condition 3 — no constraint violations**: skipped — aborted after prior failure.

Verdict: no-improvement (valid clean run, metric below baseline). Cause: k=6 (9.7M) is compute-bound at
~22ms/step → only 35 epochs fit → underfit vs the k=4 model's 77 epochs.

### Informational Metrics
- (not collected — necessary conditions did not all pass)

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

# EXP-005: Weight decay 1e-4 → 5e-4 (k=4 + Cutout)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-005
- **Commit**: (none — no-improvement; changes discarded)
- **PR**: (n/a)
- **Outcome**: failed (no-improvement: 96.05 < bar 96.10)

## Implementation Notes

### Summary
One-line change in `train.py`: `WEIGHT_DECAY` 1e-4 → 5e-4 (WRN-standard). k=4 model, Cutout, and all other
recipe knobs unchanged. Syntax + ruff pass; WIDTH_MULT confirmed still 4.

### Surprises & Discoveries
- None.

### Decisions
- Isolated WD change only (no Cutout/mixup change) for clean attribution; throughput-neutral so epochs stay ~77.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash ID be9j0idhr (local, GPU 0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-08
- **Ended**: pending

Description:
- Running k=4 WideResNet + Cutout with WEIGHT_DECAY 5e-4 (vs 1e-4) under the 300s budget on GPU 0, recipe
  otherwise unchanged. Tests EXP-005 hypothesis that stronger WRN-standard L2 regularization lifts
  best_test_acc above 96.00%. Expect ~77 epochs (WD is throughput-neutral).

Observations:
- Started cleanly: 4,299,866 params; loss healthy (~1.14 @ ep5, higher than EXP-003's ~0.95 — expected with 5x WD); LR at peak; no NaN (run.log early lines).

Key Metrics:
- best_test_acc: **96.05%** @ epoch 65 (baseline 96.00%, +0.05 pp — below the +0.1 bar) (run.log summary)
- final_test_loss: **0.1956** (vs EXP-003 0.204 — stronger WD DID reduce overfitting) | peak_vram_mb: 492.1
- num_epochs: **65** | num_steps: 25,128 (vs EXP-003's 77 / 29,931 — ~17ms/step this run vs ~10ms; WD is
  compute-neutral, so the shortfall is a transient throughput confound, not the WD change)
- total_seconds: 376.4

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS (best_test_acc present, total 376.4 < 600, 0 tracebacks).
- **Condition 2 — metric improvement (≥ 96.10)**: **FAIL**. best_test_acc 96.05 < 96.10 (only +0.05 over
  baseline 96.00, within noise). → no-improvement; remaining condition not evaluated.
- **Condition 3**: skipped — aborted after prior failure.

Verdict: no-improvement. WD 5e-4 lowered eval loss (0.204→0.196, real regularization effect) but accuracy gain
(+0.05) was within noise and below the +0.1 bar. Confound: only 65 epochs fit this run (vs 77) due to ~17ms/step
(transient; GPUs idle at check time, WD is compute-neutral) — fewer epochs may have masked a small WD benefit.

### Informational Metrics
- (not collected — necessary conditions did not all pass)

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

# EXP-019: SWA with a constant-LR averaging tail (proper Stochastic Weight Averaging)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-019
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented SWA per plan-019, all in train.py (6 edits): (1) imported `AveragedModel` from
`torch.optim.swa_utils` (core torch, no new dep); (2) added `SWA_START_FRAC=0.75`, `SWA_LR=0.05`,
`BN_RECOMPUTE_BATCHES=50`; (3) rewrote `lr_at_fraction` to cosine-anneal PEAK_LR→SWA_LR over
[WARMUP_FRAC, SWA_START_FRAC] then hold a constant SWA_LR for the tail (continuous at the join:
cos(pi)=−1 ⇒ returns SWA_LR); (4) added `recompute_bn()` — resets BN stats (momentum=None,
cumulative average) and forwards `BN_RECOMPUTE_BATCHES` augmented training batches (channels_last
+ Cutout + bf16 autocast) through the SWA model, matching the training input distribution; (5)
constructed `swa_model = AveragedModel(model)` once before the loop (eager, separate from the
compiled handle); (6) branched the per-epoch eval — in the constant-LR tail, `update_parameters` +
`recompute_bn` + evaluate the SWA model; in the main phase, evaluate the raw model as before. Exactly
one `evaluate()` per epoch in both branches. Maps to Milestone 1 (ruff clean, parses, scope=train.py only).

### Surprises & Discoveries
- `AveragedModel` defaults to `use_buffers=False` (confirmed via inspect on torch 2.11.0) — it averages
  parameters only and leaves BN running stats at their deep-copy-time values, so BN-recompute is mandatory
  before eval. This is exactly why EXP-006-style "just average and eval" would be wrong here.
- BN-recompute and the per-epoch eval run AFTER the inner step loop, so they do NOT accumulate into
  `total_training_time` (which gates the LR schedule and the 300s budget). Consequence: training still gets
  the full 300s of step-time regardless of BN overhead — SWA is genuinely compute-neutral for *training*;
  the ~0.3s/epoch BN cost only adds to wall-clock (total_seconds), staying far under the 600s hard limit.

### Decisions
- **SWA_LR=0.05 (peak/4), SWA_START_FRAC=0.75**: a moderate non-zero floor that keeps the iterate moving
  through the flat region (the precondition EXP-006 lacked) while leaving ~25% of the budget (~15–21 epochs)
  for averaging — enough snapshots for the average to matter. Standard SWA range (Izmailov 2018).
- **BN recompute mirrors the training input pipeline** (channels_last + Cutout + bf16 autocast) rather than
  using clean inputs, so the averaged model's BN stats match the distribution its weights were trained on.
- **Truncated BN pass (50 batches) instead of `swa_utils.update_bn` (full pass)**: bounds the per-epoch
  overhead to ~0.3s; 6.4k images is ample for stable BN mean/var estimates.
- **Eval the SWA model only in the tail; raw model in the main phase**: preserves the ≤1-eval/epoch constraint
  and lets the main phase's raw-model evals still contribute to best_test_acc.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Running the EXP-012 baseline recipe with the LR schedule changed to cosine→0.05-floor then a constant-0.05
  SWA averaging tail (final 25%), evaluating the BN-recomputed weight-average in the tail. SWA adds no model
  params (4,299,866 unchanged) and is training-compute-neutral, so expect ~84–91 training epochs (~8ms/step)
  with ~15–21 SWA snapshots in the tail. Hypothesis: the flat-region weight average lifts best_test_acc above
  the 96.32 bar by generalizing better than the single cosine-to-0 endpoint. Main risk: forgoing cosine-to-0
  sharpening could mildly regress if averaging under-compensates.

Observations:
- Clean startup: `params: 4,299,866` (UNCHANGED — SWA adds no model params), clean compile, no traceback, no NaN.
- LR schedule correct: decayed 0.20 → 0.05, then HELD CONSTANT at exactly `lr: 0.0500` through the tail (ep 68–91, steps ~34400–35200) (source: run.log step lines, ep 89–91). The terminal-LR floor — the precondition EXP-006 lacked — was supplied as designed.
- Tail fired at ep 68 (~75% of 91 epochs): 67 `[raw]` evals (main phase) + 24 `[swa]` evals (tail) = 91 = num_epochs → exactly one evaluate() per epoch (constraint satisfied) (source: run.log eval-line counts).
- SWA mechanism engaged strongly: the un-annealed raw iterate at the 0.05 floor dropped to 91.83% (ep 67), but the BN-recomputed weight average recovered to 93.95% (ep 68) → climbed monotonically to **95.97% (ep 91)** as snapshots accumulated — a ~+4pp lift over the raw iterate, confirming flat-region averaging works exactly as theorized (and validating the EXP-006 diagnosis that cosine-to-0 was why averaging was a no-op there).
- SWA model had the LOWEST final_test_loss in the project: **0.1788** (vs baseline 0.195 / compiled-k4 0.208) — a flatter/better-calibrated minimum — but the lower loss did NOT convert to higher top-1 accuracy.
- Throughput-neutral: 8ms/step, ~15,600 img/s, 91 epochs (identical to EXP-012's 91) — a fair same-budget test. total_seconds 421.1 < 600 (BN-recompute + tail eval overhead modest, training got the full 300s).

Key Metrics:
- best_test_acc: **95.97%** @ ep 91 (source: run.log summary) — vs baseline 96.22 (**−0.25pp**); below the 96.32 bar.
- final_test_acc: 95.97% | final_test_loss: 0.1788 (lowest in project) @ ep 91 (source: run.log summary)
- num_epochs: 91 | num_steps: 35,203 | num_params: 4,299,866 | peak_vram_mb: 469.3 | training_seconds: 300.0 | total_seconds: 421.1

## Verification Results

### Conditions Checked
- **Cond 1 — primary metric clears bar**: **FAIL**. best_test_acc = 95.97% < 96.32 bar (baseline 96.22 + 0.1).
  Δ = **−0.25pp** vs baseline. → verdict no-improvement. (Decisive condition; evaluated first.)
- **Cond 2 — clean completion within budget**: PASS (recorded for completeness). best_test_acc and total_seconds
  present; total_seconds 421.1 < 600; Traceback count 0; summary block printed (source: run.log).
- **Cond 3 — no constraint violations**: PASS (recorded for completeness). `git diff --name-only` = train.py only;
  num_params 4,299,866 unchanged; eval-count 91 == num_epochs 91 (one evaluate() per epoch — 67 raw + 24 swa);
  no new deps (`torch.optim.swa_utils` is core torch); seed 42 intact.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> (none)

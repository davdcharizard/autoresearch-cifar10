# EXP-053: AugMix(w2,d1) severity 3→6 — push op magnitude on the new winner

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-053
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (clean run; verdict no-improvement — best_test_acc 96.29 < 96.44 bar)

## Implementation Notes

### Summary
One-keyword change in `train_tf`: `AugMix(mixture_width=2, chain_depth=1)` → `AugMix(mixture_width=2, chain_depth=1, severity=6)` (default severity is 3). Built on the EXP-052 winner (merged on autoresearch/dev). Cutout, model, optimizer, schedule, seed, batch, compile, and all-image coverage unchanged. Smoke tests passed: AST OK; diff = train.py only (the severity keyword + comment); AugMix sev6 runs on 5 CIFAR samples → (3,32,32); num_params 4,299,866.

### Surprises & Discoveries
None. severity is a magnitude scalar (op count unchanged), so feasibility is not in question — it is CPU-neutral vs the EXP-052 w2,d1 base that completed in 571.9s (probed w2,d1,sev5 = 12.1ms/batch ≈ w2,d1 12.6ms).

### Decisions
Chose severity=6 (2× the default 3) as a meaningful but not extreme magnitude increase; AugMix's clean-image convex mix bounds the distribution shift, reducing over-augmentation risk. Single-variable test (only severity changes) for a clean read on the magnitude axis.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — local background PID)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (exit 0, 580.4s wall)

Description:
- Runs AugMix(w2,d1,severity=6) training on idle GPU 1 within the 300s Σdt budget. Tests whether stronger per-op magnitude adds useful augmentation diversity on the new 96.34 base (bar 96.44). Expect dt steady ~8ms, wall ~570s (CPU-neutral vs EXP-052), ~91 epochs. Early-load wall check at Milestone 2 (should pass easily — no added CPU cost).

Observations:
- Healthy. dt steady 8ms (GPU step unchanged). ep1 test_acc 45.45% (normal, ≈ baseline ~45.7% — severity=6 did not destabilize early convergence). Real-load eval-inclusive wall ~15.6ms/step at 11.6% → projected total ~533s, well under the 600s limit (feasibility confirmed — severity is CPU-neutral). No NaN. Letting run complete. (source: run.log)

Key Metrics:
- best_test_acc: 96.29% @ ep88 (source: run.log "eval ep 88"; summary)
- final_test_acc: 96.23% @ ep91; final_test_loss: 0.1961 (LOWER than EXP-052's 0.2010 — sev6 improved loss)
- total_seconds: 580.4s (wall); training_seconds: 300.0; num_epochs: 91; num_steps: 35,447; num_params: 4,299,866; peak_vram: 453.8 MB
- dt dist: 656×8ms, 48×9ms, 2×10ms, 2×11ms (steady 8ms — CPU-neutral as predicted)

## Verification Results

### Conditions Checked
- **Cond 1 — best_test_acc ≥ 96.44 (baseline 96.34 + 0.1)**: 96.29% → **FAIL** (−0.05pp vs baseline, −0.15pp vs bar). Necessary condition not met → no-improvement.
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failure (note: run DID complete cleanly, 580.4s < 600, no NaN, params unchanged — not a crash; this is a valid no-improvement).
- **Cond 3 — no hard-constraint violations**: skipped — aborted after Cond 1 failure (note: diff = train.py only, scope clean).

### Informational Metrics
- Not collected (necessary condition failed). For the record: final_test_loss 0.1961 < EXP-052's 0.2010 and ≈ original baseline 0.195 — severity=6 lowered loss but did NOT raise top-1 (a polish-vs-top1 signature). num_epochs 91 = EXP-052 (CPU-neutral confirmed).

## Errors & Dead Ends

## Human Notes

> {none — autopilot}

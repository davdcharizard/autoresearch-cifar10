# EXP-054: Intermittent full-strength AugMix via RandomApply(p=0.5)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-054
- **Commit**: 86161d9 (on autoresearch/exp-054, merged to autoresearch/dev)
- **PR**: N/A — repository is local-only by design (no git remote); commits kept local
- **Outcome**: completed

## Implementation Notes

### Summary
One-line change in `train_tf`: `AugMix(mixture_width=2, chain_depth=1)` → `RandomApply([AugMix()], p=0.5)` — apply the full default AugMix (w3,d-1) to ~50% of images. Built on the EXP-052 winner. Cutout (GPU), model, optimizer, schedule, seed, batch, compile unchanged. num_params 4,299,866. Smoke tests passed (AST, scope=train.py only, 10 samples run, params unchanged).

### Surprises & Discoveries
Dataloader feasibility probed in planning (8 workers, calibrated against EXP-052 where isolated 12.6ms → actual 571.9s): RandomApply(w3) p=0.5 = 12.9ms/batch → ~585s (tight, ~15s margin); base-only = 3.8ms; p=0.4 = 11.1ms → ~517s (safe fallback). p=0.5 is feasible but tight → early real-load wall gate required.

### Decisions
Chose p=0.5 (half the images get genuine 3-chain w3 diversity) as the primary, with an early-abort gate (project wall > 585s → fall back to p=0.4). Maximizes diversity exposure while protecting the 600s wall constraint (a breach = invalid).

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
- **Ended**: 2026-06-09 (exit 0, 593.0s wall)

Description:
- Runs RandomApply([AugMix()], p=0.5) training on idle GPU 1 within the 300s Σdt budget. Tests whether full 3-chain AugMix on ~50% of images (the live chain-count diversity lever, EXP-053) beats uniform w2,d1 (bar 96.44 vs baseline 96.34). Expect dt steady ~8ms; wall ~585s (tight — early gate at projected >585s → p=0.4 fallback). Watch coverage-reduction risk (half images get no photometric aug).

Observations:
- FEASIBILITY GATE PASSED at p=0.5 (no fallback needed). Real-load eval-inclusive ~16.0ms/step at 11.4% → projected total ~535s, under the 585s gate and 600s limit. dt steady 8ms (GPU step unchanged). ep1 test_acc 47.45% (normal, ≈/slightly above baseline ~45.7%). No NaN. Letting run complete. (source: run.log; wall measured in conversation log)

Key Metrics:
- best_test_acc: 96.45% @ ep88 (source: run.log "eval ep 88"; summary)
- final_test_acc: 96.41% @ ep91; final_test_loss: 0.1968 (< EXP-052's 0.2010 — loss AND top-1 both improved)
- total_seconds: 593.0s (wall; tight — 7s under limit; projection was 535s, RandomApply per-batch variance widened it); num_epochs: 91; num_steps: 35,328; num_params: 4,299,866; peak_vram: 453.8 MB
- dt dist: 633×8ms, 68×9ms, few 10-18ms (steady 8ms — GPU step unchanged)

## Verification Results

### Conditions Checked
- **Cond 1 — best_test_acc ≥ 96.44 (baseline 96.34 + 0.1)**: 96.45% → **PASS** (+0.11pp vs baseline; clears the +0.1 bar by 0.01pp — marginal). (source: run.log summary)
- **Cond 2 — clean completion within budget**: summary printed, total_seconds 593.0 < 600 ✓ (tight, 7s margin), num_params 4,299,866 ✓, no NaN/traceback (grep 0) → **PASS**. (source: run.log)
- **Cond 3 — no hard-constraint violations**: `git status --porcelain` = ` M train.py` only; eval/prepare untouched; AugMix/RandomApply torchvision-native (no new dep); seed 42 unchanged; eval once/epoch → **PASS**.
- **All necessary conditions PASS → Outcome: completed; verdict improvement (marginal).**

### Informational Metrics
- delta vs baseline 96.34: **+0.11pp**. num_epochs 91 (Σdt budget unaffected). final_test_loss 0.1968 < EXP-052's 0.2010 — unlike the severity null (EXP-053), BOTH loss and top-1 improved, consistent with a real (if small) diversity gain from full 3-chain AugMix on the subset. peak_vram 453.8 MB unchanged.
- CAVEAT (for analysis): +0.11pp is within the ±0.25pp noise band, and clears the bar by only 0.01pp; the 593s wall is tight with large run-to-run variance (projection 535 vs actual 593). Confidence moderate; replication advisable.

## Errors & Dead Ends

## Human Notes

> {none — autopilot}

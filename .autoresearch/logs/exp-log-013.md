# EXP-013: Reflection padding for RandomCrop

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-013
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 1 failed — 626.9s > 600s cap; metric 96.57 would also have missed; clean run, double research negative)

## Implementation Notes

### Summary

Exactly per plan-013 Milestone 1: one-argument change on branch `autoresearch/exp-013` — `transforms.RandomCrop(32, padding=4)` → `transforms.RandomCrop(32, padding=4, padding_mode="reflect")` in the `train_tf` Compose. `py_compile` passed; `git diff --stat` confirms a 1-line modification. The transform runs CPU-side in the 8 persistent workers before ToTensor (PIL path), so throughput is expected byte-identical to baseline (EXP-004 precedent: TA's far heavier PIL work was fully worker-absorbed).

### Surprises & Discoveries

- None — single-argument diff, fully idle node at launch (0 compute apps, 0% util on both GPUs).

### Decisions

- None beyond plan. Launched immediately into the idle window per the EXP-012 decision precedent (clean windows are transient on this shared node).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task bxk0hsvm4 (local, GPU 0 via CUDA_VISIBLE_DEVICES=0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed (exit 0) but BUSTED the 600s wall-clock cap — 626.9s total
- **Started**: 2026-06-10 ~08:00
- **Ended**: 2026-06-10 (626.9s later, exit 0)

Description:
- Single run of baseline 1990397 with one change: RandomCrop pads with reflected image content instead of zeros. Tests whether removing zero-band artifacts (statistics absent from the test set) from border crops converts wasted invariance capacity into signal. Expected: dt 22ms / ~139 epochs / VRAM ~1613MB all byte-identical to baseline (CPU-side change), epoch-1 eval ~30–36%, converged final≈best tail, best_test_acc ≥ 96.81 if the hypothesis holds.

Observations:
- TRAINING CLEAN: 139 epochs / 13418 steps at cum dt 22.4ms (windows 18–24ms, zero SLOW events, VRAM 1613.0 and params identical to baseline) — GPU-side execution byte-identical as predicted; contention sanity PASSED. (source: run.log pct windows; detector bbr5zwtcm empty)
- WALL-CLOCK CAP BUSTED: total_seconds 626.9 > 600. Decomposition: 300 timed + 11.4 startup + ~118 evals ≈ 430; the remaining ~197s is loader-fetch stalls vs ~50s at baseline (total 480.8). Reflect padding's extra per-image CPU cost (PIL reflective copy vs cheap zero fill) tipped the already-borderline 8-worker pipeline further below the GPU's ~23k img/s demand. Stalls land OUTSIDE dt (epochs unaffected — 139!) but consume the 600s envelope. The EXP-004 "workers absorb PIL cost" precedent did NOT generalize: it predates compile (EXP-006), which raised GPU demand and thinned the worker margin. (source: run.log summary, baseline comparison exp-log-010)
- RESEARCH SIGNAL ALSO NEGATIVE: mid-schedule ran ~6–8pp BELOW baseline (ep 20: 79.8 vs ~88; ep 60: 85.4 vs ~92; ep 100: 92.1 vs ~96) at identical throughput — reflection padding behaves as STRONGER augmentation (more diverse, harder border crops without the easy black-band cue), not as free signal. Tail converged (96.46–96.57 plateau, final 96.52 ≈ best 96.57) at −0.14pp vs baseline — the saturated-regularization curve strikes again. Hypothesis refuted on both mechanism and outcome. (source: run.log eval lines)
- Epoch-1 eval 39.30% — actually ABOVE baseline lineage (~35%), consistent with cleaner early signal, before the added augmentation pressure dominated.

Key Metrics:
- best_test_acc: 96.57% @ ep 137 (baseline 96.71, bar 96.81; −0.14pp — but condition 1 fails first on the 626.9s cap bust) (source: run.log summary block)
- num_epochs: 139 | num_steps: 13418 | training_seconds: 300.0 | total_seconds: 626.9 | startup: 11.4s | peak_vram_mb: 1613.0 | num_params: 4,286,026

## Verification Results

### Conditions Checked

0. **Pre-condition: contention sanity** — 139 epochs vs ~139 projected; zero SLOW events; cum dt 22.4ms flat. CLEAN — the run is a valid measurement (the cap bust is a property of the change itself, not of interference). (source: run.log windows; detector bbr5zwtcm)
1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log` → **626.9s**. **FAIL** (goal hard rule: runs > 10 minutes total are failures; the overage is caused by the experiment's own loader-stall growth, reproducible, not infra noise). (source: run.log summary block)
2. **best_test_acc ≥ 96.81** — skipped per first-failure stop. (Observed anyway: 96.57 — would also have failed, −0.14pp vs baseline.)
3. **Validation at most once per epoch** — skipped per first-failure stop. (Informally compliant: 139 eval lines = 139 epochs.)

### Informational Metrics

- Not collected (condition 1 failed). Observed in summary: peak_vram_mb 1613.0 (identical), num_epochs 139 (identical — stalls don't touch the timed budget), num_params 4,286,026.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

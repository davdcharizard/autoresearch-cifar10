# EXP-050: Smaller batch size (128→64) for SGD gradient-noise regularization

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-050
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-050 Milestone 1: a single-line change to `train.py` — `BATCH_SIZE = 128` → `BATCH_SIZE = 64`. Nothing else changed (PEAK_LR 0.2, WARMUP_FRAC 0.05, schedule, optimizer, augmentation, seed all identical). Smoke test passed: AST clean, `git diff` shows exactly the one `BATCH_SIZE` line, 781 batches/epoch (50000//64, drop_last=True). Tests whether the increased relative SGD gradient noise (∝ LR/√B, doubled vs batch 128 at fixed LR) finds a flatter, better-generalizing minimum.

### Surprises & Discoveries
- None at implementation — trivial one-line config change.

### Decisions
- LR and warmup deliberately left UNCHANGED (not linear-scaled). The goal is MORE gradient noise at the SAME mean update step (the Keskar flat-minima test), which requires holding LR fixed; the linear-scaling rule would preserve dynamics and defeat the test. (Contrast EXP-025, which linear-scaled LR up for batch 256.)

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID at launch — background Bash)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (378.3s total wall)

Description:
- Running the baseline k=4 ResNet-20 recipe with BATCH_SIZE halved 128→64 (LR/warmup/all else unchanged) on idle GPU 1. Tests the only genuinely-untested axis left — batch-size downward — for the SGD gradient-noise → flat-minima generalization mechanism (Keskar 2017). EXP-025 showed the net is compute-bound, so batch-64 should ~halve dt (~4.5-6ms) and roughly double gradient updates at similar total images (~70-85 epochs); epoch-saturation bounds underfit risk. Expect best_test_acc in ~95.9-96.3; clears the bar only if small-batch noise genuinely helps.

Observations:
- **dt only partially halved (5-6ms, mostly 6ms) — a launch-overhead floor**, NOT the full ~4ms the compute-bound premise predicted. img/s dropped ~26% (≈11,500 vs baseline ~15,600), so total IMAGES fell → only **66 epochs** (vs baseline 91). Updates rose to 51,121 (~1.4× baseline ~35.5k, NOT the hoped 2×). (source: run.log, dt extraction)
- **Slower early convergence from the 2× relative gradient noise at fixed LR 0.2**: ep1 28.20% (vs baseline ~45.7%). Stable, no NaN/divergence — warmup + BN + LS held it. (source: run.log eval ep1)
- **REGRESSED −0.86pp**: best_test_acc 95.36 < baseline 96.22. final_test_loss 0.2138 > baseline 0.195 (WORSE on loss too). The test_acc was still monotonically CLIMBING at the final epoch (ep62 95.32 → ep66 95.36) — the model is UNDER-RESOLVED (under-trained), not overfit and not benefiting from flatter minima. (source: run.log summary + last evals)
- Clean completion: training_seconds 300.0, total_seconds 378.3 < 600, peak_vram 342.5 MB (lower, as expected for the smaller batch), no NaN/traceback. (source: run.log summary)

Key Metrics:
- best_test_acc: 95.36% (−0.86pp vs baseline 96.22; −0.96pp vs bar 96.32) @ ep66 (= final); final_test_acc 95.36%; final_test_loss 0.2138
- num_epochs: 66; num_steps: 51,121; training_seconds: 300.0; total_seconds: 378.3; peak_vram_mb: 342.5; num_params: 4,299,866 (unchanged)
- dt: 5-6ms (274×5ms, 740×6ms, 6×7ms) — partial halving (launch floor), vs baseline 8ms

## Verification Results

### Conditions Checked
1. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 95.36 < 96.32 (−0.96pp vs bar, −0.86pp vs baseline 96.22). → no-improvement. (source: run.log `best_test_acc: 95.36%`)
2. **Run completes cleanly within budget** — PASS. Summary printed; total_seconds 378.3 < 600; training_seconds 300.0; num_params 4,299,866 (unchanged); no crash/NaN. (source: run.log summary)
3. **No hard-constraint violations** — PASS. `git diff` = `train.py`, 1 line changed (`BATCH_SIZE`); prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; no seed hacking. (source: git diff --stat)

Verdict: **no-improvement** (primary condition fails, −0.86pp; clean valid run). Run completed cleanly → Outcome = completed.

### Informational Metrics
- num_epochs / num_steps: 66 / 51,121 (fewer epochs than baseline 91 due to the ~26% img/s drop; ~1.4× the updates).
- peak_vram_mb: 342.5 (lower than baseline ~454, as expected for batch 64).
- final_test_loss: 0.2138 (WORSE than baseline 0.195) — under-resolved/under-trained, not a flat-minima generalization gain.
- dt: 5-6ms (launch-overhead floor; did not reach the ~4ms full-compute-scaling prediction).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

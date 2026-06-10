# EXP-021: Larger Cutout (CUTOUT_SIZE 16 → 20)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean; verification Cond 1 failed → no-improvement verdict, rendered in analyze)

## Implementation Notes

### Summary
Single-constant change on the clean EXP-012 baseline `train.py`: line 28 `CUTOUT_SIZE = 16` → `CUTOUT_SIZE = 20` (Milestone 1). Everything else identical — k=4 WideResNet (4.3M params), PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov SGD, WD 1e-4, label smoothing 0.1, TrivialAugmentWide + vectorized GPU Cutout, torch.compile(reduce-overhead), bf16, channels_last, seed 42. `uv run ruff check train.py` passed, AST parses, `git diff --name-only` = train.py only, the only changed line is CUTOUT_SIZE.

### Surprises & Discoveries
None — the `cutout_batch` mask-fill clips the hole window to the image border, so a larger size needs no other code change (no index errors). The change is compute- and param-neutral (Cutout is a GPU masked_fill applied per batch).

### Decisions
Chose 20px (≈39% area) over a safer 18px because the brainstorm/plan favored a decisive up-probe: EXP-013 (Cutout 16→8) under-regularized (loss rose), so the indicated direction is more occlusion; 20px makes the test decisive while still failing gracefully (graceful no-improvement if it over-occludes).

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Runs the EXP-012 recipe with Cutout hole size raised 16→20px to test whether stronger occlusion regularization reduces the residual generalization gap and lifts best_test_acc above the 96.32 bar. The model is generalization-bound at fixed k=4 capacity in 300s, and augmentation is the only recently-productive mechanism. Expected: either a small top-1 gain (loss DOWN + acc UP = larger Cutout helps) or graceful no-improvement (loss UP = 20px over-occludes, Cutout optimum ≤16, axis closes).

Observations:
- Clean run: params 4,299,866 (unchanged), clean compile, no traceback, no NaN; throughput-neutral at 8ms/step ~15.6k img/s, 92 epochs — confirming the Cutout size change adds no compute cost (source: run.log L2, tail summary block).
- best_test_acc 95.96% vs baseline 96.22 = **−0.26pp**, below the 96.32 bar. final_test_loss ROSE 0.195→0.1969 — the KEY diagnostic predicted "loss UP = over-occlusion (axis closed at ≤16)". 20px (≈39% area) removes too much signal (source: run.log tail summary).

Key Metrics:
- best_test_acc: 95.96% (source: run.log `best_test_acc:` line)
- final_test_loss: 0.1969 (source: run.log `final_test_loss:` line) — vs baseline 0.195, ROSE
- final_test_acc: 95.75%; num_epochs: 92; num_steps: 35673; total_seconds: 408.1; peak_vram_mb: 453.8; num_params: 4,299,866 (source: run.log tail summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 95.96% < 96.32 (and below baseline 96.22, −0.26pp). (source: run.log `best_test_acc: 95.96%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=408.1 < 600, Traceback count=0, metrics present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866, eval-count=92 == num_epochs=92, no new deps, seed 42 intact.)

Verdict basis: as soon as one necessary condition fails, the experiment is no-improvement; remaining conditions not evaluated.

### Informational Metrics

- Not collected (only gathered when all necessary conditions pass). For the record: peak_vram_mb=453.8, num_epochs=92, num_steps=35673, final_test_loss=0.1969 (ROSE vs baseline 0.195 → over-occlusion at 20px).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)

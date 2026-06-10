# EXP-011: Mixup (mild α=0.2) GPU-vectorized, stacked on Cutout + compile enabler

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-011
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Edits to `train.py` only (Milestone 1): (1) added `MIXUP_ALPHA = 0.2`; (2) added per-batch Mixup in the training
loop after the Cutout line — sample scalar λ~Beta(0.2,0.2), `perm = randperm`, `inputs = λ·inputs + (1−λ)·inputs[perm]`,
`targets_b = targets[perm]`; (3) changed the loss to the Mixup convex combination of two label-smoothed CEs;
(4) added `compiled_model = torch.compile(model, mode="reduce-overhead")` and routed the training forward through
it; eval unchanged on eager `model`. Parse-clean, ruff clean, diff train.py-only, seed 42. Param count to be
confirmed at runtime (expected 4,299,866, UNCHANGED — Mixup is parameter-free).

### Surprises & Discoveries
- (none at implementation time — standard per-batch Mixup on the validated compile pattern.)

### Decisions
- **Mixup mixing + loss in the EAGER loop, only the forward compiled**: this keeps the per-step varying λ entirely
  out of the compiled graph, so there's no CUDA-graph/recompile risk from λ changing each step.
- **Per-batch scalar λ** (not per-sample): the standard Mixup formulation; simple, safe, and lets the loss stay a
  scalar convex combo. **Mild α=0.2** (U-shaped Beta) to regularize without drastically slowing convergence.
- **Compile included**: Mixup slows convergence; compile buys ~89 epochs. EXP-007 showed compiled-k4 ≈ baseline
  (null), so any gain is attributable to Mixup.

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
- Running the EXP-003 k=4 + Cutout recipe with mild Mixup (α=0.2) stacked on Cutout, compiled (reduce-overhead).
  Tests whether a complementary augmentation (interpolation vs Cutout's occlusion) improves generalization at fixed
  capacity. Expect params 4,299,866 (unchanged), clean compile, ~85–89 epochs, clean run < 600s. NOTE: train loss
  reads higher/noisier under Mixup (mixed targets) — judge health by eval acc, not train loss. Success = ≥ 96.10.

Observations:
- Clean startup; **params 4,299,866 (UNCHANGED)** — confirms Mixup is parameter-free (run.log head). Clean compile,
  no traceback, no NaN.
- **Steady-state dt = 9ms/step (~14,877 img/s)** from step 50 ≈ compiled-k4 (EXP-007) — Mixup's per-step cost
  (Beta scalar + permute + lerp + 1 extra CE) is negligible. Tracking toward ~85–89 epochs ⇒ fair converged test.
- Loss 2.41 @ step 50 — slightly higher/noisier than a non-Mixup run (expected from mixed targets), no NaN/divergence.

Key Metrics:
- best_test_acc: **95.86%** @ ep 87 — BELOW baseline 96.00 (−0.14pp) and bar 96.10 (run.log summary).
- **num_epochs: 88** / num_steps 33,979 — fair, fully-converged test (eval count 88 = num_epochs ⇒ eval once/epoch);
  NOT epoch-starved, so mild Mixup got a fair shot.
- final_test_acc 95.75; **final_test_loss 0.2898** — markedly HIGHER than baseline 0.204 / compiled-k4 0.208. This
  is the expected Mixup artifact: training on interpolated/soft targets yields less-confident predictions → higher
  CE on hard one-hot test labels, NOT worse classification (accuracy is ~flat). Late evals 95.69–95.86 (ep 86–88).
- num_params 4,299,866 (UNCHANGED — Mixup parameter-free). peak_vram 453.7 MB. dt 9ms (~14,877 img/s).
- SiLU-free Mixup-k4 (95.86) ≈ compiled-k4 (95.92, EXP-007) within the ~0.2pp noise band → **mild Mixup added no
  accuracy**. The augmentation axis did not break the plateau at α=0.2.

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 95.86% present, total_seconds 398.3 < 600, no
  traceback (run.log; tracebacks=0).
- **Cond 2 — metric ≥ 96.10**: **FAIL**. 95.86 < 96.10 (also < 96.00 baseline, −0.14pp). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2. (Informational: clean — diff = only
  train.py, num_params unchanged 4,299,866, seed 42.)

### Informational Metrics

- num_epochs 88 / num_steps 33,979 (fair converged test, not epoch-starved). final_test_loss 0.2898 (raised by
  Mixup's soft-target effect, not a classification regression — acc ~flat). peak_vram_mb 453.7. img/s ~14,877.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

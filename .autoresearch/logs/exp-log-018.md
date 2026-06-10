# EXP-018: CutMix (regional label-mixing aug), GPU-vectorized per batch, on the TA+Cutout recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented CutMix per plan-018 (3 edits, all train.py): (1) added `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`
constants; (2) added a `cutmix_batch(x)` helper after `cutout_batch` — samples lam~U(0,1) (Beta(1,1)),
permutes the batch, pastes a random area-(1−lam) box from `x[perm]`, returns `(x, perm, lam_area_corrected)`;
(3) in the training loop, after the Cutout line, a per-batch coin flip `do_cutmix = rand() < CUTMIX_PROB`
applies CutMix, and the loss becomes `lam·CE(out,y) + (1−lam)·CE(out,y[perm])` on CutMix batches (plain CE
otherwise). Cutout and TrivialAugment are kept (validated recipe). Ruff clean; diff is train.py only (+36/−3).

### Surprises & Discoveries
None during implementation. α=1.0 conveniently makes the CutMix cut fraction Uniform(0,1), so no Beta sampler
(and no new dep) is needed — `torch.rand(1)` suffices.

### Decisions
- **p=0.5, α=1.0** (standard CutMix). Half the batches keep the plain TA+Cutout recipe, hedging against
  over-regularization/underfit within the short ~84–91-epoch budget (CutMix normally wants 200–300 ep).
- **CutMix after Cutout, both kept**: the cleanest reading of "add CutMix to the recipe" — isolates the addition
  of CutMix rather than trading it against Cutout exposure (EXP-013 showed reducing Cutout hurts).
- **Branch in Python via `.item()`**: the loop already does `loss.item()` + `cuda.synchronize()` every step, so
  the CutMix coin-flip / box-coord `.item()` reads add no meaningful sync cost; keeps the code simple and readable.
- Model input SHAPE is unchanged every step → `torch.compile(reduce-overhead)` CUDA graphs are unaffected.

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
- Running the EXP-012 baseline recipe + CutMix (p=0.5, α=1.0) on a single H20. CutMix is a cheap per-batch GPU
  op → expect ~8ms/step and ~84–91 epochs (throughput-neutral, fair same-budget test). Hypothesis: the regional
  label-mixing regularizer reduces the residual generalization gap and lifts best_test_acc above the 96.32 bar.
  Main risk: label-mixing augs want long schedules, so it may underfit within the budget → null/slight regression.
  Train loss will read HIGHER than baseline (soft targets) — expected, judge on test acc not loss.

Observations:
- Clean startup: `params: 4,299,866` (UNCHANGED — aug-only change), clean compile, no traceback, no NaN (run.log L1-4).
- CutMix code path ran correctly: ep 1 test_acc 33.64% (low, as expected with soft-target mixing); dt ~8–14ms.
- Ran only 71 epochs / 27,535 steps — low end of the throughput-jitter band (CutMix's extra ops + run jitter).
- Test loss high throughout (0.25–0.30 range) — the expected soft-target artifact, NOT a divergence.
- Run exited 0, total_seconds 394.9 < 600 (source: run.log final summary, background task exit 0).

Key Metrics:
- best_test_acc: 95.14% (source: run.log) — vs baseline 96.22 (**−1.08pp**, largest aug regression)
- final_test_acc: 95.03% | final_test_loss: 0.2774 (soft-target artifact — judged on acc, not loss)
- num_epochs: 71 | num_steps: 27,535 | num_params: 4,299,866 | peak_vram_mb: 453.8 | total_seconds: 394.9

## Verification Results

### Conditions Checked
- **Cond 1 — clean completion within budget**: PASS. best_test_acc and total_seconds present; total_seconds
  394.9 < 600; Traceback count 0 (source: run.log final summary).
- **Cond 2 — primary metric clears bar**: **FAIL**. best_test_acc = 95.14% < 96.32 bar. Δ = −1.08pp vs
  baseline 96.22. → verdict no-improvement. (Decisive condition.)
- **Cond 3 — no constraint violations**: skipped — not reached after Cond 2 failed. (For the record: scope clean —
  git diff = train.py only; eval-count 71 == num_epochs 71 (CutMix is train-only, eval path untouched);
  num_params 4,299,866 unchanged; seed 42 intact; no new deps — torch/math only.)

### Informational Metrics
- Not collected (only when all necessary conditions pass). For reference: num_epochs 71, final_test_loss 0.2774
  (soft-target artifact — high by design), peak_vram_mb 453.8 (unchanged).

## Errors & Dead Ends

## Human Notes

> (none)

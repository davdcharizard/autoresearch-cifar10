# Experiment Log EXP-010
- **Created**: 2026-05-27
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md

## Execution
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-010
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented CutMix batch augmentation in train.py per plan-010.md Milestone 1. Four edits applied: (1) added `import numpy as np` at top of file, (2) added `CUTMIX_ALPHA = 1.0` hyperparameter constant after MAX_STEPS, (3) added `rand_bbox(size, lam)` helper function before `main()` that computes a random bounding box given image dimensions and λ, (4) inserted CutMix logic in the training loop between GPU transfer and optimizer.zero_grad() — draws λ from Beta(α,α), shuffles indices, computes bbox, blends images, adjusts λ for actual pixel ratio, stores target pairs — and replaced the single cross-entropy loss with mixed-label loss `lam * CE(outputs, targets_a) + (1-lam) * CE(outputs, targets_b)`. All 6 Milestone 1 tasks completed. Syntax check passed (`uv run python -c "import train"`).

### Surprises & Discoveries
No surprises. The existing code structure cleanly accommodated CutMix insertion — the training loop had a clear separation between data loading/transfer and the forward pass. numpy was already a transitive dependency via torchvision but not directly imported.

### Decisions
No deviations from plan. All implementation choices followed the plan exactly: α=1.0, in-place image blending, adjusted λ for boundary clipping, mixed-label cross-entropy loss.

## Experimental Adjustments
(none)

## Run Log

### Run 1
- **Description**: Full CutMix training run. Running `uv run train.py > run.log 2>&1` locally on H20 GPU. Expecting ~98 epochs in 300s with negligible throughput overhead from CutMix (pure tensor ops). Target: best_test_acc > 95.49%.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27
- **Observations**: Training completed 96 epochs / 18591 steps in 300.0s. CutMix α=1.0 stacked on TrivialAugmentWide+RandomErasing over-regularized the model — best_test_acc peaked at 95.03% (epoch 93), 0.36pp below baseline 95.39%. The model was still slowly improving at end of budget (final_test_acc 94.91% at epoch 96), suggesting CutMix needs more training time to converge when combined with heavy per-sample augmentation. LR schedule operated normally: 0.2→0.02 at ~50%, 0.02→0.002 at ~75%. No NaN/inf, no OOM.
- **Key Metrics**:
  - best_test_acc: 95.03%
  - final_test_acc: 94.91%
  - final_test_loss: 0.2819
  - training_seconds: 300.0
  - total_seconds: 411.6
  - startup_seconds: 1.2
  - peak_vram_mb: 864.6
  - num_epochs: 96
  - num_steps: 18591
  - num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 95.49%** — **FAIL**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Actual: 95.03
- Required: > 95.49
- Result: FAIL (0.46pp below threshold, 0.36pp below baseline 95.39%)

**Condition 2: Summary block complete (10 fields)** — **PASS**
- Command: `grep -c "^best_test_acc:\|..." run.log`
- Actual: 10
- Required: 10
- Result: PASS

**Condition 3: Eval count <= num_epochs** — **PASS**
- eval_count: 96, num_epochs: 96
- Result: PASS (96 <= 96)

## Errors & Dead Ends
(none)

## Human Notes
(none — autopilot)

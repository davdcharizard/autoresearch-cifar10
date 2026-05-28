# EXP-000: Training Recipe Modernization

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Autonomy**: copilot
- **Experiment Branch**: autoresearch/exp-000
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented all 5 planned changes to `train.py` on the ResNet-20 baseline: (1) replaced `MultiStepLR` with epoch-based `CosineAnnealingLR(T_max=200)` and moved `scheduler.step()` from the per-batch loop to after each epoch's eval, (2) added a `Cutout` transform class (n_holes=1, length=16) applied after normalization in the training pipeline, (3) added `label_smoothing=0.1` to the `F.cross_entropy` call, (4) enabled `nesterov=True` on the SGD optimizer, (5) removed the `MAX_STEPS = 64000` constant and all its references so the time budget is the sole stopping criterion. Also added `import numpy as np` for the Cutout random coordinate generation. All changes pass `ruff` linting.

### Surprises & Discoveries

No surprises — the codebase was clean and the changes mapped directly to the plan. The `scheduler.step()` was previously called per-batch (step-based MultiStepLR), so switching to epoch-based required moving the call from inside the batch loop to after the eval block. This was anticipated in the plan.

### Decisions

None — all changes followed the plan exactly.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the modified ResNet-20 with cosine annealing LR, Cutout (16x16), label smoothing (0.1), and Nesterov momentum on a single H20 GPU. The time budget is 300s. We expect the LR schedule fix (cosine vs. broken MultiStepLR) and Cutout regularization to push best_test_acc from 91.72% baseline to approximately 93-94%. Removing MAX_STEPS allows the model to train for more steps within the time budget.

Observations:
- Training completed 91 epochs in 300s (vs 97 in baseline — fewer epochs due to slightly more overhead from Cutout). (source: run.log summary block)
- LR at epoch 90 was still 0.0586 — cosine annealing with T_max=200 barely decayed the LR over the actual 91-epoch training. The model spent the entire training at LR > 0.05. (source: run.log L-20)
- Loss plateaued around 0.79-0.81 from epoch ~60 onward, never dropping below 0.78. This is much higher than the baseline's converged loss, indicating insufficient LR decay. (source: run.log)
- The baseline MultiStepLR dropped LR from 0.1 to 0.01 at step 32000 (~epoch 82), which was critical for final convergence. The cosine schedule's gradual decay never reached that low. (source: baseline analysis in brainstorm-000.md)

Key Metrics:
- best_test_acc: 88.79% @ epoch 87 (source: run.log summary block) — 2.93pp BELOW baseline
- final_test_acc: 88.46% @ epoch 91 (source: run.log summary block)
- final_test_loss: 0.3972 (source: run.log summary block)
- training_seconds: 300.0 (source: run.log summary block)
- peak_vram_mb: 330.1 (source: run.log summary block)
- num_epochs: 91 (source: run.log summary block)
- num_steps: 35215 (source: run.log summary block)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 91.82% (baseline 91.72% + 0.1% delta)**: **FAIL** — best_test_acc = 88.79%, which is 2.93pp below baseline. (source: `grep "^best_test_acc:" run.log` → `88.79%`)
2. Script completes without crash: skipped — aborted after prior failure
3. Validation runs at most once per epoch: skipped — aborted after prior failure

### Informational Metrics

Not collected — necessary condition 1 failed.

## Errors & Dead Ends

### 2026-05-27 — Cosine annealing T_max=200 too large for 91-epoch training
- Error: best_test_acc 88.79% — 2.93pp below 91.72% baseline
- Root cause: CosineAnnealingLR(T_max=200) with only 91 epochs completed means LR decayed from 0.1 to only ~0.058 by end of training. The baseline's MultiStepLR dropped LR to 0.01 at step 32000 (~epoch 82), providing the critical low-LR convergence phase. The cosine schedule's gradual decay never reached comparably low LR. Additionally, Cutout and label smoothing increase training difficulty, requiring even more aggressive LR decay for convergence.
- Source: run.log summary block and step-level LR values
- Do NOT retry: cosine annealing with T_max significantly larger than actual epoch count. T_max must match or be close to the actual number of epochs that will complete in the time budget.

## Human Notes

> {Researcher can add comments, corrections, or context here}

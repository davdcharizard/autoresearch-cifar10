# EXP-022: Reflect Padding + Cutout Replacing RandomErasing

## Execution

Overall Status & Info:
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-022
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the two planned augmentation changes to train.py: (1) Added a custom `Cutout` class after the hyperparameters section — 12x12 fixed zero-fill square at random position with p=0.5, operating on normalized tensors. (2) Changed `RandomCrop(32, padding=4)` to `RandomCrop(32, padding=4, padding_mode='reflect')`. (3) Replaced `RandomErasing(p=0.25, scale=(0.02, 0.2))` with `Cutout(size=12, p=0.5)` in the augmentation pipeline. Import check passed via `uv run python -c "import train"`.

### Surprises & Discoveries

None — the changes were straightforward augmentation pipeline swaps with no unexpected code interactions.

### Decisions

Placed Cutout after `transforms.Normalize` (operates on normalized tensors) as specified in the plan, matching airbench96's ordering. This replaces RandomErasing which was also positioned after Normalize.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local background process
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- Running EXP-022 training with reflect padding + Cutout replacing RandomErasing. The model is ResNet-20 WIDTH_MULT=4 with AMP, batch 256, cosine warmup+decay schedule, TTA evaluation. Expected ~99 epochs in 300s budget with zero throughput cost from the augmentation changes. Target: best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold).

Observations:
- Training completed normally, 99 epochs in 300.0s budget (~15-16ms/step)
- Zero throughput cost from augmentation changes as expected (same 99 epochs as baseline)
- Best accuracy reached at epoch 95 (96.53%), then slight decline through epoch 99 (96.33%)
- Epoch progression in final epochs: ep93 96.51%, ep94 96.44%, ep95 96.53% (best), ep96 96.42%, ep97 96.39%, ep98 96.35%, ep99 96.33%
- Result is +0.07pp over baseline (96.46%) but below the +0.1pp improvement threshold

Key Metrics:
- best_test_acc: 96.53%
- final_test_acc: 96.33%
- final_test_loss: 0.2868
- training_seconds: 300.0
- total_seconds: 430.0
- startup_seconds: 1.2
- peak_vram_mb: 864.6
- num_epochs: 99
- num_steps: 19179
- num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: Primary metric exceeds threshold** — FAILED
- Command: `grep 'best_test_acc:' run.log | awk '{print $2}' | tr -d '%'`
- Result: 96.53
- Threshold: > 96.56 (baseline 96.46 + 0.1pp)
- 96.53 ≤ 96.56 → FAILED

**Condition 2: Training script completes and prints full summary block** — PASSED
- Command: `grep -c 'best_test_acc:\|final_test_acc:\|...' run.log`
- Result: 10 (all 10 summary fields present)
- 10 = 10 → PASSED

**Condition 3: Validation runs at most once per epoch** — PASSED
- EVALS=99, EPOCHS=99
- 99 ≤ 99 → PASSED

### Informational Metrics

- num_epochs: 99 (matches baseline, zero throughput cost confirmed)
- peak_vram_mb: 864.6 (unchanged from baseline)
- final_test_acc: 96.33% (last-epoch accuracy)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

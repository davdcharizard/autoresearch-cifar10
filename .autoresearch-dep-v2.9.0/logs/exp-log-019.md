# EXP-019: Test-Time Augmentation (Horizontal Flip)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-019
- **Commit**: 355391a
- **PR**: (failed — GraphQL: Resource not accessible by personal access token)
- **Outcome**: completed

## Implementation Notes

### Summary

Added a `tta_evaluate(model, device, loader)` function to train.py that performs horizontal flip TTA. For each test batch, the function computes logits on original images and on horizontally-flipped images (`torch.flip(inputs, dims=[3])`), averages the two logit tensors, then computes cross-entropy loss (sum reduction, divided by total count) and accuracy from the averaged logits. The function uses `@torch.inference_mode()`, `model.eval()`, and AMP autocast to match training precision. Replaced the per-epoch eval call at the original line 256 (`evaluator.evaluate(model, device)`) with `tta_evaluate(model, device, evaluator.loader)`.

### Surprises & Discoveries

None — implementation was straightforward as planned. The `evaluator.loader` attribute is public and accessible.

### Decisions

- Used AMP autocast (`torch.amp.autocast("cuda", dtype=torch.float16)`) inside the TTA function to match the training precision and ensure consistent behavior with the AMP-trained model. The plan did not explicitly specify this but it aligns with how the existing evaluator works under AMP.
- Used `memory_format=torch.channels_last` when moving inputs to device, matching the training loop's data format for optimal performance on H20.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/.autoresearch/logs/exp-019-run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the full training script with TTA evaluation replacing the standard single-crop evaluation. Training is identical to EXP-015 baseline (WIDTH_MULT=4, batch 256, AMP, label_smoothing=0.2, TrivialAugmentWide+RandomErasing). Only evaluation is modified to average logits from original and horizontally-flipped test images. Expecting ~92-98 epochs in 300s, with TTA adding ~10-20s eval overhead. Target: best_test_acc > 95.67%.

Observations:
- Training completed 98 epochs in 300.0s (matching baseline epoch count exactly)
- TTA evaluation added ~117s overhead (total_seconds 418.0 vs ~301s for baseline), averaging ~1.2s per TTA eval pass
- Best accuracy 95.91% achieved at epoch 92; final epoch accuracy 95.70%
- Training progression healthy: loss converged smoothly, LR schedule drops visible at ~50% and ~75% wall-clock
- Peak VRAM 864.6 MB — negligible increase from TTA's doubled inference passes (batch fits comfortably)
- All 10 summary fields printed correctly in final block

Key Metrics:
- best_test_acc: 95.91%
- final_test_acc: 95.70%
- final_test_loss: 0.2970
- training_seconds: 300.0
- total_seconds: 418.0
- startup_seconds: 1.3
- peak_vram_mb: 864.6
- num_epochs: 98
- num_steps: 19,069
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **Condition 1 — best_test_acc > 95.67%**: PASS. `best_test_acc: 95.91%` (95.91 > 95.67). Source: exp-019-run.log summary block.

2. **Condition 2 — Full summary block**: PASS. All 10 required fields present: best_test_acc, final_test_acc, final_test_loss, training_seconds, total_seconds, startup_seconds, peak_vram_mb, num_epochs, num_steps, num_params. Source: exp-019-run.log final summary.

3. **Condition 3 — Eval count ≤ num_epochs**: PASS. 98 eval lines ("eval ep") found, num_epochs = 98, 98 ≤ 98. Source: exp-019-run.log.

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 864.6
- final_test_acc: 95.70%
- final_test_loss: 0.2970
- num_epochs: 98
- num_steps: 19,069
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> 

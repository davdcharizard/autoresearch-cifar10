# Report EXP-019: Test-Time Augmentation (Horizontal Flip)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Log**: logs/exp-log-019.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 95.57% (EXP-015, label smoothing 0.2 on WIDTH_MULT=4 ResNet-20). Improvement threshold: baseline + 0.1pp = 95.67%.

## Idea & Hypothesis

Replace single-crop evaluation with horizontal-flip TTA: for each test batch, run the model on both original and horizontally-flipped images, average logits, and take argmax. The training loop is completely unchanged — TTA operates only at evaluation time. Since evaluation time does not count against the 300s training budget, this is a zero-cost intervention.

Hypothesis: Averaging logits from original and flipped test images will improve best_test_acc by +0.1-0.3pp because the model was trained with RandomHorizontalFlip and has learned flip-equivariant features whose prediction variance is reduced by averaging over both orientations.

## Approach

Added a `tta_evaluate(model, device, loader)` function to train.py (~17 lines) that iterates the test DataLoader, computes logits on original images and on `torch.flip(inputs, dims=[3])` flipped images under AMP autocast (FP16), averages the two logit tensors, then computes cross-entropy loss (sum reduction / total count) and accuracy. Uses `@torch.inference_mode()` and `model.eval()`. Replaced the per-epoch eval call `evaluator.evaluate(model, device)` with `tta_evaluate(model, device, evaluator.loader)`.

Implementation decisions beyond the plan: used AMP autocast (`torch.amp.autocast("cuda", dtype=torch.float16)`) inside TTA to match training precision, and `memory_format=torch.channels_last` when moving inputs to device to match the training loop's format.

## Execution

Single run, direct execution via `uv run python train.py`. Training completed 98 epochs in 300.0s (identical epoch count to baseline). TTA added ~117s evaluation overhead (total_seconds 418.0 vs ~301s for baseline), averaging ~1.2s per TTA eval pass. No errors, no retries, no adjustments.

## Results

- **Primary metric**: best_test_acc = 95.91% (baseline: 95.57%, delta: +0.34pp, +0.36%)
- **Observations**: TTA overhead was larger than estimated (~117s vs predicted 10-20s), but this is evaluation-only time outside the training budget so it has no impact on the result. Training progression was healthy and identical to baseline — same 98 epochs, same LR schedule shape, same convergence pattern. Best accuracy achieved at epoch 92; final epoch accuracy 95.70%.
- **Analysis**: The +0.34pp improvement exceeded the upper end of the predicted +0.1-0.3pp range, confirming that the model has strong flip-equivariant features whose prediction variance was meaningfully reduced by logit averaging. The gap between best (95.91%) and final (95.70%) accuracy is 0.21pp, comparable to baseline behavior — TTA did not change the variance of epoch-to-epoch accuracy, it shifted the entire curve upward.
- **Key Learning**: TTA is orthogonal to all training-side changes and provides a reliable accuracy boost through variance reduction over flip-equivariant features — it should be treated as a permanent fixture, not an experiment to iterate on.

## Verification

- **Conditions**: All 3 passed. (1) best_test_acc 95.91% > 95.67% threshold. (2) All 10 summary fields present. (3) 98 eval lines = 98 epochs.
- **Review Notes**: Results confirmed trustworthy. Training dynamics identical to baseline (same epoch count, same convergence shape), and improvement came through the intended mechanism (evaluation-time logit averaging, not training modification).
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed and primary metric improved by +0.34pp (above 0.1pp threshold).

## Unexplored Avenues

- **Multi-crop TTA (4-corner + center crops + flips)**: The original AlexNet paper used 10-crop TTA. More crops would further reduce prediction variance but at multiplicative evaluation cost. Diminishing returns likely given horizontal flip already captures the dominant equivariance.
- **Learned TTA policies (Shanmugam et al. 2021)**: Optimizing TTA augmentation weights per-class could extract more signal, but requires a validation split and additional training — complexity vs marginal gain tradeoff unclear.
- **Softmax averaging vs logit averaging**: We averaged logits (pre-softmax); averaging softmax probabilities is an alternative that gives different weighting to high-confidence predictions. Literature suggests logit averaging is generally superior for small augmentation sets.

## Next Steps

1. **OneCycleLR with tuned max LR** (medium confidence): Replace MultiStepLR with OneCycleLR to explore superconvergence. The well-validated (0.5, 0.75) schedule may be a local optimum — triangular-then-anneal could find wider minima. Risk: replaces a High Importance validated pattern.
2. **Deeper architecture (NUM_BLOCKS=5, ResNet-32)** (medium confidence): More capacity via depth rather than width. Tradeoff: ~60 epochs vs ~98, which may be insufficient. The regularization stack is calibrated for ~98 epochs.
3. **AdamW or LAMB optimizer** (low-medium confidence): Different optimizer could navigate the loss landscape more efficiently at this accuracy level. SGD+momentum may be leaving accuracy on the table in the final LR phase.

## Exit Action Results

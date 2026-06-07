# EXP-017: SE channel attention in BasicBlock

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added SEBlock module to train.py — a lightweight channel attention mechanism with global average pooling → FC reduce (r=16, min 4) → ReLU → FC expand → Sigmoid → element-wise scale. Integrated into BasicBlock by adding `self.se = SEBlock(out_channels)` in `__init__` and `out = self.se(out)` in `forward()` after the second conv+BN and before the residual addition. No hyperparameter changes. The SE module uses `bias=False` on both FC layers following common practice.

### Surprises & Discoveries

Used `max(channels // reduction, 4)` for the bottleneck dimension to avoid degeneracy at the first layer (64 // 16 = 4, which is already minimal). This ensures the SE bottleneck has enough capacity even for the narrowest layer.

### Decisions

Used `bias=False` on both FC layers in SEBlock — common in SE implementations and consistent with the rest of the model (all conv layers use bias=False). Used `x.mean(dim=(2, 3))` instead of `F.adaptive_avg_pool2d(x, 1)` for the squeeze operation — functionally identical but avoids an extra reshape.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-05-29
- **Ended**: 2026-05-29

Description:
- Running ResNet-20 k=4 with SE blocks (reduction=16) on CIFAR-10. Training with SGD+Nesterov, EMA(0.999), CutMix, label smoothing, warmup+cosine LR, AMP+torch.compile, and TTA (hflip). 4,360,010 params. Target: best_test_acc >= 96.49%.

Observations:
- Severe training instability: model stuck at ~10-21% for first 13 epochs, loss diverged to 38.3 at epoch 9 (source: run.log eval lines)
- Model eventually recovered but only reached 93.14% by epoch 47 — well below 96.39% baseline
- Root cause: ResNet's `_weights_init` applied `kaiming_normal_` to SE block FC layers, causing random SE gate values that severely distorted features early in training

Key Metrics:
- best_test_acc: 93.14% @ epoch 45 (source: run.log)
- training_seconds: 300.0s (source: run.log)
- num_epochs: 47 (source: run.log)

### Run 2

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-29
- **Ended**: 2026-05-29

Description:
- Re-running with fixed SE initialization: after model's `_weights_init`, zero-initialize FC2 weights in all SEBlock modules so sigmoid output starts at 0.5 (uniform mild scaling). This ensures SE blocks start near-identity and gradually learn channel recalibration.

Observations:
- Training converged normally (no divergence), confirming the init fix resolved the instability
- 50 epochs completed (vs ~54 baseline) — SE overhead costs ~4 epochs
- 94.59% is significantly below 96.39% baseline (and even below 95.73% pre-TTA baseline)
- SE blocks are not beneficial for this shallow 9-block architecture with limited training budget

Key Metrics:
- best_test_acc: 94.59% @ epoch 50 (source: run.log)
- final_test_acc: 94.59% (source: run.log)
- training_seconds: 300.0s (source: run.log)
- num_epochs: 50 (source: run.log)
- peak_vram_mb: 551.8 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 94.59%, 1.80% below threshold (baseline 96.39% + 0.1%). (source: run.log)
2. **Training within 300s budget**: skipped — aborted after prior failure
3. **Eval called at most once per epoch**: skipped — aborted after prior failure

### Informational Metrics

<!-- Not collected — necessary condition failed. -->

## Experimental Adjustments

- **Zero-initialize SE FC2 weights**: Kaiming normal init on SE FC layers caused severe training instability (loss diverged for 13 epochs). Fixed by adding `init.zeros_(m.fc2.weight)` for all SEBlock modules after model init, so SE gates start at sigmoid(0)=0.5. (ref: Run 1 — 93.14% vs 96.39% baseline due to init issue)

## Errors & Dead Ends

### 2026-05-29 — SE block init causes training instability
- Error: `Model stuck at 10-21% accuracy for 13 epochs, loss diverged to 38.3`
- Root cause: ResNet's `_weights_init` applies `kaiming_normal_` to SE block FC layers, causing random sigmoid gate values that severely distort features
- Source: run.log (Run 1 eval lines)
- Do NOT retry: Never use SE blocks without explicit zero-init on the gate (FC2) layer

## Human Notes

> {Researcher can add comments, corrections, or context here}

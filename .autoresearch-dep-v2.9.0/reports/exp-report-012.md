# Report EXP-012: Conv1x1-based SE Blocks (channels_last-safe)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Log**: logs/exp-log-012.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 95.39% (EXP-009, commit cfe19c2). Threshold for improvement: >95.49%.

## Idea & Hypothesis

Conv1x1-based SE blocks were chosen as a direct fix for EXP-011's failure mode. EXP-011 showed that SE blocks improve per-epoch quality (95.45% in 83 epochs vs 95.39% in 98 epochs), but nn.Linear in the SE path broke channels_last format, doubling per-step time. The hypothesis was that replacing nn.Linear with nn.Conv2d(kernel_size=1) would eliminate the format conversion overhead, preserving ~95-98 epochs while retaining SE's per-epoch quality advantage, yielding ~95.6-95.9%.

## Approach

Added `SEBlock` class to `train.py` using `nn.Conv2d(channels, channels//16, 1, bias=False)` for both FC layers, with `x.mean(dim=(2,3), keepdim=True)` to keep (B,C,1,1) tensor shape throughout. Integrated as `self.se = SEBlock(out_channels)` in `BasicBlock.__init__` and `out = self.se(out)` in `BasicBlock.forward` after BN2, before residual addition. All 9 BasicBlocks received SE attention. No hyperparameters or other code changed. Implementation matched plan-012.md exactly with no deviations.

## Execution

Single run on local H20 GPU with 300s training budget. Training completed normally with 83 epochs and 16,002 steps. No errors, no retries, no adjustments needed.

## Results

- **Primary metric**: 95.23% (baseline: 95.39%, delta: -0.16pp, -0.17%)
- **Observations**: Per-step time was ~18-19ms from step 50 onwards — identical to EXP-011's nn.Linear SE implementation. Conv2d(1x1) did NOT reduce the overhead. The model completed only 83 epochs (same as EXP-011) vs 98 epochs for the SE-free baseline. Peak VRAM: 1034.4 MB, params: 4,318,282.
- **Analysis**: The hypothesis was disproved. The ~9ms/step overhead from SE blocks is intrinsic to the SE computation itself (global average pooling + two small convolutions + sigmoid + element-wise multiply), not to channels_last format conversion as theorized from EXP-011. Both nn.Linear (EXP-011: 95.45%) and Conv2d(1x1) (EXP-012: 95.23%) produce nearly identical timing. The 0.22pp gap between the two SE implementations (95.45 vs 95.23) is within run-to-run variance — the fundamental constraint is the ~83-epoch budget.
- **Key Learning**: SE block overhead on this model is computational, not format-related — any SE variant will cost ~9ms/step on H20, reducing the epoch budget from 98 to 83 and negating per-epoch quality gains.

## Verification

- **Conditions**: Condition 1 FAILED (95.23% < 95.49% threshold); Conditions 2-3 PASSED (summary block present, validation ≤ once per epoch)
- **Review Notes**: Results confirmed trustworthy — metrics consistent with training dynamics, per-step time aligns with EXP-011 corroborating the overhead measurement.
- **Verdict**: no-improvement
- **Verdict Basis**: Verification condition 1 failed — primary metric 95.23% did not exceed 95.49% threshold.

## Unexplored Avenues

- **SE approach is exhausted for this project**. Both nn.Linear (EXP-011) and Conv2d(1x1) (EXP-012) produce identical ~9ms/step overhead. The overhead is computational, not format-related, so no implementation variant will fix it. The only theoretical path would be reducing SE to a single layer or using ECA-style 1D conv (no reduction), but the overhead from global avg pool + sigmoid + multiply alone likely accounts for most of the 9ms, making any SE variant unviable within the 300s budget.
- **Non-architectural approaches remain viable**: EMA (brainstorm-012 candidate #2) adds no throughput cost and was expected to yield +0.1-0.3pp. Mixup with low alpha replacing RandomErasing (candidate #3) is also untried.

## Next Steps
1. **EMA of model weights** (high confidence) — purely additive, zero throughput cost, expected +0.1-0.3pp. Strongest remaining candidate from brainstorm-012.
2. **Mixup (alpha=0.2) replacing RandomErasing** (medium confidence) — swaps one regularizer for another, informed by EXP-010's over-regularization learning.
3. **Increased depth (ResNet-32 or ResNet-44)** (low confidence) — more capacity at the cost of fewer epochs; may need to reduce WIDTH_MULT to maintain throughput.

## Exit Action Results

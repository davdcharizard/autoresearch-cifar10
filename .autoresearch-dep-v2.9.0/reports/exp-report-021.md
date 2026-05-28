# Report EXP-021: Pre-activation ResNet Blocks (BN-ReLU-Conv)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Log**: logs/exp-log-021.md

## Goal

Maximize best_test_acc (%) on CIFAR-10 with a ResNet-20 (WIDTH_MULT=4) trained in 300s on a single H20 GPU. Baseline: 96.46% (EXP-020). Direction: higher is better. Threshold: best_test_acc > 96.56% (baseline + 0.1pp).

## Idea & Hypothesis

Switched from post-activation (Conv-BN-ReLU) to pre-activation (BN-ReLU-Conv) block ordering per He et al. 2016 "Identity Mappings in Deep Residual Networks." Pre-activation blocks preserve clean identity mappings on the residual shortcut path, improving gradient flow — particularly beneficial during the cosine decay's final near-zero LR phase. Selected over SAM (2x compute cost too risky) and deeper architecture (moderate throughput cost). Hypothesis: +0.1-0.3pp with zero throughput cost, reaching 96.56-96.76%.

## Approach

Three changes to `train.py`: (1) BasicBlock.forward rewritten to `bn1(x) → relu → conv1 → bn2 → relu → conv2` with shortcut taken from raw `x` before BN/ReLU — no final relu after addition. bn1 was already `BatchNorm2d(in_channels)` from a prior edit so no `__init__` change was needed. (2) Removed `self.bn1 = nn.BatchNorm2d(16 * WIDTH_MULT)` from ResNet.__init__ since the first block's bn1 handles normalization of stem output. Added `self.bn_final = nn.BatchNorm2d(64 * WIDTH_MULT)` after layer3 definition. (3) ResNet.forward stem simplified to raw `self.conv1(x)` (no BN/ReLU), ending changed to `F.relu(self.bn_final(out)) → pool → fc`. No configuration changes — all hyperparameters identical to baseline.

## Execution

Single run, completed successfully. Training ran 93 epochs in 300.0s (vs 99 in baseline EXP-020). Total time including TTA evaluation: 408.3s. No errors, no retries, no adjustments needed. Implementation followed the plan exactly.

## Results

- **Primary metric**: 96.23% (baseline: 96.46%, delta: -0.23pp, -0.24%)
- **Observations**: Pre-activation blocks caused a ~6% throughput regression — 18,083 steps at avg 16.6ms/step vs 19,198 steps at 15.6ms/step in the baseline. This reduced epoch count from 99 to 93, losing 6 epochs of training. The throughput regression is likely caused by reduced cuDNN kernel fusion opportunities: cuDNN fuses Conv→BN→ReLU into a single kernel, but cannot fuse BN→ReLU→Conv. Training loss converged normally with no instability.
- **Analysis**: The hypothesis was wrong on both counts. (1) Pre-activation did NOT have zero throughput cost — it lost ~6% throughput. (2) Any gradient flow benefit from cleaner identity mappings was more than offset by the 6 fewer training epochs. The parameter count was identical (4,286,026), confirming the cost is purely in per-step compute time, not model capacity. At 300s budget where every epoch matters, ~1ms/step overhead translates directly into lost accuracy.
- **Key Learning**: In a throughput-constrained regime (300s budget), cuDNN kernel fusion patterns determine effective model architecture choices — BN-ReLU-Conv ordering is ~6% slower than Conv-BN-ReLU due to lost fusion, costing 6 epochs and -0.23pp.

## Verification

- **Conditions**: Condition 1 FAIL (96.23 < 96.56), Conditions 2-3 PASS
- **Review Notes**: Results confirmed trustworthy. 10-field summary printed correctly. Eval count (93) equals epoch count (93). The regression is consistent with the throughput loss.
- **Verdict**: no-improvement
- **Verdict Basis**: Condition 1 failure — primary metric 96.23% did not exceed threshold 96.56% (baseline 96.46% + 0.1pp)

## Unexplored Avenues

- **Custom fused pre-activation kernel**: A handwritten CUDA kernel that fuses BN→ReLU→Conv could recover the throughput loss. However, this requires CUDA kernel development expertise and is far beyond the scope of modifying `train.py`.
- **Pre-activation only at non-downsampling blocks**: Applying pre-activation ordering selectively (e.g., only at stride-1 blocks where the shortcut is clean identity) while keeping post-activation at stride-2 transitions might capture partial gradient flow benefits with less throughput cost.
- **Pre-activation combined with a throughput-positive change**: If a future experiment recovers epochs (e.g., faster data loading, reduced eval overhead), pre-activation could be reconsidered as a secondary change layered on top.

## Next Steps

1. **Deeper architecture (NUM_BLOCKS=4, ResNet-26)** — medium confidence. Adds ~33% more FLOPs but may improve capacity enough to offset the epoch reduction. EXP-020 report flagged this as the top next step. Need to estimate per-step overhead carefully given the lesson that even small per-step costs compound.
2. **SAM optimizer with reduced overhead** — low confidence. Standard SAM halves epochs (2x compute), likely fatal. But Efficient SAM (ESAM) or 1-step SAM with periodic application might work. High risk but targets a novel axis (optimization geometry vs data/architecture).
3. **Cutout replacing RandomErasing** — medium confidence. airbench96 uses 12px Cutout rather than RandomErasing. Different occlusion mechanism may yield +0.1-0.3pp. Zero throughput cost.

## Exit Action Results

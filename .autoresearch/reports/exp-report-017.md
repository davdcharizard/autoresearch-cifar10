# Report EXP-017: SE channel attention in BasicBlock
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Log**: logs/exp-log-017.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, higher is better) within a 300s single-GPU training budget, modifying only train.py. Current baseline: 96.39% (EXP-016, TTA with horizontal flip on top of k=4 ResNet-20 with EMA).

## Idea & Hypothesis

Add Squeeze-and-Excitation (SE) channel attention modules to each BasicBlock in the ResNet. SE recalibrates channel responses via global average pooling → FC reduce (r=16) → ReLU → FC expand → Sigmoid → element-wise scale. The hypothesis was that SE blocks would improve accuracy by +0.2-0.5% through adaptive channel-wise feature recalibration, making each residual block more expressive without meaningful parameter or compute overhead.

## Approach

Added SEBlock class to train.py and integrated into BasicBlock.forward() between the second conv+BN and the residual addition. Used reduction ratio r=16 with min bottleneck of 4. Used bias=False on both FC layers. Parameter increase: 4,301,898 → 4,360,010 (+58K, +1.3%).

## Execution

Two runs were required:

**Run 1**: Severe training instability — model stuck at 10-21% accuracy for 13 epochs, loss diverged to 38.3. Root cause: ResNet's `_weights_init` applied `kaiming_normal_` to SE FC layers, causing random sigmoid gate values that severely distorted features. Result: 93.14%.

**Run 2**: Fixed by zero-initializing SE FC2 weights after model init (sigmoid(0)=0.5 uniform scaling). Training converged normally but model only reached 94.59% — significantly below baseline.

## Results

- **Primary metric**: 94.59% (baseline: 96.39%, delta: -1.80%, -1.87%)
- **Observations**: The fixed model got 50 epochs (vs ~54 baseline) — SE overhead costs ~4 epochs per 300s budget. Even accounting for the epoch loss, 94.59% is far below the pre-TTA baseline of 95.73%. The sigmoid(0)=0.5 initialization means all features are initially halved, wasting early training capacity as the model must learn to compensate. best==final suggests convergence was not the issue — the model simply converges to a worse optimum with SE.
- **Analysis**: SE blocks are counterproductive for this specific setup. Three contributing factors: (1) the 0.5x initial feature scaling wastes training capacity in a budget-limited regime, (2) SE adds per-step overhead reducing total epochs from ~54 to 50, (3) the model is too shallow (9 blocks) for SE to provide meaningful channel differentiation — there aren't enough layers for the learned channel weighting to cascade into useful representations. The hypothesis that SE would improve accuracy was wrong for this architecture and training budget.
- **Key Learning**: SE blocks hurt shallow ResNets under tight time budgets; the 0.5x init scaling and per-step overhead outweigh the channel attention benefit at 9 blocks / 50 epochs.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (actual: 94.59%)
- **Review Notes**: Results confirmed trustworthy — training converged normally in Run 2, no signs of evaluation issues
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 94.59% failed to exceed baseline 96.39% + 0.1% threshold

## Unexplored Avenues

- SE with identity-like initialization (bias on FC2 initialized to +2.0 so sigmoid ≈ 0.88) might reduce the initial scaling penalty, but the per-step overhead would remain
- CBAM (channel + spatial attention) — adds spatial attention on top of channel attention, but would exacerbate the per-step overhead problem
- Lightweight attention alternatives like ECA (Efficient Channel Attention using 1D convolution instead of FC layers) — lower overhead but unclear if the fundamental issue (shallow model) would be addressed

## Next Steps

1. **Self-distillation from EMA model** (medium confidence) — use existing EMA model as teacher with KL divergence loss. No architectural overhead, leverages already-computed EMA predictions. Risk: CutMix interaction with soft targets is unclear.
2. **Extended TTA with spatial shifts** (medium confidence) — add ±1px shift augmentations at test time alongside horizontal flip. Zero training cost, builds on the +0.66% success of EXP-016 TTA. Risk: diminishing returns from additional views at 32×32 resolution.
3. **Cosine schedule tuning with higher peak LR** (low confidence) — incremental optimization of existing hyperparameters. May squeeze out 0.1-0.2% but likely near the ceiling for this approach.

## Exit Action Results

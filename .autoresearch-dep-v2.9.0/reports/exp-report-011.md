# Report EXP-011: Squeeze-and-Excitation (SE) Blocks in BasicBlock
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Log**: logs/exp-log-011.md

## Goal

Maximize CIFAR-10 test accuracy (higher is better). Baseline: 95.39% (EXP-009, commit cfe19c2). Verification threshold: best_test_acc > 95.49% (baseline + 0.1pp).

## Idea & Hypothesis

Add Squeeze-and-Excitation (SE) channel attention blocks (reduction r=16) to each BasicBlock in the WIDTH_MULT=4 ResNet-20. SE was chosen as the strongest candidate from brainstorming because it adds a genuinely new capability (channel attention) orthogonal to all prior improvements (width, augmentation, throughput, batch scaling). The hypothesis predicted SE would improve accuracy from 95.39% to ~95.6-95.9% with negligible per-step overhead (<1ms), preserving the ~98 epoch count.

## Approach

Added `SEBlock(nn.Module)` class to `train.py`: global average pooling via `x.mean(dim=(2,3))`, two FC layers (channels→channels//16→channels) with ReLU/Sigmoid, channel-wise rescaling. Modified `BasicBlock.__init__` to instantiate `self.se = SEBlock(out_channels)` and `BasicBlock.forward` to apply `out = self.se(out)` after `self.bn2(self.conv2(out))` before residual addition. Used `bias=False` on FC layers following Hu et al. No hyperparameter changes. SE's FC layers get kaiming_normal_ init via the existing `self.apply(self._weights_init)`.

## Execution

Single local run on H20 GPU, 300s training budget. Training started normally — loss decreased from 2.03 to 1.48 over first 350 steps. LR warmup correct. No NaN/Inf/OOM errors. Completed 83 epochs (16,086 steps) in 300s. No retries or adjustments needed.

## Results

- **Primary metric**: 95.45% (baseline: 95.39%, delta: +0.06pp, +0.06%)
- **Observations**: Per-step time ~18-19ms — nearly double the baseline ~9-10ms. SE overhead is ~9ms/step, far exceeding the predicted <1ms. This halved throughput from ~26K to ~13.9K img/s and reduced epoch count from ~98 to 83 (15% fewer). Peak VRAM increased from ~865MB to 1034MB (+169MB). Parameter count increased by only 28K (0.7%), as predicted.
- **Analysis**: The hypothesis was partially correct — SE blocks did improve per-epoch feature quality (95.45% in 83 epochs vs 95.39% in 98 epochs suggests the per-epoch learning rate was higher with SE). However, the severe throughput penalty completely negated this gain. The overhead likely stems from the `nn.Linear` layers breaking the channels_last memory format, causing format conversions on every forward pass through every block (9 SE modules × 2 FC layers × 2 directions). The brainstorm's <1ms estimate was based on parameter count, not memory format interaction with AMP/channels_last.
- **Key Learning**: On H20 under AMP + channels_last, `nn.Linear` inside conv blocks causes ~9ms/step overhead from memory format conversions — architectural additions must preserve the channels_last data path to maintain throughput.

## Verification

- **Conditions**: Condition 1 FAILED (best_test_acc 95.45 ≤ 95.49)
- **Review Notes**: Results confirmed trustworthy — metric extracted directly from run.log summary block, training completed normally, no anomalies detected. Both unchecked conditions (summary block present, evals ≤ epochs) would have passed.
- **Verdict**: no-improvement
- **Verdict Basis**: Condition 1 failure — primary metric did not exceed baseline + 0.1pp threshold (95.45 ≤ 95.49)

## Unexplored Avenues

- **Conv1x1-based SE instead of Linear**: Replace `nn.Linear` with `nn.Conv2d(C, C//r, 1)` operating on the (B, C, 1, 1) pooled tensor. This preserves the channels_last memory format throughout the SE path, potentially eliminating the ~9ms/step overhead while providing identical mathematical computation. If overhead drops to <1ms as originally predicted, the ~98 epoch count is preserved and the per-epoch quality gain from SE would compound over more epochs.
- **Selective SE (layer3 only)**: Apply SE only to the third layer group (channels=256) where the reduction ratio r=16 gives a 16-neuron bottleneck — most expressive. This reduces overhead by 2/3 while targeting the layers with the richest feature space.
- **ECA (Efficient Channel Attention)**: Replace SE's FC layers with a single 1D convolution (`nn.Conv1d` with adaptive kernel size), which has even lower overhead and avoids the Linear layer format issue entirely.

## Next Steps

1. **Conv1x1-based SE** (high confidence): The most direct fix — same mechanism, channels_last-compatible. If the overhead hypothesis is correct, this should recover the ~98 epoch count while retaining SE's quality benefit. Expected: ~95.5-95.7%.
2. **OneCycleLR schedule** (medium confidence): Smooth cosine profile may better exploit the current architecture than step-decay, especially in the final training phase. But contradicts the validated (0.5, 0.75) HIGH importance pattern.
3. **Stochastic Weight Averaging (SWA)** (medium confidence): Purely additive to the existing recipe in the final 25% of training. Zero throughput cost. Expected: modest +0.1-0.3pp gain.

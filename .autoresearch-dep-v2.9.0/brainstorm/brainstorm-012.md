# Brainstorm EXP-012
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention for Deep CNNs** (https://arxiv.org/pdf/1910.03151)
  ECA avoids FC-layer dimensionality reduction entirely, using a single 1D convolution for cross-channel interaction. Achieves competitive accuracy with minimal parameters. Key insight: avoiding FC layers eliminates the memory format conversion overhead observed in EXP-011.

- **Lightweight Channel Attention for Efficient CNNs (LCA)** (https://arxiv.org/html/2601.01002)
  Compares SE, ECA, and LCA on CIFAR-10 with ResNet-18. LCA achieves 94.68% on ResNet-18 adapted for CIFAR-10. Confirms channel attention modules provide consistent gains on small CIFAR models.

- **Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits** (https://arxiv.org/html/2411.18704v1)
  EMA models learn more general representations. Decay ~0.999 is typical for ~100 epoch training. Update is trivial (no throughput cost). EMA improves generalization without any architecture or schedule changes.

- **Averaging Weights Leads to Wider Optima and Better Generalization (SWA)** (https://arxiv.org/pdf/1803.05407)
  SWA achieves >0.4% improvement on CIFAR-10 with PreResNet-164. Finds wider optima. However, SWA traditionally requires switching to a constant/cyclical LR in the averaging phase, which would conflict with our validated wall-clock-fractional schedule.

## Experimental History Review

- **Current best**: 95.39% (EXP-009, batch 256 + LR 0.2 + 5-epoch warmup, commit cfe19c2)
- **Improvement trajectory**: BASE 91.72 → width-2x 92.29 → +aug 92.92 → +WD 93.33 → +AMP 94.44 → width-4x 94.82 → batch-256 95.39. Each improvement compound: capacity, then regularization, then throughput.
- **Throughput ceiling**: ~98 epochs in 300s at ~9-10ms/step with WIDTH_MULT=4, batch 256. Batch doubling gave sublinear returns (EXP-009). torch.compile zero benefit (EXP-008). Throughput gains from here require architecture-level changes that maintain channels_last.
- **Channel attention (EXP-011)**: SE blocks with nn.Linear doubled per-step time (~18-19ms) due to channels_last format breaking. But per-epoch quality was higher: 95.45% in 83 epochs vs 95.39% in 98 epochs. The idea has merit — the implementation was wrong.
- **Over-regularization (EXP-010)**: CutMix alpha=1.0 stacked on TrivialAugmentWide+RandomErasing+WD=5e-4 over-regularized. Insight: "reduce α or replace RandomErasing rather than adding on top."
- **LR schedule is near-optimal**: The (0.5, 0.75) wall-clock-fractional step decay is HIGH importance — shifting drops earlier hurt (EXP-006). Any schedule replacement is high risk.
- **Untried approaches**: Conv1x1-based SE (direct fix for EXP-011), EMA/weight averaging (purely additive), lighter cross-sample augmentation (Mixup with low alpha), increased depth.

## Candidate Ideas

### 1. Conv1x1-based SE Blocks
**Summary**: Replace the `nn.Linear` layers in SE blocks with `nn.Conv2d(kernel_size=1)` operating on the (B, C, 1, 1) pooled tensor. This preserves the channels_last memory format throughout the SE path, eliminating the ~9ms/step overhead discovered in EXP-011. The SE module becomes: global_avg_pool → Conv2d(C, C//16, 1) → ReLU → Conv2d(C//16, C, 1) → Sigmoid → channel-wise rescale. Applied to all 9 BasicBlocks. Mathematical computation identical to EXP-011's SE, only the tensor layout changes.

**Reasoning**: EXP-011 demonstrated that SE blocks improve per-epoch feature quality (95.45% in 83 epochs vs 95.39% in 98 epochs), meaning the model extracts more accuracy per training step with channel attention. The failure was purely an implementation issue — nn.Linear breaks channels_last format on H20, causing memory format conversions every forward pass. Conv2d(1x1) operates natively on NCHW/channels_last tensors, so the overhead should be negligible (<1ms/step). With ~98 epochs preserved, the per-epoch quality gain compounds over 18% more epochs than EXP-011 got. Expected accuracy: ~95.6-95.9%.

**Sources**: EXP-011 report (reports/exp-report-011.md § Unexplored Avenues #1), ECA-Net paper (1D conv avoids FC overhead), goal-learnings § Failed Approaches (SE blocks medium-importance with mechanism identified)

**Estimated Effort**: low — minimal code change (replace Linear with Conv2d, adjust tensor shapes)

**Risk Assessment**: Main risk is that Conv2d(1x1) on (B,C,1,1) tensors still incurs some overhead from the tiny spatial dimensions, though much less than format conversion. Worst case: small overhead reduces epochs from 98 to ~93-95, and the per-epoch quality gain is too small to clear the 0.1pp threshold. The change is self-contained and easy to revert.

### 2. Exponential Moving Average (EMA) of Model Weights
**Summary**: Maintain an exponential moving average of all model parameters during training. After each optimizer step, update shadow parameters: `ema_param = decay * ema_param + (1 - decay) * param`. At evaluation time, swap in the EMA parameters. Use decay=0.999 (standard for ~100 epoch training). Implemented purely in the training loop — no architecture changes, no schedule changes, no augmentation changes.

**Reasoning**: EMA averages over the SGD trajectory, smoothing out per-batch noise and finding a flatter region of the loss landscape. The literature confirms EMA improves generalization on CIFAR-10 without throughput cost. The update is a single vectorized multiply-add per parameter per step — negligible compared to the forward/backward pass. This is purely additive to the existing recipe and cannot hurt throughput. At ~98 epochs with ~16K steps, the EMA has sufficient history to provide meaningful averaging.

**Sources**: "Exponential Moving Average of Weights in Deep Learning" (arxiv 2411.18704), PyTorch EMA implementations (github.com/fadel/pytorch_ema), Mean Teacher paper (arxiv 1703.01780)

**Estimated Effort**: low — add ~15 lines to training loop (shadow dict, update step, eval swap)

**Risk Assessment**: Very low risk. EMA cannot degrade throughput meaningfully (one multiply-add per parameter per step). The only risk is that the gain is too small to cross the 0.1pp threshold — EMA typically provides modest improvements (+0.1-0.3pp). Worst case: no-improvement verdict with identical throughput.

### 3. Mixup (alpha=0.2) Replacing RandomErasing
**Summary**: Replace `RandomErasing(p=0.25, scale=(0.02, 0.2))` with `Mixup(alpha=0.2)` applied at the batch level during training. Mixup blends pairs of images and their labels: `x = lambda*x_i + (1-lambda)*x_j`, `y = lambda*y_i + (1-lambda)*y_j`, where lambda ~ Beta(0.2, 0.2). This swaps one regularizer for another rather than stacking, informed by EXP-010's learning that cross-sample augmentation over-regularizes when added on top of the existing stack.

**Reasoning**: EXP-010 showed CutMix(alpha=1.0) over-regularized when stacked on TrivialAugmentWide+RandomErasing+WD=5e-4. The goal-learnings insight says "reduce α or replace RandomErasing rather than adding on top." Mixup with alpha=0.2 is lighter than CutMix(alpha=1.0) — the Beta(0.2, 0.2) distribution concentrates near 0 and 1, meaning most samples are barely mixed. By replacing RandomErasing (which destroys information) with Mixup (which interpolates information), the augmentation provides inter-class boundary smoothing without the information loss.

**Sources**: EXP-010 report, goal-learnings § Failed Approaches (CutMix over-regularization), Zhang et al. 2018 "mixup: Beyond Empirical Risk Minimization"

**Estimated Effort**: low — replace one transform with batch-level mixing in training loop

**Risk Assessment**: Medium risk. Even with low alpha, Mixup changes the loss landscape (soft labels instead of hard), which interacts with the LR schedule. The Beta(0.2, 0.2) distribution is bimodal near 0/1 so most samples are barely mixed, limiting the disruption. However, soft labels may interact poorly with the step decay schedule in ways that are hard to predict. Worst case: similar to EXP-010 where convergence slows and the model doesn't reach its potential in 98 epochs.

## Idea Evaluation

**Evidence strength**: Conv1x1-SE has the strongest evidence — we have direct empirical proof from EXP-011 that SE improves per-epoch quality, and the root cause of the failure (nn.Linear breaking channels_last) is well-understood with a clear fix. EMA has strong theoretical backing and literature support but no project-specific evidence of its magnitude on this exact setup. Mixup has indirect evidence from EXP-010's failure analysis but is more speculative.

**Mechanism clarity**: Conv1x1-SE has the clearest mechanism — channel attention helps the model adaptively weight feature channels, and the Conv2d fix preserves the data path that maintains throughput. EMA's mechanism is well-understood (trajectory averaging → flatter minima) but the magnitude depends on how noisy the SGD trajectory is in this setup. Mixup's mechanism (inter-class boundary smoothing) is clear but its interaction with the existing augmentation stack is uncertain.

**Expected impact**: Conv1x1-SE has the highest expected impact. EXP-011's 95.45% in 83 epochs extrapolates to ~95.6-95.9% with 98 epochs — comfortably above the 95.49% threshold. EMA is expected to add +0.1-0.3pp, which may or may not clear the threshold. Mixup is harder to predict given the over-regularization concern.

**Risk profile**: EMA is safest (cannot hurt), Conv1x1-SE is second-safest (self-contained architectural change with clear fallback), Mixup is riskiest (changes loss landscape, uncertain interaction with schedule).

**Conclusion**: Conv1x1-SE dominates on evidence strength, mechanism clarity, and expected impact, while having acceptable risk. It is the clear lead.

## Chosen Idea
**Selected**: Conv1x1-based SE Blocks

**Why this idea**:
Conv1x1-SE has the strongest evidence base of any candidate — EXP-011 directly demonstrated that SE blocks improve per-epoch accuracy, and the root cause of the throughput penalty (nn.Linear breaking channels_last format) is well-understood with a targeted fix (Conv2d(1x1) on (B,C,1,1) tensors). This is the highest-confidence path to exceeding the 95.49% threshold because it compounds a proven per-epoch quality gain over the full ~98 epoch budget.

**Hypothesis**:
Replacing nn.Linear with nn.Conv2d(kernel_size=1) in the SE module will reduce per-step overhead from ~9ms to <1ms, preserving ~95-98 epochs in the 300s budget. With the per-epoch quality advantage observed in EXP-011 (95.45% in 83 epochs) now compounding over ~15 additional epochs, best_test_acc should reach ~95.6-95.9%, exceeding the 95.49% threshold.

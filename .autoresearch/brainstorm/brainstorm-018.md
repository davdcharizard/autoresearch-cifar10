# Brainstorm EXP-018
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **PyTorch Performance Tuning Guide — channels_last** (https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  Channels last (NHWC) memory format is recommended for convolutional networks with AMP. Conv and BatchNorm cudnn backends natively support NHWC. Combining channels_last with AMP yields 8-35% speedup on Volta GPUs for ResNet models.

- **PyTorch channels_last memory format tutorial** (https://h-huang.github.io/tutorials/intermediate/memory_format_tutorial.html)
  4D tensors can be converted to channels_last via `x.to(memory_format=torch.channels_last)`. Operators like Conv2d and BatchNorm2d support NHWC natively through cuDNN, avoiding internal format conversions.

- **Multi-crop TTA for CIFAR** (https://arxiv.org/html/2404.00498v2)
  Airbench uses 6-view TTA: original + hflip, plus ±1px spatial shifts and their flips, with weighted average. This is a standard technique for extracting more accuracy at eval time.

- **Self-Knowledge Distillation through EMA** (https://link.springer.com/article/10.1007/s00371-025-04032-2)
  EMA model as self-distillation teacher outperforms SOTA self-KD methods. Key concern: requires extra forward pass through EMA model during training, adding per-step overhead.

## Experimental History Review

- **19 experiments** (BASE through EXP-017), baseline 96.39% (EXP-016, TTA hflip)
- **Training accuracy**: 95.73% (EXP-007, the best without TTA). TTA added +0.66%
- **Architecture sweet spot**: k=4 width (4.3M params, ~54 epochs in 300s)
- **Critical learning from EXP-017**: Architectural additions with per-step overhead are costly — SE blocks reduced epochs from 54 to 50 and yielded 94.59%. Zero-overhead interventions are strongly preferred.
- **Failed approaches**: SE blocks, k>=6, stochastic depth, pre-activation, AdamW, dropout, deeper model, TrivialAugment+CutMix, Mixup, VGG-style arch, batch=256
- **Working recipe**: SGD+Nesterov, WD=5e-4, EMA(0.999), CutMix(α=1.0, p=0.5), label smoothing 0.1, warmup 5ep + cosine T_max=49, AMP + torch.compile, TTA (hflip)
- **Key pattern**: More training epochs → better convergence (k=4 with 54 ep beats k=6 with 32 ep). Any intervention that increases epochs-per-budget-second is high value.
- **Untried**: channels_last memory format, extended TTA, self-distillation, learning rate / schedule tuning

## Candidate Ideas

### 1. Channels_last (NHWC) Memory Format
**Summary**: Convert the model and training inputs to PyTorch's channels_last memory format (NHWC instead of default NCHW). This allows cuDNN convolutions and BatchNorm to operate in their native NHWC format, avoiding internal memory format conversions. Combined with AMP (already enabled), this yields 8-35% training speedup on GPUs with Tensor Cores. More speed → more epochs in the same 300s budget → better convergence. The change is: `model = model.to(memory_format=torch.channels_last)` and `inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)`.

**Reasoning**: This is the highest-leverage zero-risk intervention available. The model computations are mathematically identical — only the memory layout changes. Currently we get ~54 epochs. With 15-20% speedup, we'd get ~62-65 epochs, giving the model 8-11 more epochs of training. Since T_max=49, the cosine schedule would still reach near-zero LR by epoch 54, so additional epochs would train at very low LR. To fully exploit the extra epochs, T_max should be increased to match the actual epoch count. The key insight from the experiment history: more training epochs is the most reliable way to improve accuracy (k=4 with 54 ep significantly outperforms k=6 with 32 ep).

**Sources**: PyTorch Performance Tuning Guide, PyTorch channels_last tutorial, EXP-017 failure analysis (per-step overhead is costly → speedups are valuable)

**Estimated Effort**: low — 2 lines of code change (model format + input format), plus T_max adjustment

**Risk Assessment**: Very low risk. Memory format change is mathematically identical — cannot cause accuracy regression by itself. The only risk is if torch.compile has issues with channels_last, but PyTorch documentation explicitly states they are compatible. If speedup is minimal (e.g., GPU doesn't have Tensor Cores), worst case is no change from baseline.

### 2. Extended TTA with Spatial Shifts
**Summary**: Extend test-time augmentation beyond horizontal flip by adding ±1px spatial shifts in 4 cardinal directions. This gives 6 views per test image: original, hflip, shift-left, shift-right, shift-up, shift-down. Average all 6 logits for the final prediction. Implement by padding the input with 1-pixel reflection padding and slicing. Zero impact on training time since TTA only runs during evaluation.

**Reasoning**: EXP-016 demonstrated +0.66% from hflip TTA alone, proving the model benefits from prediction averaging. The training augmentation includes RandomCrop(32, padding=4) which teaches translation invariance. Small shifts create meaningfully different inputs at 32×32 resolution — even 1px changes ~6% of all pixel values along one edge. Each additional view reduces prediction variance. The airbench approach uses exactly this 6-view TTA pattern.

**Sources**: EXP-016 (+0.66% from hflip TTA), airbench (https://arxiv.org/html/2404.00498v2)

**Estimated Effort**: low — modify forward() in eval mode, add ~8 lines for padding and slicing

**Risk Assessment**: Very low risk. Cannot hurt training accuracy. Worst case: negligible improvement over hflip-only TTA if predictions are already stable to small shifts. Eval time increases ~3x (6 views vs 2) but doesn't count against 300s budget.

### 3. Online Self-Distillation from EMA
**Summary**: Use the existing EMA model as a teacher for online self-distillation. Add a KL divergence loss term: `total_loss = (1-α)*task_loss + α*T²*KL(student/T, ema/T)` where T=4.0 (temperature), α=0.5 (distillation weight). Delay until epoch 5 (after warmup) since the EMA model needs time to become a useful teacher. The EMA model's soft predictions encode inter-class relationships that hard labels miss.

**Reasoning**: Born Again Networks and EMA-SKD demonstrate consistent accuracy gains from self-distillation. We already maintain an EMA model at every step — this leverages existing computation. The soft targets expose class similarity structure that helps the student generalize.

**Sources**: Born Again Neural Networks, EMA-SKD paper

**Estimated Effort**: medium — requires extra forward pass through EMA model per step, KL loss computation, and careful integration with CutMix loss

**Risk Assessment**: Medium-high risk. **The fatal concern is per-step overhead.** The EMA model is uncompiled. An extra forward pass through it would add 30-50% per-step overhead, potentially reducing epochs from 54 to ~36-40. EXP-017 showed that even 4 fewer epochs (54→50) caused a massive accuracy drop. The distillation benefit would need to overcome the loss of 14-18 epochs of training, which is unlikely. Compiling the EMA model separately would help but adds ~6s startup time and is uncertain to work with manual parameter updates.

## Idea Evaluation

**Evidence strength**: Channels_last has the strongest evidence — 8-35% speedup is documented in official PyTorch docs and confirmed across ResNet architectures. Extended TTA has moderate evidence from our own EXP-016 and airbench. Self-distillation has good academic evidence but the per-step overhead concern is fatal for our specific setup.

**Mechanism clarity**: Channels_last has the clearest mechanism — NHWC avoids cuDNN format conversion overhead → faster per-step → more epochs → better convergence. Extended TTA's mechanism is clear (variance reduction through ensembling). Self-distillation's mechanism is sound (soft targets) but the practical implementation would negate the benefit through overhead.

**Expected impact**: Channels_last: if we gain 8-11 more epochs (with corresponding T_max increase), the model continues to improve at near-zero LR, likely yielding +0.2-0.5% training accuracy which then compounds with TTA. Extended TTA: likely +0.1-0.3% on top of existing hflip TTA. Self-distillation: net negative due to epoch loss.

**Risk profile**: Channels_last and Extended TTA are both very safe. Self-distillation is high risk given the overhead lessons from EXP-017.

**Verdict**: Channels_last is the clear winner. It targets the core bottleneck (training epochs per budget) with zero risk and strong evidence. Extended TTA is a good follow-up experiment. Self-distillation should be avoided due to per-step overhead incompatibility with our tight 300s budget.

## Chosen Idea
**Selected**: Channels_last (NHWC) Memory Format

**Why this idea**:
Channels_last is the highest-leverage zero-risk intervention: mathematically identical training with 8-35% documented speedup for ResNet+AMP. More speed → more epochs in 300s → better convergence. This directly addresses the core bottleneck (limited training iterations) without any of the overhead risks that doomed EXP-017.

**Hypothesis**:
Converting to channels_last memory format will speed up training by 10-20%, yielding ~60-65 epochs instead of 54. With T_max adjusted to match, this will improve best_test_acc from 96.39% to ~96.5-96.8% through better convergence in the extended training window.

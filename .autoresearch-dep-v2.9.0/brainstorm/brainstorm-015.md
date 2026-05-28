# Brainstorm EXP-015
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **hlb-CIFAR10 (fast CIFAR-10 training reference)** (web search: hlb-CIFAR10 GitHub)
  Uses label_smoothing=0.2 with cross-entropy loss, aggressive EMA warmup only in last few epochs, achieves ~95.79% in ~110s on A100. Label smoothing 0.2 is a key ingredient — higher than the 0.1 tried in EXP-004.

- **cifar10-airbench (96% in 46.3s on A100)** (web search: cifar10-airbench GitHub)
  Uses patch-whitening, Dirac init, 12-pixel cutout — architecture-specific techniques not directly transferable to standard ResNet. Confirms label smoothing and aggressive augmentation as common ingredients in fast-training recipes.

- **Mixup training literature** (web search: mixup CIFAR-10 optimal alpha)
  α ∈ [0.1, 0.4] is optimal for CIFAR-10; mixup prefers SMALLER weight decay (1e-4 vs 5e-4) because mixup itself provides regularization that partially substitutes for WD. Over-training with Mixup may hurt — aligned with our time-budget regime. Higher capacity models benefit more from mixup.

- **Label smoothing best practices** (web search: label smoothing CIFAR-10 ResNet)
  Label smoothing 0.2 validated by hlb-CIFAR10 and common in modern training recipes. EXP-004's failure at smoothing=0.1 was confounded by Nesterov overhead (cost 4 epochs), smaller width-2x model, and lower smoothing value. The technique itself has near-zero computational overhead.

## Experimental History Review

- **Current best**: 95.39% (EXP-009, commit cfe19c2) — batch 256, LR 0.2, 5-epoch warmup, ~98 epochs in 300s
- **Trajectory**: BASE 91.72 → EXP-001 92.29 → EXP-002 92.92 → EXP-003 93.33 → EXP-005 94.44 → EXP-007 94.82 → EXP-009 95.39 (7 improvements out of 16 experiments)
- **What worked**: Width scaling (+0.57pp per doubling), augmentation (TrivialAugmentWide+RandomErasing +0.63pp), WD=5e-4 (+0.41pp synergy with aug), AMP throughput (+1.11pp from 1.54x epochs), batch scaling (+0.57pp from 18% more epochs)
- **What failed**: SE blocks exhausted (2 attempts, intrinsic ~9ms overhead), EMA β=0.999 too conservative (+0.05pp), CutMix α=1.0 over-regularizes when stacked, torch.compile zero speedup, shifted LR drops reduce ceiling, Nesterov+LS 0.1 reduced epochs
- **Key pattern**: Throughput-to-accuracy conversion is the primary improvement driver — each additional epoch contributes meaningfully. Zero-overhead regularization changes are the safest bets.
- **Untried approaches**: Label smoothing 0.2 standalone (different from EXP-004's 0.1+Nesterov), Mixup with reduced WD, lower EMA β with cached references

## Candidate Ideas

### 1. Label Smoothing 0.2 (Standalone)
**Summary**: Add `label_smoothing=0.2` to the `F.cross_entropy` call in train.py. This is a single-parameter change with zero computational overhead — PyTorch's cross_entropy natively supports the `label_smoothing` kwarg. No Nesterov, no other changes. The smoothing factor 0.2 is validated by hlb-CIFAR10 and is double the 0.1 value that failed in EXP-004.

**Reasoning**: EXP-004 tried label_smoothing=0.1 + Nesterov momentum and failed (93.28% vs 93.33% baseline). However, that failure was confounded by three factors: (1) Nesterov momentum added per-step overhead costing 4 epochs, (2) the model was width-2x (~1M params) vs current width-4x (~4.3M params), and (3) smoothing=0.1 is conservative. With the current setup (width-4x, AMP, batch 256, ~98 epochs), label smoothing 0.2 standalone has zero throughput cost (no Nesterov), operates on a higher-capacity model that can absorb stronger regularization, and uses the exact value validated by hlb-CIFAR10. Label smoothing regularizes the OUTPUT distribution space, which is complementary to existing INPUT space augmentation (TrivialAugmentWide + RandomErasing). The technique works by preventing the model from becoming overconfident, improving calibration and generalization.

**Sources**: hlb-CIFAR10 (web search), EXP-004 failure analysis (goal-learnings), EXP-010 over-regularization lesson (goal-learnings)

**Estimated Effort**: low — single parameter change to F.cross_entropy call

**Risk Assessment**: Main risk is over-regularization when combined with existing TrivialAugmentWide + RandomErasing + WD=5e-4, similar to what happened with CutMix in EXP-010. However, label smoothing is much lighter than CutMix (it softens targets by 0.2 rather than blending entire images). Worst case: marginal no-improvement like EXP-014 (+0.05pp), with zero throughput cost. The approach does NOT repeat EXP-004 — different smoothing value, no Nesterov, larger model, more epochs.

### 2. Mixup α=0.2 Replacing RandomErasing
**Summary**: Replace RandomErasing with Mixup (α=0.2) as the batch-level augmentation strategy. Mixup blends pairs of training images and their labels: `x = λ*x_i + (1-λ)*x_j, y = λ*y_i + (1-λ)*y_j` where λ ~ Beta(α, α). Remove RandomErasing from the transform pipeline and add Mixup logic in the training loop after loading a batch. Reduce WD from 5e-4 to 1e-4 per literature guidance that Mixup substitutes for WD.

**Reasoning**: EXP-010 showed CutMix α=1.0 over-regularized when stacked on TrivialAugmentWide+RandomErasing+WD=5e-4. The lesson was to REPLACE rather than stack cross-sample augmentation, and use lower α. Mixup α=0.2 is conservative (α ∈ [0.1, 0.4] optimal per literature), and replacing RandomErasing avoids the stacking problem. However, reducing WD from 5e-4 to 1e-4 changes TWO variables simultaneously (augmentation type + WD), making attribution harder. The Mixup implementation adds per-step overhead (tensor ops for blending + label mixing).

**Sources**: Mixup literature (web search), EXP-010 CutMix failure (goal-learnings), EXP-003 WD synergy pattern (goal-learnings)

**Estimated Effort**: medium — requires training loop changes (batch mixing logic, label mixing), transform pipeline modification, and WD adjustment

**Risk Assessment**: Multi-variable change (Mixup + WD reduction) makes attribution difficult if the result is no-improvement. Mixup adds per-step computational overhead (blending operations), which could cost 2-5 epochs — the same throughput penalty that hurt SE blocks and EMA. WD reduction from 5e-4 to 1e-4 reverses the validated +0.41pp gain from EXP-003. If Mixup's regularization doesn't fully compensate for the lost WD, the net effect could be negative.

### 3. Cosine Annealing with Correct T_max
**Summary**: Replace the wall-clock-fractional MultiStepLR schedule with CosineAnnealingLR using T_max calibrated to actual training duration (~98 epochs). EXP-000 tried CosineAnnealingLR but with T_max=200, which was far too large for the actual ~91 epochs completed. With T_max=100 (conservative estimate of actual epochs), the LR decays smoothly from 0.2 to near 0 over the training budget, providing a gradual transition rather than the abrupt 10x drops at 50%/75%.

**Reasoning**: The current MultiStepLR with drops at 0.5/0.75 is validated and works well, but the abrupt LR drops cause instability — EXP-005 noted AMP FP16 instability at LR=0.01 during the middle phase. Cosine annealing provides a smoother transition that may be more friendly to FP16 training. However, goal-learnings explicitly state "the (0.5, 0.75) schedule is near-optimal" and "high-LR exploration time is the primary driver of accuracy ceiling." Cosine annealing spends less time at high LR compared to the 50% hold in MultiStepLR.

**Sources**: EXP-000 failure (goal-learnings — T_max mismatch), EXP-006 failure (goal-learnings — shifting drops earlier hurts), Patterns (goal-learnings — wall-clock-fractional schedule is validated)

**Estimated Effort**: low — replace the LambdaLR with CosineAnnealingLR, set T_max

**Risk Assessment**: High risk. Goal-learnings contain two strong signals against this: (1) EXP-006 showed shifting LR drops earlier reduces the accuracy ceiling — cosine starts decaying from step 1, meaning less time at peak LR. (2) The (0.5, 0.75) schedule is explicitly called "near-optimal" in Patterns. The only new argument is correct T_max, but the fundamental issue is less exploration time at high LR, which is the same failure mode as EXP-006.

## Idea Evaluation

**Label Smoothing 0.2** has the strongest evidence-to-risk profile. It's validated by hlb-CIFAR10 at the exact smoothing value, has zero throughput cost (critical given the ~98 epoch budget), and the prior failure (EXP-004) was clearly confounded by Nesterov overhead + smaller model + lower smoothing. The mechanism is clear: softening target distributions prevents overconfidence and improves generalization, operating in the output distribution space which is orthogonal to existing input-space augmentation. Worst case is a marginal no-improvement with no throughput penalty.

**Mixup α=0.2** has decent evidence but higher risk. The multi-variable change (Mixup + WD reduction) makes it harder to interpret results. The per-step overhead could cost epochs, and reversing the validated WD=5e-4 gain is a concern. It's a reasonable next experiment if label smoothing fails, but not the best first choice.

**Cosine Annealing** contradicts two explicit goal-learnings entries. While EXP-000's failure was specifically about T_max mismatch, the broader lesson from EXP-006 is that less time at high LR hurts. This is the weakest candidate.

## Chosen Idea
**Selected**: Label Smoothing 0.2 (Standalone)

**Why this idea**:
Strongest evidence (hlb-CIFAR10 validates this exact value), zero throughput cost (critical for our epoch-limited regime), clear mechanism (output distribution regularization complementary to input augmentation), and the prior failure at smoothing=0.1 is fully explained by confounding factors (Nesterov overhead, smaller model). This is the lowest-risk, highest-evidence candidate.

**Hypothesis**:
Adding label_smoothing=0.2 to F.cross_entropy — with no other changes — will prevent output overconfidence on the width-4x model, improving generalization by 0.1-0.3pp to exceed the 95.49% verification threshold. The zero throughput cost preserves the ~98 epoch budget, and the higher-capacity model (4.3M params vs 1M in EXP-004) can absorb the additional regularization without underfitting.

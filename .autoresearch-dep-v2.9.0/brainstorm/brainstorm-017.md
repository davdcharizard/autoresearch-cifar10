# Brainstorm EXP-017
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **"Mixup Without Hesitation" (Yu et al. 2021)** (https://arxiv.org/abs/2101.04342)
  Mixup in early training explores the loss landscape; later it hinders fine convergence. mWh gradually replaces mixup with standard augmentation, achieving better accuracy than full-duration mixup with less training time. Key insight: for short training budgets, mixup is most valuable during exploration (high-LR phase) and should be phased out during exploitation (low-LR phase).

- **"mixup: Beyond Empirical Risk Minimization" (Zhang et al. 2018)** (https://arxiv.org/pdf/1710.09412)
  Mixup (α=1.0) on PreActResNet-18 achieves 4.77% error on CIFAR-10 vs 5.12% without. However, α=1.0 requires more training epochs. For shorter budgets, lower α (0.1–0.4) is recommended as it provides regularization without excessive interpolation.

- **"Using mixup as regularization and tuning hyper-parameters for ResNets" (Galdran et al. 2021)** (https://arxiv.org/pdf/2111.11616)
  Systematic study of mixup alpha tuning on CIFAR-10 with ResNet. Confirms α∈[0.1, 0.4] is effective for moderate training budgets.

- **cifar10-airbench (Keller Jordan 2024)** (https://github.com/KellerJordan/cifar10-airbench)
  96% accuracy in 27.3s on A100. Key technique for 96%: 12-pixel Cutout augmentation plus wider architecture with 3 convolutions per block. The airbench94 recipe uses ghost batch norm, which is not applicable here since we use standard BN. Key insight: augmentation stacking strategy and model width are the primary levers at this accuracy range.

- **hlb-CIFAR10 (tysam-code)** (https://github.com/tysam-code/hlb-CIFAR10)
  95.79% in ~110s on A100. Uses label smoothing (confirmed by our EXP-015), high BN momentum (not useful per our EXP-016), and test-time augmentation with horizontal flip for ~0.2pp gain.

## Experimental History Review

- **Current best**: 95.57% (EXP-015, label smoothing 0.2). Baseline commit: 626e9d1.
- **Total experiments**: 18 (BASE through EXP-016), 8 improvements, 10 no-improvements.
- **Trajectory**: 91.72 → 92.29 → 92.92 → 93.33 → 94.44 → 94.82 → 95.39 → 95.57

**What worked (compounding stack)**:
- Width-2x → width-4x (capacity), AMP (throughput → more epochs), batch 256 (throughput), TrivialAugmentWide+RandomErasing (regularization), WD=5e-4 (regularization), label smoothing 0.2 (output regularization)

**What failed and why**:
- CutMix α=1.0 (EXP-010): over-regularized when stacked on existing aug → approach: replace instead of stack, lower α
- SE blocks (EXP-011/012, count=2): intrinsic ~9ms/step overhead on H20 — exhausted
- EMA β=0.999 parameter-only (EXP-013): BN buffer mismatch — fundamentally broken
- EMA β=0.999 full state_dict (EXP-014): only +0.05pp, β too conservative for 92 epochs, throughput cost offset smoothing
- BN momentum 0.5 (EXP-016): negligible at 98 epochs
- torch.compile (EXP-008): no fusion opportunities on H20 for this model
- Shifted LR schedule (EXP-006): high-LR exploration time is primary driver
- Nesterov+LS=0.1 (EXP-004): per-step overhead cost epochs

**Untried gaps**:
- Mixup/cross-sample augmentation at moderate α replacing (not stacking) existing augmentation
- Stochastic depth / DropPath (throughput-neutral structural regularization)
- EMA with lower β (0.995) tuned for ~98 epoch budget (EXP-014 insight: "idea not exhausted")
- Gradient clipping for AMP stability (low confidence per EXP-016 report)

## Candidate Ideas

### 1. Mixup α=0.2 Replacing RandomErasing
**Summary**: Replace RandomErasing(p=0.25, scale=(0.02, 0.2)) with Mixup(α=0.2) applied at the batch level during training. Mixup generates interpolated training pairs: x̃ = λx_i + (1-λ)x_j, ỹ = λy_i + (1-λ)y_j, where λ ~ Beta(α, α). This swaps per-sample occlusion regularization for cross-sample interpolation regularization — a qualitatively different signal that encourages smoother decision boundaries. α=0.2 produces λ values concentrated near 0 and 1 (median ~0.85), providing mild regularization without the excessive interpolation that caused EXP-010's failure.

**Reasoning**: EXP-010 failed because CutMix α=1.0 was stacked on top of TrivialAugmentWide+RandomErasing+WD=5e-4, causing over-regularization (model still improving at epoch 96). This idea addresses both failure modes: (1) replacing instead of stacking reduces total regularization burden, (2) α=0.2 is far milder than α=1.0. The "Mixup Without Hesitation" paper (Yu et al. 2021) confirms mixup is most beneficial during the high-LR exploration phase, which aligns with our wall-clock-fractional schedule where 50% of budget is high-LR. Literature consistently shows mixup improving generalization on CIFAR-10 ResNets by 0.2-0.5pp. EXP-016 report specifically recommended this as the top next step.

**Sources**: Zhang et al. 2018 (https://arxiv.org/pdf/1710.09412), Yu et al. 2021 (https://arxiv.org/abs/2101.04342), Galdran et al. 2021 (https://arxiv.org/pdf/2111.11616), EXP-010 failure analysis (goal-learnings § Failed Approaches), EXP-016 report § Next Steps

**Estimated Effort**: low — ~15 lines of code change in train.py (remove RandomErasing transform, add batch-level mixup in training loop)

**Risk Assessment**: Main risk is that even mild mixup (α=0.2) combined with TrivialAugmentWide + WD=5e-4 + label_smoothing=0.2 could still over-regularize for a 98-epoch budget. However, RandomErasing removal frees regularization budget. Worst case: no-improvement, with accuracy near baseline. Mixup implementation must handle the soft targets correctly with cross-entropy — requires switching from hard-target cross_entropy to manual soft-target loss computation.

### 2. Stochastic Depth (DropPath) on BasicBlock
**Summary**: Add stochastic depth (Huang et al. 2016) to the ResNet-20 BasicBlock: during training, each residual block is randomly skipped with probability that increases linearly from 0 (first block) to p_max (last block). When a block is skipped, only the shortcut path propagates. This is a throughput-neutral structural regularizer orthogonal to all existing augmentation and regularization. At inference, all blocks are active with outputs scaled by (1 - drop_prob).

**Reasoning**: All current regularization operates on input space (TrivialAugmentWide, label smoothing) or weight space (WD). Stochastic depth regularizes the architecture itself by preventing co-adaptation between blocks — a fundamentally different axis. For ResNet-20 with 9 BasicBlocks, a modest p_max=0.1 means the last block drops 10% of iterations. The compute overhead is near-zero (a single random sample + multiplication per block), preserving the 98-epoch budget. This is well-established for deeper ResNets; for shallow ResNets the gain is smaller but still consistently positive in literature.

**Sources**: Huang et al. 2016 "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382), commonly used in EfficientNet/DeiT architectures

**Estimated Effort**: low — ~10 lines added to BasicBlock.forward() for the stochastic skip logic

**Risk Assessment**: With only 9 blocks in ResNet-20, stochastic depth has less room to help compared to deeper architectures (50+ layers). p_max must be very conservative (0.05-0.15) to avoid under-training the later blocks. Worst case: negligible improvement, model trains normally with occasional block skips that don't materially change learning dynamics. The technique is very safe — no risk of divergence or instability.

### 3. Full State_dict EMA with Lower β=0.995
**Summary**: Revisit Exponential Moving Average (EMA) with full state_dict (including BN buffers) but with β=0.995 instead of the β=0.999 that was too conservative in EXP-014. β=0.995 averages over the last ~200 steps (vs ~1000 for β=0.999), making the shadow model more responsive to recent optimization progress. Additionally, use in-place tensor operations for the EMA update to minimize per-step overhead (EXP-014 lost ~6 epochs from state_dict copying).

**Reasoning**: EXP-014 established that full state_dict EMA fixes the BN mismatch from EXP-013, and the goal-learnings explicitly state "EMA β must be tuned to epoch count (lower β for shorter training). Cached tensor references could recover throughput. Idea not exhausted — lower β + efficient implementation may work." With β=0.995 and ~98 epochs (~19K steps), the effective averaging window is ~200 steps, which should track the optimization trajectory closely enough to smooth late-training noise without lagging. In-place updates (`shadow.lerp_(param, 1-β)`) avoid the per-step state_dict+deepcopy overhead.

**Sources**: EXP-013 report (parameter-only EMA failure), EXP-014 report (full state_dict EMA, β too conservative), goal-learnings § Failed Approaches (explicit "idea not exhausted" note)

**Estimated Effort**: medium — requires maintaining a shadow parameter dict with in-place updates, swapping for evaluation, and handling the BN buffer correctly

**Risk Assessment**: Even with in-place ops, EMA adds per-step overhead that costs epochs. If the overhead exceeds ~1ms/step, we lose ~5 epochs which may offset the smoothing benefit (same dynamic as EXP-014). β=0.995 might be too aggressive for early training where parameters change rapidly. There's also implementation complexity in correctly handling the swap for evaluation without disrupting BN running stats. Worst case: small overhead with marginal gain, similar to EXP-014's +0.05pp.

## Idea Evaluation

**Evidence strength**: Mixup has the strongest evidence base — multiple papers demonstrate consistent 0.2-0.5pp improvement on CIFAR-10 with ResNet, the failure mode of EXP-010 is well-understood (stacking + aggressive α), and the fix (replacing + mild α) directly addresses both root causes. EMA has project-specific evidence that the idea is "not exhausted" but the marginal gain from EXP-014 was only +0.05pp even with the correct implementation. Stochastic depth has strong literature support but primarily for deeper networks; evidence for 20-layer ResNets is weaker.

**Mechanism clarity**: Mixup's mechanism is clear — cross-sample interpolation produces smoother decision boundaries, which is a qualitatively different regularization signal from per-sample augmentation. Stochastic depth's mechanism (preventing block co-adaptation) is well-understood but its magnitude for only 9 blocks is uncertain. EMA's mechanism (averaging out late-training noise) is clear, but the key question is whether the averaging window at β=0.995 is well-matched to our noise level.

**Expected impact**: Mixup has the highest expected impact — literature suggests 0.2-0.5pp, and we're replacing (not adding) regularization. Stochastic depth is likely +0.05-0.15pp for a shallow network. EMA at β=0.995 is speculative — could match or slightly exceed the +0.05pp from β=0.999.

**Risk profile**: All three are safe (worst case: no-improvement). Mixup requires careful loss computation with soft targets but has no instability risk. Stochastic depth is trivially safe. EMA has the most implementation complexity.

**Feasibility**: All are low-to-medium effort. Mixup and stochastic depth are simpler to implement correctly.

**Verdict**: Mixup α=0.2 replacing RandomErasing is the strongest choice — it has the best evidence, clearest mechanism, highest expected impact, and directly addresses the identified failure mode from EXP-010 with a well-motivated alternative approach.

## Chosen Idea
**Selected**: Mixup α=0.2 Replacing RandomErasing

**Why this idea**:
Strongest evidence from both literature (consistent 0.2-0.5pp gains on CIFAR-10 ResNets) and project history (EXP-010 failure root cause is well-understood and directly addressed by replacing instead of stacking, and using mild α=0.2 instead of aggressive α=1.0). The technique provides a qualitatively different regularization signal (cross-sample interpolation) orthogonal to existing per-sample augmentation, and removing RandomErasing frees regularization budget for the new technique.

**Hypothesis**:
Replacing RandomErasing with Mixup α=0.2 will improve best_test_acc by 0.1-0.4pp over the 95.57% baseline, achieving ≥95.67%. The mild α produces near-identity interpolations that regularize without the convergence slowdown seen with α=1.0 in EXP-010, and the replacement (rather than stacking) avoids over-regularization. The cross-sample signal encourages smoother decision boundaries that compound with TrivialAugmentWide's per-sample augmentation.

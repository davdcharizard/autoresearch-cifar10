# Brainstorm EXP-021
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **SAM: Sharpness-Aware Minimization (Foret et al. 2020)** (https://arxiv.org/pdf/2010.01412)
  SAM adds a perturbation step before each gradient update, seeking parameters that have uniformly low loss across neighborhoods. Reported +0.5-1.0pp on CIFAR-10 ResNets over SGD. 2x compute cost per step (two forward-backward passes). Efficient variants (ESAM) reduce cost to ~1.4x.

- **Pre-activation ResNet blocks (He et al. 2016)** (https://arxiv.org/pdf/1603.05027)
  BN-ReLU-Conv ordering (instead of Conv-BN-ReLU) preserves clean identity mappings on the residual path. Standard in all competitive CIFAR-10 recipes (WRN, speedrun architectures). Improves optimization and acts as mild regularization. Zero throughput cost — same parameter count and FLOPs.

- **airbench96 recipe (Keller Jordan 2024)** (https://github.com/KellerJordan/cifar10-airbench)
  Reaches 96.05% in 46.3 A100-seconds with 9-layer ConvNet. Key techniques: 12px Cutout, translation padding=4, TTA (flip+translate), wider channels. Uses pre-activation-style blocks.

- **Wide Residual Networks (Zagoruyko & Komodakis 2016)** (https://github.com/szagoruyko/wide-residual-networks)
  Width strongly preferred over depth at CIFAR-10 scale. WRN-28-10 outperforms thin ResNet-1001 by +0.92pp. But our WIDTH_MULT=4 already widens significantly (64/128/256 channels).

## Experimental History Review

- **Current best**: 96.46% (EXP-020, cosine LR schedule)
- **Trajectory**: BASE 91.72 → 92.29 → 92.92 → 93.33 → 94.44 → 94.82 → 95.39 → 95.57 → 95.91 → 96.46 (10 improvements in 20 experiments)
- **Regularization stack is saturated**: CutMix (EXP-010, -0.36pp), Mixup (EXP-017, -0.04pp), DropPath (EXP-018, -0.33pp) all hurt when stacked on TrivialAugmentWide + RandomErasing + WD=5e-4 + LS=0.2
- **SE blocks too slow**: 9ms/step overhead regardless of implementation (EXP-011, EXP-012) — throughput is the bottleneck
- **EMA marginal**: Full state_dict EMA at β=0.999 gives only +0.05pp (EXP-014), too conservative for ~100 epoch budget
- **torch.compile zero effect on H20**: (EXP-008) — skip compiler optimizations
- **Validated recipes**: Cosine decay to ~0 (EXP-020), TTA (EXP-019), LS=0.2 (EXP-015), batch 256 + warmup (EXP-009), AMP (EXP-005), width-4x (EXP-007)
- **Untried approaches**: Pre-activation blocks, SAM/ASAM optimizer, deeper model (NUM_BLOCKS=4), Cutout replacing RandomErasing, gradient clipping

## Candidate Ideas

### 1. Pre-activation ResNet Blocks (BN-ReLU-Conv ordering)

**Summary**: Change the BasicBlock from post-activation (Conv-BN-ReLU) to pre-activation (BN-ReLU-Conv) ordering per He et al. 2016 "Identity Mappings in Deep Residual Networks." In the current implementation, the forward path is `F.relu(bn1(conv1(x)))` → `bn2(conv2(...))` → `add shortcut` → `F.relu(...)`. The pre-activation version moves BN and ReLU before the convolutions: `bn1 → relu → conv1 → bn2 → relu → conv2`, and the residual addition happens without any activation after it — creating a clean identity mapping on the shortcut path. This structural change improves gradient flow and acts as implicit regularization with zero throughput cost (same param count, same FLOPs).

**Reasoning**: Pre-activation blocks are the standard in virtually all competitive CIFAR-10 recipes (WRN, airbench96, hlb-CIFAR10). The original ResNet paper showed pre-activation improves optimization particularly for deeper networks, but the benefit extends to shallower networks too — cleaner gradient flow through identity shortcuts helps convergence in the final cosine decay phase where the LR is very small. With our cosine schedule decaying to ~0, the optimizer needs the cleanest possible gradient signal in the final epochs. Zero throughput cost means no epoch reduction.

**Sources**: He et al. 2016 (https://arxiv.org/pdf/1603.05027), airbench96 recipe, WRN architecture

**Estimated Effort**: medium — requires rewriting BasicBlock and adjusting the initial conv/BN/ReLU and final pooling layer

**Risk Assessment**: Low risk. Pre-activation is extremely well-validated across the CIFAR-10 literature. The main concern is implementation correctness — the shortcut path changes when using pre-activation (need to handle the first block of each layer group differently). Worst case: small regression from incorrect handling, easily caught by verification.

### 2. Sharpness-Aware Minimization (SAM) Optimizer

**Summary**: Replace the standard SGD optimizer with SAM-wrapped SGD. SAM performs a two-step optimization: (1) perturb weights in the direction of steepest loss increase (ascent step with ρ=0.05), (2) compute the gradient at the perturbed point and update using that gradient (descent step). This seeks parameters that lie in flat minima, which generalize better. The implementation adds ~15 lines: a SAM wrapper class that stores the perturbation, does the ascent step, and restores weights before the descent step.

**Reasoning**: SAM consistently adds +0.5-1.0pp on CIFAR-10 across ResNet architectures in the literature. However, SAM requires 2 forward-backward passes per step, roughly doubling per-step compute. At our current ~16ms/step with AMP, this could push to ~30ms/step, reducing epoch count from ~99 to ~50. The question is whether the per-epoch quality gain from flat-minima seeking outweighs losing ~50 epochs. Given the regularization stack saturation (CutMix, Mixup, DropPath all failed), SAM targets a fundamentally different axis — optimization geometry rather than data augmentation — which might break through the plateau.

**Sources**: Foret et al. 2020 (https://arxiv.org/pdf/2010.01412), WRN+SAM benchmarks showing 97.21% on WRN-28-10

**Estimated Effort**: medium — SAM wrapper class + integration with AMP GradScaler (need to handle scaled gradients correctly)

**Risk Assessment**: High risk. The 2x per-step compute cost is the critical concern. At ~50 epochs with cosine decay, the model may not have enough training steps to converge. The cosine schedule's ESTIMATED_EPOCHS=100 would need adjustment to ~50, but the actual epoch count depends on the per-step overhead. AMP + SAM interaction needs careful handling — the ascent step should use unscaled gradients. If the throughput cost is too high, this could easily regress like SE blocks did (EXP-011/012: -15 epochs → worse accuracy despite better per-epoch quality).

### 3. Deeper Architecture (NUM_BLOCKS=4, ResNet-26)

**Summary**: Increase NUM_BLOCKS from 3 to 4, changing from ResNet-20 (6×3+2=20 layers) to ResNet-26 (6×4+2=26 layers). This adds 2 BasicBlocks per resolution stage (6 total blocks, +66% depth), increasing from 9 to 12 total blocks and from ~4.29M to ~5.71M parameters. The wider representation at each stage gets more nonlinear transformations, potentially improving feature quality.

**Reasoning**: EXP-020 report suggested this as the top next step (medium confidence). Width has been scaled to 4x (EXP-007) and further widening would increase FLOPs more than depth does per parameter. Adding 3 blocks costs ~3ms/step overhead based on the proportional FLOP increase (4.29M→5.71M = 1.33x FLOPs), which would reduce epochs from ~99 to ~75. This is a moderate throughput cost (not as severe as SE blocks' 2x slowdown). The cosine schedule adapts naturally — just set ESTIMATED_EPOCHS=75.

**Sources**: EXP-020 report Next Steps, WRN literature (depth vs width trade-offs), He et al. 2015

**Estimated Effort**: low — single constant change (NUM_BLOCKS = 4) plus adjusting ESTIMATED_EPOCHS

**Risk Assessment**: Medium risk. The per-step overhead could be higher than estimated if memory bandwidth becomes a bottleneck at 5.71M params (currently at 864.6 MB VRAM, H20 has ~96GB so headroom is ample). The epoch reduction from ~99 to ~75 means less total optimization — whether the deeper model's capacity overcomes this depends on whether the current model is capacity-limited or optimization-limited. Given the regularization saturation evidence, the model may indeed be capacity-limited.

## Idea Evaluation

**Pre-activation blocks vs SAM vs deeper architecture:**

- **Evidence strength**: Pre-activation blocks have the strongest evidence — they are the universal standard in competitive CIFAR-10 recipes and backed by the seminal He 2016 paper. SAM has strong evidence but the 2x compute cost makes the evidence less transferable to our throughput-constrained setting. Deeper architecture has moderate evidence (general depth scaling, EXP-020 suggestion).

- **Mechanism clarity**: Pre-activation has a clear mechanism — clean identity mappings improve gradient flow, especially beneficial during the cosine decay's final near-zero LR phase. SAM's mechanism (flat minima seeking) is well-understood but its effectiveness is unclear when halving the epoch count. Deeper architecture's mechanism is straightforward capacity increase.

- **Expected impact**: Pre-activation is likely +0.1-0.3pp based on the literature for pre-act vs post-act at this model scale. SAM could be +0.5-1.0pp IF the throughput cost doesn't destroy it, but could also regress. Deeper architecture is likely +0.1-0.4pp net.

- **Risk profile**: Pre-activation has the safest failure mode — zero throughput cost means worst case is a small regression from implementation differences. SAM has the worst risk — 2x compute cost could cause the same failure mode as SE blocks (EXP-011/012). Deeper architecture has moderate risk — some throughput cost but manageable.

- **Feasibility**: Pre-activation requires careful but well-documented code changes. SAM requires AMP+GradScaler integration. Deeper architecture is trivially implemented.

Pre-activation blocks are the clear winner: strongest evidence, zero throughput cost (critical given our throughput-constrained setup), clear mechanism aligned with our cosine schedule, and lowest risk. The fact that every competitive CIFAR-10 recipe uses pre-activation blocks while our model still uses post-activation is a clear gap to close.

## Chosen Idea
**Selected**: Pre-activation ResNet Blocks (BN-ReLU-Conv ordering)

**Why this idea**:
Pre-activation blocks are the standard in all competitive CIFAR-10 recipes and our model is the only remaining component using the older post-activation ordering. The change has zero throughput cost (same params, same FLOPs, same epoch count), which is critical in our throughput-constrained 300s budget where slower techniques like SE blocks and SAM lose more from reduced epochs than they gain in per-epoch quality. The mechanism — clean identity mappings improving gradient flow — is directly synergistic with our cosine decay to near-zero LR, where gradient signal quality matters most.

**Hypothesis**:
Switching from post-activation (Conv-BN-ReLU) to pre-activation (BN-ReLU-Conv) blocks will improve best_test_acc by +0.1-0.3pp (from 96.46% to 96.56-96.76%) by improving gradient flow through identity shortcuts, particularly during the final cosine decay phase. Epoch count and throughput will remain unchanged at ~99 epochs in 300s.

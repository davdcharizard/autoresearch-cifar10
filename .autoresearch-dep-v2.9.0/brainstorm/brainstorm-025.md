# Brainstorm EXP-025
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Gradient Centralization: A New Optimization Technique for Deep Neural Networks** (https://arxiv.org/pdf/2004.01461, https://github.com/Yonghongwei/Gradient-Centralization)
  GC operates directly on gradients by centralizing gradient vectors to have zero mean. For weight tensors with dim > 1 (conv/linear), subtract the mean across all dimensions except the output channel. GC regularizes both weight space and output feature space, improves Lipschitzness of the loss function and its gradient, making training more stable and efficient. Implementation is one line of code per parameter. The paper demonstrates improved generalization across multiple architectures and datasets including CIFAR-10. Works with SGD, Adam, and other optimizers.

- **GCSAM: Gradient Centralized Sharpness Aware Minimization** (https://arxiv.org/html/2501.11584v1)
  GCSAM combines gradient centralization with sharpness-aware minimization. Achieves higher test accuracy than both SAM and Adam on ResNet50, VGG16, ViT, and Swin Transformer on CIFAR-10. Demonstrates that GC normalizes gradients, reducing noise and variance, improving stability during training.

- **CIFAR-10 ResNet-9 96% recipe** (https://github.com/eliott-bourdon-novellas/CIFAR10-ResNet9-Optimization)
  Custom ResNet-9 achieving 96.04% with AdamW optimizer, cosine decay + warmup, gradient clipping, Swish activation, and advanced augmentation. Confirms that optimizer-level tricks (gradient clipping, adaptive optimizers) are a key avenue for accuracy gains beyond 96%.

## Experimental History Review

- **Current best**: 96.46% (EXP-020) — cosine warmup+decay LR schedule
- **Improvement trajectory**: 10 successful improvements across 25 experiments, from 91.72% (BASE) to 96.46%
- **Key validated recipes**: Cosine LR decay to ~0 (EXP-020), TTA horizontal flip (EXP-019), label smoothing 0.2 (EXP-015), batch 256 + LR scaling (EXP-009), AMP FP16 (EXP-005), WIDTH_MULT=4 (EXP-007), TrivialAugmentWide + RandomErasing (EXP-002), WD=5e-4 (EXP-003)
- **Exhausted approach classes**: EMA (3 variants, EXP-013/014/023), SE blocks (2 variants, EXP-011/012), regularization stacking (CutMix/Mixup/DropPath/Cutout-swap, EXP-010/017/018/022), pre-activation blocks (EXP-021), torch.compile (EXP-008), BN momentum tuning (EXP-016), BN bias 64x LR (EXP-024)
- **Key constraint**: ~99 epochs in 300s budget; any per-step overhead directly reduces epoch count. Regularization stack is near saturation. Gains must come from optimization dynamics or capacity.
- **Untried gaps**: Gradient centralization (zero-cost optimizer trick), alternating flip augmentation (airbench96), Nesterov momentum (revisited — EXP-004 was in a completely different context), deeper architecture (NUM_BLOCKS=4)

## Candidate Ideas

### 1. Gradient Centralization (GC)
**Summary**: Apply gradient centralization to all weight tensors (dim > 1) during training. After computing gradients via backward pass, subtract the mean of each gradient tensor across all dimensions except the output channel before the optimizer step. This is inserted between `scaler.unscale_(optimizer)` and `scaler.step(optimizer)` in the training loop. Only applied to conv and linear weight parameters (not biases, not BN parameters), which is the standard GC prescription.

**Reasoning**: GC constrains weight updates to be mean-free, which has two effects: (1) it regularizes the weight space by preventing individual output channels from growing disproportionately, acting as an implicit L2 constraint on the mean direction, and (2) it improves the Lipschitzness of the loss landscape, making optimization more stable and efficient. At 96.46% with regularization near saturation, GC targets optimization dynamics — a fundamentally different axis than augmentation or capacity. The technique is validated on CIFAR-10 across multiple architectures (ResNet, VGG, etc.) and works with SGD. Zero throughput cost — the mean subtraction is a single tensor operation per parameter, negligible compared to the forward/backward pass.

**Sources**: Yong et al. 2020 (arxiv 2004.01461), GCSAM (arxiv 2501.11584), goal-learnings § Patterns (regularization near saturation, optimization dynamics is the remaining lever)

**Estimated Effort**: low — ~10 lines of code (add `scaler.unscale_()` call, gradient centralization loop, done)

**Risk Assessment**: GC adds a mild regularization effect. With the existing regularization stack near saturation, there's a small risk of over-regularization (similar to DropPath/CutMix failures). However, GC operates on a different axis (weight space vs. input space), and the regularization effect is mild (just mean subtraction). Worst case: no improvement or very slight regression. The GradScaler interaction requires `scaler.unscale_()` before gradient modification — well-documented pattern. Zero throughput cost.

### 2. Nesterov Momentum (Revisited)
**Summary**: Enable Nesterov momentum by adding `nesterov=True` to the existing `optim.SGD()` call. This is a single-parameter change. Nesterov momentum modifies the gradient computation to use a "look-ahead" position, providing better gradient estimates that can improve convergence speed.

**Reasoning**: EXP-004 tried Nesterov but it was bundled with label_smoothing=0.1 (weaker than current 0.2), had no AMP, no batch 256, no cosine schedule — a completely different training setup. The 4-epoch throughput cost observed in EXP-004 was likely due to the pre-AMP, smaller-batch setup where the per-step overhead was proportionally larger. With AMP and batch 256 at 16ms/step, the negligible overhead of Nesterov (just a different momentum update formula — same FLOPS) should have zero throughput impact. Nesterov is widely used in modern training recipes and provides better convergence in the final polish phase where cosine decay brings LR close to zero.

**Sources**: EXP-004 (failed in different context, count: 1), goal-learnings § Failed Approaches (Nesterov + LS=0.1 classified as low importance — approach-specific, not idea-exhausted)

**Estimated Effort**: low — single parameter change (`nesterov=True`)

**Risk Assessment**: Very safe — worst case is no improvement. The 4-epoch throughput cost from EXP-004 is unlikely to recur with AMP+batch 256. Even if there is a small throughput impact (1 epoch), Nesterov's better gradient estimates in the polish phase could compensate. The existing cosine schedule + warmup should work unchanged with Nesterov.

### 3. Alternating Flip Augmentation
**Summary**: Replace the stochastic `RandomHorizontalFlip()` in the transform pipeline with a deterministic alternating flip that flips ALL images in even epochs and applies no flip in odd epochs. Remove `transforms.RandomHorizontalFlip()` from the train transform and add `inputs = inputs.flip(-1)` conditionally in the training loop when `epoch % 2 == 0`, applied after moving data to GPU.

**Reasoning**: Random flip gives each image a 50% chance of being flipped each epoch — over 99 epochs, some images may be predominantly seen in one orientation by chance. Alternating flip guarantees equal exposure to both orientations across consecutive epochs, reducing variance in the augmentation distribution. This is used in the airbench96 recipe (96.05% on CIFAR-10). The flip is applied to GPU tensors (free — fused with data loading). Zero throughput cost.

**Sources**: airbench96 repository (https://github.com/KellerJordan/cifar10-airbench), brainstorm-024 § Candidate Ideas (previously considered but not selected)

**Estimated Effort**: low — ~5 lines of code change

**Risk Assessment**: Interaction with TrivialAugmentWide is unknown — TrivialAugmentWide already includes random geometric transforms that may partially subsume the benefit. The deterministic pattern could interact with the model's learning dynamics in unexpected ways. At 96.46%, augmentation swaps have been shown to be in the noise floor (EXP-022: +0.07pp from augmentation swap). Worst case: no improvement or negligible regression. Zero throughput cost.

## Idea Evaluation

**Evidence strength**: Gradient Centralization has the strongest evidence — peer-reviewed paper (ECCV 2020) demonstrating improved generalization on CIFAR-10 across multiple architectures including ResNet, plus follow-up work (GCSAM) confirming the benefit. Nesterov has textbook-level evidence for improved convergence but the prior experiment (EXP-004) failed in a different context. Alternating flip has indirect evidence from airbench96 but its isolated contribution is unknown (bundled with many other tricks).

**Mechanism clarity**: GC has the clearest mechanism — mean-free weight updates constrain the weight space, improving loss landscape smoothness and generalization. This targets optimization dynamics specifically, the one major untried category at this accuracy level. Nesterov's mechanism (look-ahead gradients) is well-understood but less specifically targeted to our bottleneck. Alternating flip's mechanism (variance reduction in augmentation) is modest — EXP-022 showed augmentation swaps are in the noise floor.

**Expected impact**: GC targets a fundamentally different axis than all prior improvements — weight space regularization via gradient modification. At 96.46% with input-space regularization saturated, this is the most promising untried direction. Nesterov offers marginal improvement in convergence speed. Alternating flip is likely <0.1pp given augmentation saturation evidence.

**Risk profile**: All three fail gracefully (worst case: no-improvement). GC has the most implementation complexity (but still low — ~10 lines). Nesterov is the simplest change (one parameter).

**Feasibility**: All are low effort. GC requires understanding the GradScaler interaction but the pattern is well-documented.

## Chosen Idea
**Selected**: Gradient Centralization (GC)

**Why this idea**:
Strongest evidence (peer-reviewed, validated on CIFAR-10 with ResNet), clearest mechanism (weight-space regularization via mean-free gradient updates), and targets the one major untried optimization category — gradient modification. At 96.46% with input-space regularization and augmentation saturated, optimizer-level improvements operating on a different axis (weight space vs. input space) are the most promising remaining avenue. Zero throughput cost.

**Hypothesis**:
Applying gradient centralization to conv/linear weight gradients will improve best_test_acc by 0.1-0.3pp (to 96.56-96.76%) by constraining weight updates to be mean-free, improving loss landscape smoothness and generalization without any throughput cost. The effect is orthogonal to existing input-space regularization (augmentation, RandomErasing) and output-space regularization (label smoothing), targeting a third regularization axis — weight space geometry.

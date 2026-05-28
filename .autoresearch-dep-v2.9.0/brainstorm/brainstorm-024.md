# Brainstorm EXP-024
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **airbench96 training recipe** (https://github.com/KellerJordan/cifar10-airbench)
  BN bias parameters receive 64× the base LR (`bias_scaler: 64.0`). Implementation: separate `norm_biases` (params with 'norm' in key name and requires_grad) from `other_params`, apply `lr_biases = lr * 64.0`. This is a zero-throughput-cost optimizer trick that allows BN biases to converge faster, improving calibration of feature normalization. The 64x multiplier was tuned for airbench's specific architecture (custom CNN, lr=9.0, 37 epochs) — may need adjustment for our ResNet-20 with lr=0.2 and ~99 epochs.

- **airbench96 alternating flip** (https://github.com/KellerJordan/cifar10-airbench)
  Deterministic flip of ALL images every other epoch instead of random per-image flip. Increases effective data diversity by ensuring every image is seen in both orientations across consecutive epochs, rather than stochastically (50% chance per image per epoch). Simple toggle based on epoch parity.

## Experimental History Review

- **Current best**: 96.46% (EXP-020) — cosine warmup+decay LR schedule replacing MultiStepLR, +0.55pp
- **Improvement trajectory**: 10 successful improvements across 24 experiments, from 91.72% (BASE) to 96.46%
- **Key validated recipes**: Cosine LR decay to ~0 (EXP-020), TTA horizontal flip (EXP-019), label smoothing 0.2 (EXP-015), batch 256 + LR scaling (EXP-009), AMP FP16 (EXP-005), WIDTH_MULT=4 (EXP-007), TrivialAugmentWide + RandomErasing (EXP-002), WD=5e-4 (EXP-003)
- **Exhausted approach classes**: EMA (3 variants, EXP-013/014/023), SE blocks (2 variants, EXP-011/012), regularization stacking (CutMix/Mixup/DropPath/Cutout-swap, EXP-010/017/018/022), pre-activation blocks (EXP-021), torch.compile (EXP-008), BN momentum tuning (EXP-016)
- **Key constraint**: ~99 epochs in 300s budget; any per-step overhead directly reduces epoch count, which is the binding constraint for accuracy
- **Untried gaps**: BN bias LR scaling (airbench96 recipe, zero throughput cost), deeper architecture (NUM_BLOCKS=4), alternating flip augmentation, knowledge distillation

## Candidate Ideas

### 1. BN Bias 64x LR Multiplier
**Summary**: Apply a 64× learning rate multiplier to all BatchNorm bias parameters while keeping all other parameters at the base LR. Implementation: create separate optimizer parameter groups — one for BN biases (`norm_biases`) with `lr = LR * 64.0` and adjusted weight decay, one for all other parameters with the base LR. The cosine warmup+decay schedule applies to both groups uniformly via the existing LambdaLR. No architectural changes, no new modules.

**Reasoning**: BN biases control the shift of normalized activations — they determine the operating point of each ReLU gate. At standard LR, BN biases converge slowly relative to conv weights because their gradient signal is averaged over spatial dimensions. A large LR multiplier lets BN biases quickly find their optimal operating points in early training, improving feature utilization throughout. This is validated in airbench96 (96.05% in 37 epochs on CIFAR-10) as a zero-cost trick. Our setup already has BN layers in every BasicBlock (2 per block × 9 blocks = 18 BN layers) plus 1 in the stem = 19 BN bias parameters to accelerate.

**Sources**: airbench96 repository (bias_scaler=64.0), goal-learnings § Patterns (cosine decay, label smoothing 0.2 as compatible baseline)

**Estimated Effort**: low — ~10 lines of code change (parameter group separation in optimizer construction)

**Risk Assessment**: The 64x multiplier was tuned for a very different setup (custom CNN, lr=9.0, 37 epochs, batch 1024). With our lr=0.2, the effective BN bias LR would be 12.8 — potentially too aggressive or too mild depending on our model's BN landscape. Worst case: BN biases overshoot, causing training instability or slight accuracy regression. The cosine decay should mitigate runaway instability since LR decays to ~0. Zero throughput cost means no epoch count reduction.

### 2. Alternating Flip Augmentation
**Summary**: Replace the stochastic `RandomHorizontalFlip()` with a deterministic alternating flip that flips ALL images in even epochs and applies no flip in odd epochs (or vice versa). This ensures every training image is seen in both orientations across consecutive epochs, maximizing the diversity benefit of horizontal flip augmentation. Implementation: remove `transforms.RandomHorizontalFlip()` from the transform pipeline, and in the training loop, apply `inputs = inputs.flip(-1)` when `epoch % 2 == 0` after loading the batch to GPU.

**Reasoning**: Random flip gives each image a 50% chance of being flipped each epoch — over 99 epochs, some images may be seen predominantly in one orientation by chance. Alternating flip guarantees equal exposure to both orientations, reducing variance in the augmentation distribution. This is used in airbench96's recipe. The key advantage is that it's applied to the GPU tensor batch (free — fused with data loading), and it provides more structured diversity than random sampling.

**Sources**: airbench96 repository (alternating flip implementation), EXP-019 report (TTA horizontal flip works because model learns flip-equivariant features — alternating flip reinforces this)

**Estimated Effort**: low — ~5 lines of code change (remove RandomHorizontalFlip, add conditional flip in training loop)

**Risk Assessment**: Interaction with TrivialAugmentWide is unknown — TrivialAugmentWide already includes random geometric transforms that may partially subsume the benefit. The deterministic pattern could introduce subtle periodicity in training dynamics. Worst case: no improvement or very slight regression (within noise). Zero throughput cost.

### 3. Deeper Architecture (NUM_BLOCKS=4, ResNet-26)
**Summary**: Increase `NUM_BLOCKS` from 3 to 4, creating a ResNet-26 (6×4+2=26 layers) with the same WIDTH_MULT=4. This adds one BasicBlock per stage (3 stages × 1 block = 3 additional blocks, 6 additional conv layers), increasing parameter count by ~33% from ~4.3M to ~5.7M. The additional depth provides more representational capacity and gradient flow paths.

**Reasoning**: The model may be capacity-limited at 96.46% — regularization approaches are exhausted (augmentation stack near saturation per EXP-010/017/018/022) and optimization is well-tuned (cosine schedule, label smoothing, AMP). Adding depth is orthogonal to all prior improvements. ResNet architectures were originally designed with depth as the primary capacity scaling axis. The additional parameters come at a throughput cost (~25% fewer epochs based on linear scaling of per-step time), but each epoch would have higher learning capacity.

**Sources**: EXP-007 report (WIDTH_MULT=4 gave +0.57pp, confirming capacity gains compound), goal-learnings § Patterns (throughput-to-accuracy conversion still strong), EXP-023 report Next Steps (suggested NUM_BLOCKS=4 at low-medium confidence)

**Estimated Effort**: low — single constant change (`NUM_BLOCKS = 4`)

**Risk Assessment**: ~25% throughput regression (from ~99 to ~74 epochs) may negate the capacity gain, similar to how SE blocks (EXP-011/012) failed despite per-epoch quality improvement. The pattern from prior experiments is clear: in our 300s budget, epoch count is the binding constraint. A 25-epoch reduction is a significant loss. The deeper model also increases gradient path length, potentially requiring LR adjustment. Worst case: accuracy regression from insufficient training epochs on a larger model.

## Idea Evaluation

**Evidence strength**: BN Bias 64x LR has direct validation from airbench96 (96.05% on CIFAR-10). Alternating flip is also from airbench96 but its isolated contribution is unknown (bundled with many other tricks). Deeper architecture has indirect support from EXP-007 (width scaling worked) but depth scaling has different dynamics.

**Mechanism clarity**: BN Bias 64x LR has a clear mechanism — faster BN bias convergence improves feature utilization throughout training, especially during the critical early high-LR phase. Alternating flip has a clear mechanism — guaranteed equal exposure to both orientations. Deeper architecture has a generic mechanism — more capacity — without a specific bottleneck it addresses.

**Expected impact**: BN Bias 64x LR is the most promising because it's zero-cost (no throughput hit) and targets an untouched aspect of optimization (per-parameter-group LR). At 96.46%, we need precisely targeted interventions, and optimizer tricks are the one major category we haven't explored. Alternating flip is likely in the noise floor given our existing TrivialAugmentWide + RandomErasing stack (EXP-022 showed augmentation swaps yield <0.1pp). Deeper architecture risks the same throughput trap as SE blocks.

**Risk profile**: BN Bias 64x LR fails gracefully (worst case: slight regression, no crashes). Alternating flip also fails gracefully. Deeper architecture has moderate risk of throughput-driven regression.

**Feasibility**: All three are low effort. BN Bias 64x LR requires the most code (~10 lines for param group separation) but is still straightforward.

## Chosen Idea
**Selected**: BN Bias 64x LR Multiplier

**Why this idea**:
Strongest evidence (validated in airbench96 at 96.05%), clearest mechanism (faster BN bias convergence), zero throughput cost (no epoch count reduction), and targets the one major untried optimization category — per-parameter-group learning rates. At 96.46% with regularization and augmentation near saturation, optimizer-level improvements are the most promising remaining avenue.

**Hypothesis**:
Applying a 64× LR multiplier to BN bias parameters will improve best_test_acc by 0.1-0.3pp (to 96.56-96.76%) by allowing BN biases to quickly find optimal operating points for ReLU gates, improving feature utilization throughout training without any throughput cost. The cosine decay schedule will prevent instability by smoothly reducing the effective BN bias LR to ~0 by training end.

# Brainstorm EXP-017
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Squeeze-and-Excitation Networks** (https://arxiv.org/pdf/1709.01507)
  SE blocks add channel attention via global-avg-pool → FC-reduce → ReLU → FC-expand → Sigmoid, then recalibrate feature maps. Adds <1% parameters but consistently improves accuracy across architectures. SE-ResNet outperforms vanilla ResNet on CIFAR-10 with negligible compute overhead (~0.26% FLOP increase on ResNet-50).

- **RES-SE-NET: Boosting Performance of Resnets by Enhancing Bridge-connections** (https://arxiv.org/pdf/1902.06066)
  Integrating SE blocks strategically into ResNet improves generalization on CIFAR-10 and CIFAR-100 beyond both vanilla ResNet and standard SE-ResNet placement.

- **Born Again Neural Networks** (https://arxiv.org/abs/1805.04770)
  Self-distillation (training a student with identical architecture using teacher's soft targets) outperforms the teacher. BANs based on DenseNets achieve 96.5% on CIFAR-10 (3.5% error). The technique extracts dark knowledge from inter-class relationships in soft targets.

- **Self-Knowledge Distillation through Ensemble Model Averaging (EMA-SKD)** (https://link.springer.com/article/10.1007/s00371-025-04032-2)
  Using EMA model as self-distillation teacher outperforms SOTA self-KD methods across architectures without extra parameters. Key insight: the EMA model naturally provides better-calibrated soft targets as training progresses.

- **cifar10-airbench** (https://github.com/KellerJordan/cifar10-airbench)
  Achieves 96% in 27s on A100. Key techniques: patch whitening layer, Cutout augmentation, increased random-translation strength. Shows that augmentation and input preprocessing are still potent accuracy levers.

## Experimental History Review

- **17 experiments completed** (BASE through EXP-016), accuracy improved from 91.81% → 96.39%
- **Current best**: 96.39% (EXP-016) — TTA horizontal flip added +0.66% on top of 95.73% training accuracy
- **Architecture sweet spot**: k=4 width (4.3M params, ~54 epochs in 300s). k>=6 converges too slowly (EXP-005). Deeper models (ResNet-26) also fail for the same reason (EXP-012)
- **Working recipe**: SGD+Nesterov, WD=5e-4, EMA(0.999), CutMix(α=1.0, p=0.5), label smoothing 0.1, warmup 5ep + cosine T_max=49, AMP + torch.compile, TTA (hflip)
- **Failed augmentation stacking**: TrivialAugment+CutMix too aggressive (EXP-006); Mixup worse than CutMix (EXP-015)
- **Failed regularization**: Stochastic depth (EXP-008), Dropout (EXP-011) — redundant with existing regularization or unsuitable for shallow model
- **Failed architecture changes**: Pre-activation blocks (EXP-009) — too shallow for gradient flow to be bottleneck
- **Failed optimizer**: AdamW (EXP-014) — underperforms SGD for this architecture
- **Untried gaps**: Channel attention (SE blocks), self-distillation from EMA, extended TTA beyond hflip, learning rate schedule refinement

## Candidate Ideas

### 1. Squeeze-and-Excitation (SE) Blocks
**Summary**: Add SE channel attention modules to each BasicBlock in the ResNet. After the second conv+BN, insert a squeeze-and-excitation path: global average pooling → FC(channels→channels//r) → ReLU → FC(channels//r→channels) → Sigmoid → element-wise multiply with the feature map. Use reduction ratio r=16. This recalibrates channel responses at each block, allowing the network to emphasize informative features.

**Reasoning**: SE blocks are one of the most well-validated architectural improvements for convolutional networks. The original SE paper shows consistent improvements across ResNet depths with <1% parameter overhead. For our k=4 model (channels 64/128/256), SE adds approximately 10K parameters (64²/16×2 + 128²/16×2 + 256²/16×2 ≈ 10K) on top of 4.3M — negligible. The compute overhead is similarly minimal since SE operates on 1×1 spatially-pooled features. The mechanism is well-understood: SE performs adaptive channel-wise feature recalibration, boosting important channels and suppressing noisy ones. This is an orthogonal improvement to all existing techniques (augmentation, EMA, TTA).

**Sources**: Hu et al. 2018 (https://arxiv.org/pdf/1709.01507), RES-SE-NET (https://arxiv.org/pdf/1902.06066)

**Estimated Effort**: low — add ~15 lines of code to BasicBlock, no hyperparameter search needed

**Risk Assessment**: Very low risk. SE blocks have been validated extensively. Worst case: negligible improvement (~0.0%) if the model is already effectively utilizing its channels. The small parameter increase shouldn't affect convergence within the 300s budget. Cannot cause regression since SE is a multiplicative scaling (identity-like at initialization with sigmoid centering at 0.5).

### 2. Self-Distillation from EMA Model
**Summary**: Use the existing EMA model as a teacher for online self-distillation. Add a KL divergence loss term: `total_loss = (1-α)*task_loss + α*T²*KL(student_logits/T, ema_logits/T)` where T=3.0 is the temperature and α=0.3 is the distillation weight. The EMA model's predictions provide soft targets that encode inter-class similarity structure, helping the student generalize better. Delay distillation until epoch 5 (after warmup) since the EMA model needs time to become a meaningful teacher.

**Reasoning**: Born Again Networks demonstrate that self-distillation consistently improves accuracy. We already maintain an EMA model at every step — this is essentially free additional supervision. The EMA-SKD paper specifically validates using EMA as a self-distillation teacher, showing improvements across architectures without extra parameters. The key insight: as training progresses, the EMA model accumulates a smoother loss landscape that produces better-calibrated soft targets. These soft targets expose inter-class relationships (e.g., "cat is more similar to dog than to car") that hard labels miss.

**Sources**: Born Again Neural Networks (https://arxiv.org/abs/1805.04770), EMA-SKD (https://link.springer.com/article/10.1007/s00371-025-04032-2)

**Estimated Effort**: low — add ~10 lines for the KL divergence loss computation, no architectural changes

**Risk Assessment**: Medium risk. The EMA model early in training may provide poor soft targets. Mitigated by delaying distillation until after warmup. The KL loss weight α=0.3 is conservative. Worst case: distillation loss conflicts with task loss and accuracy drops ~0.2-0.3%. The CutMix augmentation complicates distillation since the EMA model sees the same mixed input — this is fine for the teacher but means the soft targets are for the mixed image, which may dilute the benefit.

### 3. Extended Test-Time Augmentation (Multi-Shift TTA)
**Summary**: Extend the current TTA beyond horizontal flip by adding spatial shifts at test time. Average predictions over: original, horizontal flip, and 4 directional shifts (±1 pixel horizontal, ±1 pixel vertical with reflection padding). This gives 6 total views per test image. Implement by padding the input with 1-pixel reflection padding and extracting shifted 32×32 crops. Since TTA only runs during evaluation (not training), it has zero impact on the 300s training budget.

**Reasoning**: The current horizontal flip TTA already demonstrated +0.66% improvement (EXP-016), showing the model benefits from averaging over augmented views. The training augmentation includes RandomCrop(32, padding=4), so the model has learned translation invariance — shifted views should produce meaningfully different but valid predictions whose average is more robust. Each additional TTA view reduces prediction variance by averaging. At 32×32 resolution, even 1-pixel shifts create non-trivial input variation.

**Sources**: EXP-016 (TTA hflip gave +0.66%), standard TTA practice in competitive ML

**Estimated Effort**: low — modify the forward() method in eval mode, add ~10 lines

**Risk Assessment**: Low risk. Cannot hurt training accuracy. Worst case: minimal improvement over hflip-only TTA if the model's predictions are already stable to small shifts. Evaluation time increases ~3x (6 views vs 2 views), but this doesn't count against training budget. Risk of slight slowdown in total runtime but well within the 10-minute total budget.

## Idea Evaluation

**Evidence strength**: SE blocks have the strongest evidence — validated across hundreds of papers and architectures with consistent improvements and near-zero overhead. Self-distillation from EMA is well-supported by the Born Again Networks and EMA-SKD papers but less validated in our specific setting (short training, CutMix interaction). Extended TTA has moderate evidence from our own EXP-016 results.

**Mechanism clarity**: SE blocks have the clearest mechanism — adaptive channel recalibration makes each conv block more expressive without adding significant parameters. Self-distillation's mechanism (soft target dark knowledge) is well-understood but its interaction with CutMix augmented inputs is less clear. Extended TTA's mechanism is straightforward (prediction variance reduction through ensembling) but the incremental benefit over hflip diminishes with each additional view.

**Expected impact**: SE blocks target the core model capacity — making the architecture itself better. This compounds with all existing improvements (EMA, CutMix, TTA). Self-distillation could provide +0.2-0.5% but has uncertainty around the CutMix interaction. Extended TTA likely provides diminishing returns — the hflip already captured the largest variance reduction; 1px shifts may add only +0.1-0.2%.

**Risk profile**: SE blocks and Extended TTA have the safest failure modes (worst case: no improvement). Self-distillation has medium risk of mild regression if the KL loss conflicts with task loss.

**Feasibility**: All three are low effort. SE blocks require the most code changes but are still simple.

**Verdict**: SE blocks are the clear winner — strongest evidence, clearest mechanism, highest expected impact, lowest risk, and the improvement is architectural (compounding with everything else). Extended TTA is a good follow-up experiment but has lower expected incremental impact. Self-distillation is promising but carries more risk for this specific setup.

## Chosen Idea
**Selected**: Squeeze-and-Excitation (SE) Blocks

**Why this idea**:
SE blocks are the most well-validated architectural improvement available, with consistent accuracy gains across architectures and negligible overhead (<1% params, <1% FLOPs). They provide an orthogonal improvement dimension (channel attention) that hasn't been explored in our experiment history. The mechanism is clear and the risk is minimal.

**Hypothesis**:
Adding SE channel attention (reduction ratio r=16) to each BasicBlock will improve test accuracy from 96.39% to ~96.6-96.9% by enabling adaptive channel-wise feature recalibration, making each residual block more expressive without meaningful parameter or compute overhead.

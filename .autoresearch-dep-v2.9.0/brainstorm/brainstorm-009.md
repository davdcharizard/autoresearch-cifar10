# Brainstorm EXP-009
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (Yun et al., ICCV 2019)
  CutMix with alpha=1.0 adds +0.97% on CIFAR-10 (ResNet-50, 300 epochs). Effectiveness at ~83 epochs uncertain — the regularization benefit requires sufficient training iterations to manifest.

- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** (Goyal et al., 2017 — Facebook Research)
  Linear scaling rule: when multiplying batch size by k, multiply LR by k. Gradual warmup of 5 epochs stabilizes training at large batch sizes. Validated up to batch 8192 on ImageNet.

- **David Page's fast CIFAR-10 training** (myrtle.ai blog, dawn-bench)
  94-95% achievable with batch 512-1024, triangular/1cycle LR policy, label smoothing 0.2, and aggressive augmentation. Key insight: larger batch + higher LR + cyclic schedule extracts more from limited epoch budgets.

- **Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates** (Smith & Topin, 2018)
  1cycle LR policy with max_lr up to 10x default enables faster convergence at large batch sizes. The triangular schedule ramps LR up then down over training, avoiding the instability plateau seen with step decay at intermediate LR values.

- **PyTorch CutMix implementation** (torchvision reference)
  CutMix applied at batch level (not per-image in augmentation pipeline). Requires modifying the training loop to mix images and interpolate labels. Compatible with existing augmentation pipeline.

## Experimental History Review

**Current best**: 94.82% (EXP-007, WIDTH_MULT=4 + AMP + augmentation + WD=5e-4, commit 1c37a9f)

**Trajectory**: BASE 91.72 → width-2x 92.29 → +aug 92.92 → +WD 93.33 → +AMP 94.44 → width-4x 94.82 (6 improvements across 8 experiments)

**Key patterns**:
- Throughput is the binding constraint: more epochs → higher accuracy. AMP gave 1.54x throughput (+1.11pp). Width-4x maintained 83 epochs at 9ms/step with 484 MB VRAM on H20 (~96 GB available).
- Wall-clock-fractional LR schedule (drops at 0.5/0.75) is validated and near-optimal. Shifting drops earlier hurts (EXP-006).
- WD+augmentation synergy amplifies the second LR drop contribution.
- AMP is unstable at LR=0.01 but the LR=0.001 phase is where gains materialize.
- torch.compile provides zero benefit on H20 for this model size (EXP-008).

**Untried approaches**:
- Batch size increase (massive VRAM headroom: 484 MB / ~96 GB)
- CutMix / Mixup batch-level augmentation
- Cyclic/1cycle LR schedules (cosine was tried with wrong T_max in EXP-000, but 1cycle is different)
- Gradient accumulation for effective larger batch
- Architecture changes (deeper network, squeeze-excite blocks)

## Candidate Ideas

### 1. Batch Size 256 with Linear LR Scaling
**Summary**: Double batch size from 128 to 256 and scale LR from 0.1 to 0.2 following the linear scaling rule. Add a 5-epoch LR warmup for stability. This directly increases GPU utilization — at 484 MB VRAM, we are using ~0.5% of the H20's capacity, so a 2x batch size increase is trivially within resource limits.

**Reasoning**: Throughput (epochs per 300s) is the binding constraint for accuracy. Every successful accuracy improvement in this project's history has been driven by either more capacity (width) or more throughput (AMP). At 9ms/step with batch 128, doubling batch size should yield ~30-50% throughput increase (reduced per-sample overhead, better GPU occupancy), potentially reaching ~108-120 epochs vs current 83. The linear scaling rule (Goyal et al.) is well-validated and the fast CIFAR-10 literature confirms batch 512-1024 works well for this task. The wall-clock-fractional LR schedule automatically adapts to the new epoch count.

**Sources**: Goyal et al. 2017 (linear scaling rule), David Page fast CIFAR-10 (batch 512-1024 effective), EXP-005 (AMP throughput → accuracy), EXP-007 (current baseline profile).

**Estimated Effort**: Low — change BATCH_SIZE, LR, add ~5 lines for warmup in the LR lambda.

**Risk Assessment**: Low risk. The linear scaling rule is extensively validated. Warmup prevents early divergence. Failure mode is benign (no-improvement if throughput gain is smaller than expected or if larger batch hurts generalization). VRAM is not a concern (even 4x batch would fit). The AMP instability at LR=0.01 might be slightly worse at LR=0.02, but the wall-clock schedule spends only 25% of training at this level.

### 2. CutMix Batch-Level Augmentation
**Summary**: Add CutMix (alpha=1.0) applied at the batch level in the training loop. CutMix cuts a rectangular patch from one image and pastes it onto another, with labels interpolated proportionally. This provides stronger regularization than pixel-level augmentation alone.

**Reasoning**: CutMix adds +0.97% on CIFAR-10 in the published benchmark (ResNet-50, 300 epochs). Our model at 83 epochs may not have enough training iterations to fully exploit CutMix's regularization benefit. However, CutMix is complementary to existing augmentation (TrivialAugmentWide operates per-image; CutMix operates per-batch) and has near-zero throughput cost since it's a simple tensor operation on GPU. The risk is that at 83 epochs, the regularization may be too strong, effectively slowing convergence similar to the Nesterov+label_smoothing failure (EXP-004).

**Sources**: Yun et al. ICCV 2019 (CutMix paper, +0.97%), EXP-004 (label smoothing over-regularization precedent), torchvision reference implementation.

**Estimated Effort**: Low-medium — add CutMix function in train.py, modify the training loop to apply it before forward pass, adjust loss to use soft labels.

**Risk Assessment**: Medium risk. The +0.97% published result is at 300 epochs; at 83 epochs, CutMix could over-regularize and slow convergence (similar to EXP-004 where label smoothing hurt). CutMix also interacts with RandomErasing — both modify spatial content, potentially creating overly corrupted training samples. Soft-label CrossEntropy might add slight per-step overhead.

### 3. Batch Size 512 with LR 0.4 and Warmup
**Summary**: Quadruple batch size from 128 to 512 with LR scaled to 0.4 (linear scaling rule) and a 5-epoch gradual warmup. More aggressive than Idea 1, targeting maximum throughput gain from the massive VRAM headroom.

**Reasoning**: If 2x batch works, 4x should deliver even more throughput. At 484 MB baseline VRAM, even 4x batch should stay well within the H20's ~96 GB capacity (estimated ~1.5-2 GB). The fast CIFAR-10 literature uses batch 512-1024 successfully. The larger batch reduces per-epoch wall-clock time proportionally more than 2x, potentially reaching 130-160 epochs. However, the generalization gap at batch 512 is more pronounced, and LR=0.4 with AMP may cause instability — the existing AMP instability at LR=0.01 (EXP-005) suggests FP16 is sensitive to high LR values.

**Sources**: Goyal et al. 2017, Smith & Topin 2018 (super-convergence at large batches), David Page fast CIFAR-10, EXP-005 (AMP instability at intermediate LR).

**Estimated Effort**: Low — same changes as Idea 1 but with different values.

**Risk Assessment**: Medium-high risk. LR=0.4 with FP16 autocast may cause gradient overflow early in training. The generalization gap at batch 512 is well-documented and may require additional regularization to overcome. If LR=0.4 diverges, the experiment produces no useful result (crash rather than no-improvement). The 5-epoch warmup helps but may not be sufficient for FP16 stability.

## Idea Evaluation

**Evidence strength**: Idea 1 (batch 256 + LR 0.2) has the strongest evidence — the linear scaling rule is the most replicated result in deep learning optimization, and 2x batch is conservative enough to be within the well-validated regime. Idea 2 (CutMix) has strong published results but at 300 epochs, making extrapolation to 83 epochs uncertain. Idea 3 (batch 512) has supporting literature but pushes into a regime where FP16 stability and generalization gap become real concerns.

**Mechanism clarity**: Idea 1 has the clearest mechanism — more images per second → more epochs → more accuracy, following the same throughput → accuracy pattern that drove EXP-005 (+1.11pp from AMP throughput). Idea 2's mechanism is regularization-based and its interaction with the existing augmentation pipeline at low epoch counts is unclear. Idea 3 shares Idea 1's mechanism but with compounding risks.

**Expected impact**: Idea 1 should yield ~30-50% more epochs (108-120 vs 83). Given that AMP's 1.54x throughput increase yielded +1.11pp, a ~1.3-1.5x increase should yield +0.3-0.5pp — enough to clear the 94.92% threshold. Idea 3 has higher potential upside (~1.5-2x epochs) but higher variance. Idea 2's impact is uncertain at 83 epochs.

**Risk profile**: Idea 1 has the safest failure mode — if batch 256 doesn't help, we get a clean no-improvement result. Idea 3 risks divergence (crash). Idea 2 risks over-regularization with no clear diagnostic signal.

**Feasibility**: All three are low effort to implement. Idea 1 and 3 are nearly identical in implementation.

**Decision**: Idea 1 (batch 256 + LR 0.2) is the clear winner — highest evidence strength, clearest mechanism, safest risk profile, and sufficient expected impact. Idea 3 (batch 512) is a natural follow-up if Idea 1 succeeds. Idea 2 (CutMix) is worth trying later, potentially combined with larger batch size to increase epoch count first.

## Chosen Idea
**Selected**: Batch Size 256 with Linear LR Scaling

**Why this idea**:
Batch size 256 + LR 0.2 has the strongest evidence base (linear scaling rule is one of the most replicated results in DL), the clearest causal mechanism (throughput → epochs → accuracy, validated by EXP-005's AMP success), the safest risk profile, and sufficient expected impact to clear the 94.92% verification threshold. It directly addresses the identified bottleneck (GPU underutilization at 484 MB / ~96 GB VRAM) with a conservative 2x increase.

**Hypothesis**:
Doubling batch size from 128 to 256 with LR scaled to 0.2 and 5-epoch warmup will increase throughput by ~30-50%, yielding ~108-120 epochs in the 300s budget. The additional training iterations will push accuracy above 94.92%, following the throughput → accuracy pattern established by AMP in EXP-005. The wall-clock-fractional LR schedule will automatically adapt. We expect best_test_acc ≥ 95.0%.

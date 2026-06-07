# Brainstorm EXP-000
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **94% on CIFAR-10 in 3.29 Seconds on a Single GPU** (https://arxiv.org/html/2404.00498v2)
  Keller Jordan et al. achieve 94% in 3.29s using a custom ConvNet with whitening layers, Muon optimizer, derandomized flipping, and CutOut. The 96% variant adds residual connections, wider channels (128/512), and 12px CutOut over 40 epochs. Key: architecture-aware augmentation and optimizer co-design matters more than raw model size.

- **cifar10-airbench repository** (https://github.com/KellerJordan/cifar10-airbench)
  96% accuracy in 27 seconds on A100. Techniques: patch whitening, Muon optimizer, CutOut augmentation, wider convolutions. Shows that CIFAR-10 accuracy ceiling is well above 94% even with tiny compute budgets.

- **OpenMixup CIFAR-10/100 Benchmarks** (https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_cifar.html)
  Comprehensive benchmarks showing CutMix (alpha=0.8) and label smoothing (0.1) consistently improve ResNet accuracy on CIFAR-10 by 0.5-1.5%. Cosine annealing outperforms step LR across all settings.

- **DeVries & Taylor 2017 — Cutout** (referenced in search results)
  CutOut augmentation achieves 96.01% on CIFAR-10 with WideResNet. For smaller models like ResNet-20, CutOut with 16px patches typically adds 0.5-1.0% accuracy.

- **PyTorch Lightning CIFAR-10 baseline** (https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
  Demonstrates OneCycleLR scheduling reaching ~94% with a standard ResNet. Confirms cosine/one-cycle schedules are standard modern practice.

## Experimental History Review

First experiment under this goal — no prior history.

**Baseline analysis (from codebase)**:
- ResNet-20: 269,722 params, {16, 32, 64} channel widths
- 94 epochs / 36,551 steps in 300s (~3.2s/epoch)
- SGD lr=0.1, step LR at milestones [32000, 48000], MAX_STEPS=64000
- **Critical finding**: The step LR schedule's second milestone (48k) is NEVER reached — only 36,551 steps complete in 300s. The schedule was designed for 64k steps but the time budget cuts it short, meaning training never benefits from the final low-LR fine-tuning phase.
- Augmentation: only RandomCrop(32, pad=4) + RandomHorizontalFlip — no modern augmentation
- No regularization beyond weight_decay=1e-4

## Candidate Ideas

### 1. Modern Training Recipe (Cosine LR + CutOut + Label Smoothing)
**Summary**: Replace the step LR schedule (which is misaligned with the actual 300s training duration) with cosine annealing that naturally spans the full training duration. Add CutOut augmentation (16px patches) which is proven to improve CIFAR-10 accuracy by 0.5-1.0% on small models. Add label smoothing (0.1) for better generalization. Add 5-epoch linear warmup for training stability. Keep the ResNet-20 architecture unchanged.

**Reasoning**: The baseline's step LR schedule never reaches its second milestone at 48k steps (only ~36.5k steps complete). Cosine annealing eliminates this issue by smoothly decaying LR over the actual training duration. CutOut is the single most impactful augmentation for CIFAR-10 on small models (DeVries & Taylor 2017). Label smoothing adds consistent 0.2-0.5% gains. Together these are the highest-evidence, lowest-risk improvements.

**Sources**: OpenMixup benchmarks, DeVries & Taylor 2017, PyTorch Lightning baseline, codebase analysis of train.py LR schedule misalignment

**Estimated Effort**: low

**Risk Assessment**: Very low risk. All three techniques are individually well-validated on CIFAR-10 with ResNets. Cosine annealing is a drop-in replacement for step LR. CutOut is a simple augmentation transform. Label smoothing is a one-line change to the loss function. Worst case: marginal improvement rather than large gains.

### 2. Wider ResNet + Modern Recipe
**Summary**: Increase the ResNet channel widths from {16, 32, 64} to {64, 128, 256} (a 4x width multiplier, creating a "WideResNet-20-4" style model) combined with all modern recipe changes from Idea 1 (cosine LR, CutOut, label smoothing). The wider model has much more representational capacity to learn features, and with 98GB VRAM the memory is not a constraint.

**Reasoning**: The baseline ResNet-20 has only 270K params — extremely small by modern standards. Wider models consistently achieve higher accuracy on CIFAR-10. The airbench 96% model uses 128/512 channels. With ~300s of compute and an H20 GPU, a wider model should still complete enough epochs to converge. The risk is that throughput drops (fewer epochs in 300s), but the per-epoch improvement from higher capacity should more than compensate.

**Sources**: cifar10-airbench architecture (128/512 channels for 96%), OpenMixup benchmarks (WideResNet results), He et al. 2015 noting wider variants improve accuracy

**Estimated Effort**: medium

**Risk Assessment**: Medium risk. The wider model may train significantly fewer epochs in 300s due to increased compute per step. If the model doesn't converge in fewer epochs, accuracy could actually decrease. Need to balance width increase with training time.

### 3. Architecture Overhaul: ConvNeXt-style + Aggressive Augmentation
**Summary**: Replace ResNet-20 with a modern ConvNeXt-style architecture adapted for CIFAR-10: depthwise separable convolutions, GELU activations, LayerNorm instead of BatchNorm, inverted bottlenecks. Combine with aggressive augmentation (CutMix + Mixup + CutOut) and cosine schedule. This targets 95%+ by using a fundamentally more efficient architecture.

**Reasoning**: Modern architectures like ConvNeXt extract more useful features per FLOP than classic ResNets. The airbench results show custom architectures outperform ResNets given the same compute budget. Depthwise separable convolutions are more parameter-efficient, allowing a larger effective model within the same compute envelope.

**Sources**: cifar10-airbench paper (custom architectures key to speedrun results), ConvNeXt paper (Liu et al. 2022)

**Estimated Effort**: high

**Risk Assessment**: High risk for a first experiment. Complete architecture rewrite with many moving parts — if any component is misconfigured, the model may not converge at all. Multiple simultaneous changes make it hard to diagnose failures. Better suited for a later experiment after simpler improvements are established.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest individual evidence — cosine annealing, CutOut, and label smoothing are each proven on CIFAR-10 with ResNets in published benchmarks. Idea 2 combines proven width scaling with the modern recipe but introduces the throughput-vs-capacity trade-off which is harder to predict a priori. Idea 3 relies on architectural analogies from different settings (ConvNeXt was designed for ImageNet scale).

**Mechanism clarity**: Idea 1 has the clearest mechanism — fixing the misaligned LR schedule directly addresses a known training deficiency, and CutOut/label smoothing have well-understood regularization effects. Idea 2's mechanism (more capacity) is clear but the net effect depends on the epoch-count reduction. Idea 3 has the least clear mechanism for this specific setting.

**Expected impact**: Idea 1 should yield 1-3% improvement (92.5-94.5%). Idea 2 could potentially reach higher (94-95%+) but with more uncertainty. Idea 3 has the highest ceiling but also the highest variance.

**Risk profile**: Idea 1 fails gracefully (worst case: marginal improvement). Idea 2 could fail if too few epochs complete. Idea 3 could crash or produce invalid results.

**Strategy**: As the first experiment, Idea 1 establishes a solid improved baseline. Future experiments can layer on model capacity changes (Idea 2) and architecture changes (Idea 3) on top of the proven modern recipe.

## Chosen Idea
**Selected**: Modern Training Recipe (Cosine LR + CutOut + Label Smoothing)

**Why this idea**:
It has the strongest evidence base, the clearest causal mechanism (fixing the misaligned step LR schedule alone is a concrete deficiency), the lowest risk, and it establishes the modern training recipe that all future experiments can build upon. Each component (cosine LR, CutOut, label smoothing) is individually proven on CIFAR-10 with small ResNets.

**Hypothesis**:
Replacing step LR with cosine annealing (fixing the schedule misalignment), adding 16px CutOut augmentation, and adding label smoothing (0.1) will improve best_test_acc from the 91.81% baseline to approximately 93-94%, primarily driven by proper LR scheduling over the full training duration and CutOut's regularization effect.

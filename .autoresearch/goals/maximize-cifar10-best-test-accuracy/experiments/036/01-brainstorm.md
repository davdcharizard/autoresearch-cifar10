# Brainstorm EXP-036
**Created**: 2026-08-06

## Web Search & Literature Review

- **Torchvision RandomCrop documentation and installed 0.24.1 source** (https://docs.pytorch.org/vision/0.24/generated/torchvision.transforms.RandomCrop.html; distilled in `experiments/035/proposals/idea-03.md`)
  Constant padding is the default; reflection mirrors without repeating the edge value. The installed PIL path converts through NumPy for reflection, so semantics are supported but loader cost must be measured.
- **Deep Residual Learning for Image Recognition** (https://arxiv.org/abs/1512.03385)
  The original CIFAR recipe establishes four-pixel pad-and-crop geometry but does not establish a superior padding mode; EXP036 preserves the geometry and isolates the border prior.
- **PyTorch Channels Last Memory Format** (`knowledge/references/pytorch-channels-last.md`; https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  Channels-last preserves logical NCHW while potentially changing CUDA Conv/BN kernels, but tiny FP32 CIFAR speedup and its accuracy value through extra exposure remain unproven.

## Experimental History Review

- EXP010 remains the 94.15% frontier. Its width-2 model, 80% N1/M7 plus p0.5 CutMix plateau, ordinary momentum/decay, and simultaneous weak/LR transition are protected; stronger or earlier data-policy changes repeatedly suppressed fit.
- EXP035's all-site SiLU stayed concentration-free and globally bounded but remained unscored because non-specific ratio gates failed. The resulting protocol rule is load-bearing here: denominator-safe metrics must first qualify on controls, and a data candidate needs paired counterfactual inputs rather than reusing constant-padded tensors.
- Reflection padding was a fully developed deferred finalist in EXP035. It differs from failed Cutout/Random Erasing because it deletes no interior source pixels, differs from Mixup/CutMix changes because targets and regional composition stay accepted, and changes no model/optimizer path.
- Channels-last remains the only direct attack on the 75.46% backward bottleneck, but two prior reviews judged the accuracy link through a few percent more exposure weak. Explicit classifier-bias symmetry is cheaper but likely below the ten-image threshold.

## Objective Limiter Diagnosis

The frontier needs better generalization without deepening the protected short strong-phase underfit. Reflection padding targets view quality: with four-pixel padding and uniform 32-of-40 cropping, only one of 81 offset pairs avoids padding and about 13.41% of output area is padded in expectation. Constant raw zeros become a fixed negative-color border after mean subtraction and reveal crop displacement; reflection preserves local texture but may duplicate edge objects. The hypothesis is therefore meaningful but low-margin. Unlike another model-path change, its principal systems risk is measurable worker-side PIL/NumPy overhead, while GPU computation remains unchanged.

## Collected Ideas

## Combinations

## Candidate Ideas

### Explicitly Zero the Final Classifier Bias
**Summary**: Preserve the accepted Linear constructor and every RNG draw, then explicitly zero only its ten-element bias after accepted initialization. Keep the bias trainable under ordinary SGD/decay, so the intervention removes random initial class-prior offsets without changing parameter count, graph, or recurring operations.

**What it targets**: Early class symmetry in a balanced ten-class problem, avoiding another hidden-feature or optimizer reparameterization.

**Reasoning**: The change is denominator-safe, zero-cost, and narrowly addresses class offset rather than feature scale. It may reduce early one-class asymmetry seen even in accepted controls. However, the random bias is tiny relative to BN-pooled feature logits, SGD can relearn offsets immediately, and no local evidence suggests ten bias values limit terminal generalization. Its plausible effect is below the required ten examples.

**Sources**: accepted `train.py` Linear construction; EXP014/020/024/028/034/035 class-geometry evidence.

**Estimated Effort**: low.

**Risk Assessment**: Low implementation risk but high null-effect risk; zeroing after construction must preserve post-construction RNG and all other tensors exactly.

### Reflection-Padded Strong and Weak Crops
**Summary**: Add `padding_mode="reflect"` to both accepted `RandomCrop(32, padding=4)` transforms, preserving order, flip, N1/M7, CutMix, the 80% switch, model, optimizer, schedule, seed, timer, and evaluator. Preflight pairs source indices and per-sample RNG states so crop/flip/RandAugment/CutMix decisions and targets align while constant/reflection pixels intentionally differ. Full prior specification: `experiments/035/proposals/idea-03.md`.

**What it targets**: Boundary-view quality and crop-position shortcut removal without deleting class evidence or adding strong-phase regularization operations.

**Reasoning**: Constant padding affects roughly one eighth of most crops and introduces a fixed normalized border; reflection retains texture and may better approximate unpadded evaluation statistics. The two-keyword production diff preserves every validated recipe component and no GPU operator changes. Direct accuracy evidence is weak, reflection can duplicate semantic fragments, and the PIL/NumPy path must retain loader queue margin. A new paired corpus is mandatory because prior post-transform constant tensors cannot express the intervention.

**Sources**: torchvision RandomCrop docs/source; original ResNet paper; EXP005/006/011/026/027/033/035; `experiments/035/proposals/idea-03.md`.

**Estimated Effort**: medium; implementation is trivial but paired stochastic-data and loader-throughput evidence are not.

**Risk Assessment**: Medium. Expected effect is near the 0.10-point gate, mirrored borders may be an equally artificial prior, and reflection could expose worker stalls despite current prefetch headroom.

### End-to-End FP32 Channels-Last Training
**Summary**: Convert the ordinarily initialized model to channels-last and transfer every 4-D input with the same memory format, while preserving logical NCHW values, FP32/default-TF32, data, batch size, optimizer, schedule, timer, and evaluator semantics. Require stride/profiler proof and seven fresh paired full-step trials before production.

**What it targets**: The measured systems bottleneck—Conv/BN backward is 75.46% of counted time—through kernel/layout selection and potentially more same-recipe exposure.

**Reasoning**: Official PyTorch support makes the semantic mechanism credible, and no data or optimization policy changes. Yet 32x32 FP32 kernels may gain nothing, Option-A slicing/padding can trigger conversions, and local experiments have not shown that roughly 1-3% more updates improve this frontier. Evaluation layout normalization also expands the diff and max-metric opportunity count needs a cap.

**Sources**: `knowledge/references/pytorch-channels-last.md`; `02-system-understanding.md`; EXP013/023/029/034 brainstorm reviews.

**Estimated Effort**: high.

**Risk Assessment**: High measurement risk and medium accuracy risk; the likely result is a timing veto or a valid but flat run.

## Review

Claude's independent review (`01-idea-review.md`) selected **Reflection-Padded Strong and Weak Crops**, scoring evidence/reasoning 6.5/10 and potential impact 5.5/10. It rejected classifier-bias zeroing as a likely sub-threshold null and channels-last because both a tiny-FP32 speedup and an exposure-to-accuracy link are unproven. Its main reflection concern is honest effect size: the prior 94.27 point prediction sits only two examples above the formal gate, while reflection can duplicate edge objects or remove useful missing-context regularization.

I adopt the pick and the review's preflight correction. EXP036 will still prove paired RNG/target/crop semantics and run a bounded real-input candidate-specific safety screen because EXP033 showed that a data-only perturbation can change early class geometry. It will not reuse EXP035's per-site or zero-denominator relative gates. Every numerical threshold must first pass accepted controls, while paired strong/weak loader throughput and non-rollover waits are the load-bearing implementation gates. Switch fit and NLL remain explanatory and cannot alter the formal metric verdict.

## Idea Evaluation

- **Reflection-padded crops** — Advance. It directly tests view quality without deleting pixels, changing targets, or altering the model/optimizer, and its host cost is measurable before scoring.
- **Channels-last** — Defer. It targets the systems bottleneck but still lacks shape-specific speed evidence and a causal accuracy mechanism for a few percent more exposure.
- **Zero classifier bias** — Reject. Its ten-value initial symmetry effect is unlikely to survive ordinary SGD or clear the ten-example threshold.

## Chosen Idea
**Selected**: Reflection-Padded Strong and Weak Crops

**Why this idea**:
Reflection changes the crop boundary prior while preserving every accepted augmentation, label, schedule, model, and optimizer choice. It is mechanistically distinct from failed deletion/mixing experiments and introduces no recurring GPU operation. The evidence is low-margin, but the question is clean: whether removing a fixed negative-color crop-position cue and retaining edge texture improves terminal generalization enough to clear 94.25%.

**Hypothesis**:
Adding `padding_mode="reflect"` to both four-pixel RandomCrop transforms will preserve aligned crop/flip/RandAugment/CutMix decisions and targets, sustain at least 95% paired loader throughput with no starvation, retain at least 99% optimizer exposure, avoid candidate-only early class concentration, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. One valid miss or candidate-specific safety/throughput veto retires this exact two-phase reflection policy without padding-mode, width, phase, or seed tuning.

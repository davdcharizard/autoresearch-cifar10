# Brainstorm EXP-041
**Created**: 2026-07-27

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the compute-effective accepted spatial backbone; new capacity should remain after pooling because spatial backward is the measured cost bottleneck.
- **RandAugment** (`knowledge/papers/randaugment.md`): the accepted one-operation early policy supplies cheap global input invariance, so any new augmentation must target a distinct invariance and preserve the clean tail.
- **mixup** (`knowledge/papers/mixup.md`): input/target interpolation is the strongest accepted regularizer and should remain exact; auxiliary objectives must reuse its one scalar and target pair.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early-only regularization can preserve late convergence, supporting a bounded early occlusion proposal but not proving that additive regularization will help this already calibrated stack.

No network, remote source, or new retrieval was used. This thorough pass is offline and uses only the persistent knowledge base, accepted source, measured system understanding, and 40 completed experiment records.

## Experimental History Review

- EXP036 remains the 94.48% frontier with 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Its bias-free scale-0.1 `128 -> 64 -> 128` residual MLP established cheap post-pooling nonlinear remapping as the only recent mechanism that improved both top-1 and loss.
- EXP037/038 locally bracket terminal classifier decay, EXP039 rejects 39.46% more hard-tail LR area, and EXP040 rejects exact equal-row classifier radii. Preserve the accepted classifier freedom, decay, and global cosine.
- The learner nearly interpolates its hard-label tail. Generalization and boundary quality, rather than input delivery, memory, or wall time, limit the metric; about 98% of counted-step work is model forward/backward and spatial stages dominate. New work should add a representation or invariance constraint at near-zero spatial cost and retain at least 127 passes.
- Additive masking has poor local evidence: early residual dropout and drop-path lost 0.55 and 0.41 points, and shared-rectangle CutMix lost 0.35. A small label-preserving input hole remains mechanically distinct but should be considered low confidence.
- Immediate width/scale/activation tuning of the successful pooled head was deliberately deprioritized after EXP036. A learned gain is distinguishable from a fixed sweep because it preserves the initial accepted function, but there is no diagnosis that 0.1 is miscalibrated and the second matrix can already change effective amplitude.
- Direct-path auxiliary supervision is untested. It can preserve linear usefulness of the dominant raw pooled feature without inference changes, but it also weakens accepted main-head supervision and forces one classifier to serve two feature geometries.
- The frontier is not saturated: 94.48% is far from known CIFAR-10 ceilings, and EXP027/036 produced material gains through interactions after many standalone misses. The evidence supports a narrow orthogonal search, not stopping.

## Collected Ideas

- **Learn only the pooled-residual gain** - replace the accepted Python `0.1` by one zero-decay scalar parameter initialized exactly to FP32 0.1. This preserves the initial function, common state, data, and spatial compute while allowing training to allocate direct and nonlinear paths over time. It is cheap and clean, but functionally redundant with scaling the decayed second head matrix and lacks evidence that the fixed gain is limiting.
- **Training-only direct-path auxiliary CE** - classify both the accepted refined pooled feature and the raw pooled feature through the same classifier, optimizing a fixed 90/10 convex blend anchored to the existing 0.1 branch scale. Evaluation stays bitwise accepted. This targets representation robustness after pooling, but changes backbone/classifier gradients and reduces the useful head's supervision to 90%.
- **Early post-mixup 8x8 Cutout** - erase one per-example mean-color 8x8 square only in the existing early mixup branch with a private CUDA generator, retaining RandAugment and the full clean tail. It imports localized missing-evidence invariance rather than global transformations, but stacks a third regularizer against strong local masking failures.
- **Cached-feature head refinement step** - reuse detached pooled features for a second head/classifier SGD step, attacking the measured spatial-backward bottleneck. It is rejected because the accepted learner already nearly interpolates and a clean implementation either creates competing momentum owners or doubles head decay/update ordering, obscuring attribution (`proposals/idea-04.md`).
- **One-time Nesterov reset at the hard-label boundary** - zero live momentum buffers immediately before the first hard-label update while preserving the accepted global cosine. It is parameter-free and isolates transition state, but inherited velocity decays below 1% in only about 44 updates, giving little direct exposure and no positive local evidence.
- **Bias-free classifier simplification** - remove ten final biases while keeping all weights and construction RNG exact. Balanced classes and final normalization make class offsets potentially redundant, but EXP040 warns against imposing classifier symmetry and the likely effect is below the 0.10-point margin.
- **Gradient centralization after backward** - subtract each convolution/linear gradient's non-output-axis mean before accepted Nesterov. This zero-inference-cost optimization geometry could regularize filters, but no persistent source diagnoses mean-gradient drift, reduction kernels consume counted time, and interaction with coupled decay is underdetermined.
- **Pooled manifold-mixup moonshot** - move or add interpolation at the raw/refined pooled feature rather than only at input pixels. It could regularize exactly the successful low-dimensional representation, but replacing accepted input mixup discards the strongest causal gain while stacking both creates ambiguous compound labels and an unsupported coefficient.

## Combinations

- **Direct-path auxiliary CE + learned gain**: the auxiliary objective could keep raw pooled features linearly useful while the learned scalar allocates nonlinear correction strength. This is more adaptive than either alone, but it creates two coupled degrees of freedom and makes a miss uninterpretable; isolate them first.
- **Early Cutout + accepted pooled head**: localized missing-evidence invariance may regularize the extra post-pooling capacity, echoing EXP027's capacity-plus-invariance interaction. It is stronger than generic Cutout because the capacity already exists, but unlike EXP027 both accepted regularizers are already active and local masking evidence is negative.
- **Direct-path auxiliary CE + boundary momentum reset**: clearing mixed-target velocity could make the new dual-path hard-label objective start cleanly. The auxiliary remains always-on and the reset is transient, but the combination cannot identify whether representation supervision or state discontinuity matters and has no evidence advantage over isolation.

## Candidate Ideas

### Training-Only Direct-Path Auxiliary Cross-Entropy
**Summary**: During training only, compute accepted main logits from `fc(z + 0.1*h(z))` and auxiliary logits from the shared `fc(z)`, then optimize a fixed 90/10 convex CE blend for both the same mixup targets and hard targets. The default/evaluator forward stays accepted, parameters and RNG stay exact, and evaluation has no auxiliary work.

**What it targets**: The generalization and boundary-quality gap after the successful pooled remapping, encouraging the dominant raw pooled feature to remain independently linearly useful without adding spatial compute or inference parameters.

**Reasoning**: EXP036's gain shows nonlinear pooled refinement matters, while its residual form leaves a strong direct representation with no independent objective. A weak shared-classifier auxiliary can act as deep supervision at negligible FLOPs and stays orthogonal to failed classifier-radius/decay changes. The 90/10 convex weight reuses the existing structural scale and avoids increasing nominal CE scale, but it is still heuristic and may suppress the accepted head or create gradient conflict. Full contract: `proposals/idea-02.md`.

**Sources**: `experiments/036/04-analysis.md`; `experiments/040/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: The raw and refined features may prefer different class boundaries; sharing `fc` can compromise both, while the main head receives only 90% of accepted data gradient. A miss closes the exact always-on shared-classifier 90/10 objective, not all auxiliary supervision.

### Early 8x8 Cutout After Mixup
**Summary**: In every early mixup batch, erase one independently located 8x8 square per mixed image with normalized zero fill using a private device generator. Preserve accepted targets, RandAugment, parameter state, and hard tail exactly; disable Cutout with mixup at 65%.

**What it targets**: The remaining generalization gap through local missing-evidence invariance, a spatially localized input constraint distinct from RandAugment's global operations and mixup's dense interpolation, with no model or inference cost.

**Reasoning**: The persistent Cutout idea is established for CIFAR-scale regularization and early-only removal aligns with `time-matters-regularization.md`. The exact 6.25% hole is deliberately milder than EXP003's mean 31% CutMix region. Local evidence is still mostly negative: CutMix and residual masking failed, so this is a restrained one-score interaction rather than a parameter family. Full contract: `proposals/idea-03.md`.

**Sources**: `knowledge/papers/time-matters-regularization.md`; `knowledge/papers/randaugment.md`; `experiments/003/04-analysis.md`; `experiments/006/04-analysis.md`; `experiments/030/04-analysis.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Stacking three early regularizers may weaken the useful depth/RandAugment interaction, and a mean-color square can become a synthetic cue. A miss closes adjacent additive early Cutout/erasing rescues on this accepted stack.

### Learn Only the Accepted Pooled-Residual Gain
**Summary**: Register one scalar `pooled_head_scale` initialized exactly to the accepted FP32 value 0.1 and use it in `out + scale * pooled_head(out)`. It enters the existing zero-decay group at the accepted LR/momentum and adds one parameter, no RNG draw, and negligible compute. Every accepted tensor and the initial function remain exact.

**What it targets**: The generalization/boundary-quality limiter identified in `02-system-understanding.md`, specifically whether the relative utility of direct pooled features and the successful nonlinear correction changes over the early regularized and clean-tail phases.

**Reasoning**: EXP036 proves the correction is useful at one fixed amplitude and reports an initial branch/direct norm ratio of 0.120864. Learnability tests temporal allocation without a fixed-value sweep or inference expansion. Counterevidence is material: the second head matrix already changes amplitude, the scalar changes conditioning/decay factorization rather than function capacity, and EXP036 gave no miscalibration diagnosis. Full contract: `proposals/idea-01.md`.

**Sources**: `experiments/036/04-analysis.md`; `experiments/037/04-analysis.md`; `experiments/040/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: Full-LR Nesterov may move a one-dimensional zero-decay gain too aggressively or simply reallocate norm from the decayed second matrix. A miss closes the direct unconstrained learned scalar and immediate parameterization/LR/decay rescues.

## Review

The offline adversarial reviewer selected training-only direct-path auxiliary CE at 3/5 evidence and 3/5 impact. I adopted its central corrections: accepted main CE already supervises raw `z` through the identity residual path, so the candidate tests a different coupled gradient objective rather than restoring absent supervision; multiplying main CE by 0.9 also reduces pooled-head data gradients while leaving its decay unchanged, raising its decay/data ratio by 11.1%; and success cannot isolate direct-path shaping from shared-classifier conflict, backbone-gradient changes, or head suppression. These confounds remain part of the exact treatment rather than being "repaired" with another coefficient, loss rescale, or decay exception. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. The direct-path objective is the only finalist that introduces a distinct representation-training signal after the measured spatial bottleneck while preserving inference, state, data, and accepted classifier geometry. Learned gain mostly reparameterizes amplitude through a new zero-decay route with no diagnosis; Cutout conflicts with three local masking misses; cached refinement is not score-worthy because optimizer ownership and decay exposure cannot be isolated.

## Chosen Idea
**Selected**: Training-Only Direct-Path Auxiliary Cross-Entropy

**Why this idea**:
During training, optimize the exact always-on objective `0.9 * CE(fc(z + 0.1*h(z))) + 0.1 * CE(fc(z))`, reusing the same accepted mixup target pair early and hard targets late. Default inference remains byte-equivalent to accepted, no parameters or RNG are added, and the extra work occurs only after pooling. The test is exploratory but causally bounded: it asks whether a weak shared-boundary objective on raw and refined representations improves generalization despite known head-gradient suppression and possible classifier conflict.

**Hypothesis**:
If the raw and refined pooled representations admit sufficiently compatible class boundaries, the exact shared-classifier 90/10 objective will add a nonredundant boundary-shaping gradient without materially suppressing the accepted residual head, retain at least 127 projected and realized passes, and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final accuracy at least 94.45% and loss at most 0.2456 are corroboration only. A valid normal-exposure miss closes this exact always-on objective and immediate coefficient/cutoff/detach/separate-head/distillation/head-scale rescues, not independently motivated intermediate supervision or another loss family.

# Brainstorm EXP-014
**Created**: 2026-08-06

## Web Search & Literature Review

- **Generalizing Pooling Functions in Convolutional Neural Networks** (https://proceedings.mlr.press/v51/lee16a.html)
  AISTATS 2016 develops mixed max-average and learned pooling functions and reports improvements over conventional pooling across several CNN benchmarks with modest overhead. It supports the family, not a specific CIFAR ResNet-20 endpoint coefficient.
- **Fine-tuning CNN Image Retrieval with No Human Annotation** (https://arxiv.org/abs/1711.02512)
  GeM provides a differentiable power-mean continuum between average and max and improved global image descriptors. Its retrieval setting is indirect evidence for balanced CIFAR classification, but the smooth salience mechanism is relevant to an 8x8 final map.
- **CutMix** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/cutmix.md`)
  Local EXP-010 evidence confirms that class-bearing spatial regions improve the accepted model. Pooling candidates must still respect CutMix's area-proportional labels rather than assuming peak evidence alone is aligned.

## Experimental History Review

- EXP-010 remains the 94.15% frontier with width-2 postactivation ResNet-20, p=0.5 alpha-1 CutMix on N1/M7 strong views through 80%, and a hard weak tail. It finished at its best with healthy 89.73% switch fit and 26,898 updates.
- Decay variants, stronger CutMix, and canonical preactivation failed. The latter two crossed the 87.08 strong-underfit marker; preserve the accepted optimizer, postactivation blocks, and augmentation pressure.
- EXP-012 showed representation geometry can reach 94.22% at equal compute but weakened strong fit. EXP-013 found batch 256 supplies only 18.91% fresh-paired image-throughput gain, below its 20% gate; more image exposure is not currently a sufficiently strong isolated systems lever.
- EXP-013's mandatory plan review found that max-based metrics require equal evaluation opportunity. EXP-014 preserves batch 128, so the accepted 390-step epoch structure and 19-evaluation behavior remain unchanged.
- The system decomposition attributes 97.6% of step cost to model forward/backward and only 0.61% of H20 memory. A one-endpoint pooling change has memory room but must retain at least 97% of updates because SE demonstrated that nominally small reductions can be launch-bound.
- The live gap is representation/generalization without more strong-phase suppression. The final 128x8x8 post-ReLU map is currently reduced by pure average, leaving spatial aggregation as a narrow, untested readout lever.

## Collected Ideas

Quick path: the previous thorough review already identified final spatial aggregation as the strongest live runner-up after batch scaling. This loop compares three exact members of that narrow family rather than reopening discarded optimizer, regularization, compiler, or identity-initialization directions.

## Combinations

None. Every finalist preserves the complete EXP-010 recipe and changes only final pooling/readout; combinations would weaken attribution.

## Candidate Ideas

### Fixed Average-Max Feature Blend
**Summary**: Replace global average features with `torch.lerp(global_avg, global_max, 0.5)` while keeping the exact 128-to-10 classifier and every parameter tensor. The fixed symmetric blend preserves dense area evidence while adding peak salience, with no learned gate, capacity, or RNG shift.

**What it targets**: Compact class-bearing responses on the final 8x8 map can be diluted by pure averaging. The blend gives every location half of the average-path gradient and one maximum location an additional half, testing localized readout without changing strong augmentation or model capacity.

**Reasoning**: Mixed pooling has direct peer-reviewed CNN evidence, and EXP-010 validates spatially localized CutMix information. This is the cleanest attribution test because shared activations, classifier initialization, state, and parameter count remain identical. The upward feature-scale shift is explicitly part of the net method, not claimed away.

**Sources**: AISTATS mixed-pooling paper; EXP-013 `proposals/idea-05.md`; EXP-010; EXP-013 Claude review.

**Estimated Effort**: medium

**Risk Assessment**: Max is area-insensitive against CutMix targets, an 8x8 map limits the dilution premise, feature magnitude rises, extrema may be augmentation artifacts, and half the gradient concentrates 65x at one location. One extra reduction/backward must pass a 97% exposure gate.

### Fixed GeM-3 Pooling
**Summary**: Replace global average pooling with parameter-free generalized mean pooling at fixed `p=3`: clamp nonnegative post-ReLU features to a declared epsilon, average their cubes spatially, then take the cube root. Retain the 128-wide classifier and all accepted mechanics; do not learn or tune `p`.

**What it targets**: GeM smoothly emphasizes high responses without max pooling's single-index gradient or complete area insensitivity. On the 8x8 final map it can preserve compact salience while distributing gradient across every positive activation according to magnitude.

**Reasoning**: GeM is evidence-backed for global visual descriptors and supplies a principled continuum (`p=1` average, large `p` max). Fixed `p=3` is a common moderate operating point and avoids a learned scalar confound. It may align with CutMix better than a 50% max blend because patch extent still influences the power mean.

**Sources**: Radenovic et al. GeM paper; EXP-010 CutMix evidence; `02-system-understanding.md`.

**Estimated Effort**: medium

**Risk Assessment**: Retrieval gains may not transfer to balanced classification; epsilon and power operations change scale/numerics and add multiple sequential kernels; gradients favor large activations and can amplify RandAugment artifacts. The fixed p/epsilon cannot be tuned after timing or accuracy.

### Concatenated Average-Max Classifier
**Summary**: Concatenate global average and maximum features into 256 dimensions and use `Linear(256,10)`, with the average-half weights copied bitwise from the accepted classifier and the max-half weights initialized exactly zero. Construct the expanded layer inside a CPU RNG fork after normal accepted-model initialization, preserving post-construction RNG and initial logits. Each class can then learn its own peak contribution rather than receiving a universal coefficient.

**What it targets**: Different CIFAR classes may rely differently on distributed texture versus localized parts. A class-specific linear readout can exploit both statistics while leaving the convolutional representation and training recipe intact.

**Reasoning**: Learned mixed pooling is the strongest form supported by the AISTATS evidence. Zero max-half initialization makes average-only behavior an exact starting special case: shared model/average-classifier gradients match control on the first backward, while max-half weights receive gradients and can depart only through learned evidence. The parameter cost is 1,280 weights and width previously showed that useful capacity can beat fixed-time cost.

**Sources**: AISTATS mixed/gated pooling paper; Wide Residual Networks knowledge entry; EXP-007/010.

**Estimated Effort**: medium

**Risk Assessment**: The refined fork/copy/zero policy removes initial-logit, fan-in, and RNG confounds, but capacity and aggregation still change together by design. Max-area mismatch and sparse gradients emerge after learned max weights grow; 1,280 new parameters may overfit; launch overhead needs measurement. A below-87.08 switch fails the strong-fit diagnostic but cannot trigger partial freezing or coefficient tuning.

## Review

The mandatory Claude idea review completed successfully and is preserved in `01-idea-review.md`; no fallback reviewer was used. It selected the concatenated readout at 7/10 evidence and 7/10 potential impact because it has the highest ceiling and is the only finalist that can learn class-specific peak/extent balance instead of imposing an unsupported universal statistic. I adopted its high-value zero-initialized max-half refinement and strengthened it with exact RNG discipline: construct/initialize the accepted model and classifier normally, then create the expanded classifier inside `torch.random.fork_rng(devices=[])`, copy accepted weight/bias into the average half, and zero the max half. This preserves post-construction CPU RNG, initial logits, and first-backward shared gradients.

The review rejected the fixed 50/50 blend as anti-aligned with area-proportional CutMix labels and too gradient-concentrating, and rated GeM second because p=3 is indirect retrieval evidence with more kernel/numeric risk. It also required all pooling candidates to preserve the >=97% exposure guard and diagnose a switch checkpoint below 87.08% as strong-fit failure. A bare 94.25-94.35 pass remains weak single-seed causal evidence.

## Idea Evaluation

- **Concatenated average-max classifier**: selected with the zero-max-half refinement. It is a strict learned superset that begins output-identical to EXP-010 and offers the highest ceiling with only 1,280 additional weights.
- **Fixed GeM-3**: runner-up on smooth area-sensitive gradients and parameter-free attribution, but p=3 lacks local classification support and adds the heaviest sequential endpoint math.
- **Fixed average-max blend**: not selected because hard max is area-insensitive, forces a 50% peak coefficient, shifts scale immediately, and concentrates gradient against the diagnosed generalization limiter.

## Chosen Idea
**Selected**: Zero-Initialized Concatenated Average-Max Classifier

**Why this idea**:
It turns average-only pooling into an exact initial special case while adding a class-specific path for peak evidence. The accepted classifier weights, logits, shared gradients, and RNG/data stream are preserved at initialization; any later max contribution must be learned from the unchanged EXP-010 recipe. Compared with a fixed blend or GeM exponent, the model can ignore max evidence when CutMix area labels make it unhelpful and use it only for classes/features where localized salience improves generalization.

**Hypothesis**:
A 256-to-10 classifier whose average half starts bitwise at the accepted solution family and whose max half starts at zero will learn useful class-specific localized readout without weakening initial strong-phase dynamics, retain at least 97% of EXP-010's 26,898 updates, and raise `best_test_acc` from 94.15% to at least 94.25%. A switch checkpoint below 87.08% diagnoses strong-fit failure but cannot trigger max-path freezing, rescaling, or a rerun.

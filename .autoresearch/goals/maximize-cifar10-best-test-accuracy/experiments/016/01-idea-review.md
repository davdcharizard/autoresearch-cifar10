# Adversarial Review - EXP-016 Finalists

## Prioritized Feedback

1. **BF16-width3 funding must account for TF32.** On the H20, the default FP32 convolution arm may already use TF32 tensor-core paths, so the relevant comparison is BF16 against the actual accepted default rather than against idealized true FP32. Record `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32`, leave the defaults unchanged, and treat failure to reach the predeclared BF16-over-default margin as the expected no-go rather than a reason to alter the operating point.
2. **Width 2 to 3 is an uncertain diminishing-returns extrapolation.** EXP-007 proves only width 1 to 2. Width 3 has 2.25x the width-2 parameters, fewer updates, and unchanged LR/decay, so saturation or capacity starvation remain plausible even if timing passes.
3. **BlurPool overlaps with existing crop augmentation.** Every training view already uses padded random cropping, so the translation-robustness mechanism may add less on 32x32 CIFAR with two transitions than the cited ImageNet evidence suggests. The ResNet20 pooling-search paper shows that downsampling matters but does not validate this fixed binomial kernel; the honest prior is closer to flat than the proposal's optimistic range.
4. **BlurPool can reproduce the dominant strong-underfit signature by detail suppression.** Blurring the two key boundaries may erase class detail and dilute CutMix regions, while dense transition convolution adds backward cost. This differs mechanistically from EXP-012/015 identity initialization but can fail in the same observable way.
5. **Nesterov has the weakest accuracy mechanism.** The EXP-001 deconfounding case is valid, but Nesterov adds no data, capacity, invariance, or exposure and lacks direct support at this operating point. A 94.25-94.30 result would be near single-seed resolution.
6. **Only width 3 has a ceiling plausibly above the 0.10-point gate's noise scale.** Nesterov and BlurPool are credible tests, but a bare pass would provide weaker causal evidence without an allowed reroll.

## Scored Verdict

| Candidate | Evidence and reasoning | Potential impact |
|---|---|---|
| BF16-funded width 3 | **6/10** - width is the only locally proven multi-point lever, but the funding argument must survive default TF32 and width 2 to 3 is unproven. | **8/10** - highest ceiling and the only candidate whose success could clearly exceed seed noise if both numerical and funding gates pass. |
| Full-path BlurPool | **6/10** - coherent and outside the failed identity family, but external evidence is not direct for this kernel under CIFAR crop/CutMix. | **5/10** - modest ceiling with meaningful detail-loss, switch-fit, and compute risks. |
| Isolated Nesterov | **6/10** - clean attribution and legitimately unresolved, but its evidence does not support a gain over tuned SGD here. | **3/10** - likely upside lies on the acceptance boundary. |

## Pick

**BF16-Funded Width-3 Postactivation ResNet-20.** It targets capacity, the only intervention family with a demonstrated multi-point local gain, and has the largest credible upside. Its hard three-arm timing and paired-numerical gates prevent an unfunded or unstable candidate from consuming the sole fixed-seed run. The plan must resolve the TF32-default question first and must stop without substitution if BF16 does not fund the declared width-3 exposure.

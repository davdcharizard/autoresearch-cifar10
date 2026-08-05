**Prioritized Feedback**

1. **Idea-01 BlurPool is the only finalist that directly attacks the diagnosed structural limiter.**  
   The history says this is a generalization ceiling, not epoch, optimizer, capacity, BN-noise, or augmentation bound (`03-experiment-learnings.md`, `project-insights.md`). Replacing the untouched downsampling stack in `train.py:149-152` is a real architectural inductive-bias change. This is the strongest fit to the mandate.

2. **BlurPool’s main risk is hidden throughput and padding/phase correctness, not scope.**  
   It is `train.py`-only and no-deps, but `MaxPool2d(stride=1) + reflect pad + depthwise conv` may be slower than the proposal assumes because small grouped convs and `F.pad` can be memory/kernel-launch bound. Also verify padding/phase against the Zhang/Adobe pattern; output shape equality is not enough. Address with shape smokes, kernel-sum smokes, memory-format smokes, and a hard `num_epochs >= ~135` gate. Avoid per-forward `self.kernel.to(x.dtype)` allocation if possible.

3. **Do not blur the final 4x4 head pool in the primary BlurPool cell.**  
   The proposal’s cA, blurring only layer1/2/3 and leaving `nn.MaxPool2d(4)` unchanged, is clean. The final-pool variant is less specified and could over-smooth or produce a different output shape if implemented as a naive drop-in. Treat final blur as a follow-up only after the main downsampling test.

4. **Idea-02 SE is plausible but underspecified around initialization and residual scaling.**  
   The proposal preserves ReZero identity for `GatedResidual(256)` in `train.py:119-137`, but it also wants layer3 SE, where `Residual(512)` is not gated. A vanilla sigmoid SE gate can shrink the residual branch at init and disturb the validated recipe. Fix by identity-initializing the gate, for example zero-init the final SE projection and use an identity-centered scale such as `2 * sigmoid(...)`, or restrict the first test to explicitly identity-safe placement.

5. **SE’s evidence is real but less tailored to this ceiling.**  
   Hu et al. supports SE on ImageNet-scale backbones, but this CIFAR net is already wide, heavily augmented, EMA’d, and capacity-saturated. SE is a new functional form, not just width, but it is still extra trainable capacity/channel recalibration in a setting where width/depth and regularization axes have repeatedly tied. Expected effect may sit inside the ~0.1pp noise floor.

6. **Idea-03 AdaptiveConcatPool is feasible but too shallow for the stated limiter.**  
   It is almost free and likely correct, but it changes only the final readout after the feature extractor. The diagnosis asks for a different architectural inductive bias; avg⊕max pooling is a weak version of that. The fastai/NiN support is generic and not strong evidence that this saturated CIFAR head is the bottleneck. It is better as a rider on a stronger architectural change than as EXP-018’s main bet.

7. **All finalists require same-session control and confirmation.**  
   Stored 96.38 is too close to the noise floor. EXP-016/017 showed how a low control draw can fake a win. The winning cell must beat same-session c0 by >0.1pp and clear the absolute 96.48 bar, then be confirmed.

**Scored Verdict**

| Idea | Evidence / Reasoning | Potential Impact | Overall |
|---|---:|---:|---:|
| **Idea-01 BlurPool** | **8/10**: Strong mechanistic match to untouched aliasing-prone downsampling and supported by Zhang 2019; fits the “new inductive bias” diagnosis. | **7/10**: Best chance to move the ceiling, though CIFAR-10 small images plus strong crop/flip/aug may shrink the gain. | **7.5/10** |
| **Idea-02 SE** | **6.5/10**: Well-known method and feasible, but transfer evidence is less specific and init details need tightening. | **5.5/10**: Could help via channel attention, but may be redundant with a saturated high-capacity CIFAR model. | **6/10** |
| **Idea-03 AdaptiveConcatPool** | **5/10**: Correct, cheap, and easy to test, but the cited evidence weakly supports this exact bottleneck. | **3.5/10**: Low upside; likely a sub-noise readout tweak rather than a ceiling breaker. | **4.5/10** |

**Pick: Idea-01, BlurPool anti-aliased downsampling.**

Run the primary layer1/2/3 BlurPool variant as EXP-018, with final head pooling unchanged. It is the strongest because it is the clearest architectural inductive-bias change, touches an untested structural weakness in the current model, and avoids the already-saturated optimizer/capacity/augmentation/regularization lanes. The main refinement is implementation discipline: verify padding/shape/throughput before trusting the result.

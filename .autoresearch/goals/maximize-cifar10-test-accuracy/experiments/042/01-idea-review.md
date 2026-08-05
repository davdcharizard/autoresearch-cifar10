# Adversarial Idea Review EXP-042

## Prioritized Feedback

1. **The common premise is plausible but not diagnosed: discarded spatial
   degrees are not evidence that they improve the boundary.**
   `01-brainstorm.md` correctly identifies that GAP discards 63 of 64 spatial
   degrees per channel, but neither `02-system-understanding.md` nor the prior
   scores identify lost spatial arrangement or salience as a current error
   mode. EXP-036 proves only that cheap nonlinear processing of the retained
   means helps. All three candidates are therefore exploratory representation
   tests, not targeted repairs. Keep exactly one score, avoid favorable
   training-statistic gates, and state a positive as support for the complete
   pooling treatment rather than proof that it found objects or recovered a
   diagnosed information bottleneck.

2. **The exact-neutral centered-attention algebra is sound, including the
   first-step gradient, but actual batch cancellation and subsequent sharpness
   are the main hidden assumptions.**
   For `S=64`, `q=0` gives `a_s=1/64`, so the correction is zero and the
   production mean path remains the accepted `adaptive_avg_pool2d`. With
   `g_b = dL/dz_b`, differentiating the softmax gives
   `dL/dq = sum_b Cov_population(X_b) g_b`; the `1/S` is already inside the
   population covariance, and a mean-reduced CE is handled through the scale
   already present in `g_b`. At this state, the correction contributes zero
   feature gradient because both `a-u` and `q` are zero, leaving
   `dL/dx_s = g_b/S`. The proposal's formula is therefore correct. What remains
   unproven is whether the full 256-example mixup/hard gradients cancel to a
   negligible value or whether accepted LR/Nesterov rapidly drives unscaled
   logits into concentrated softmax. The preflight should report, without
   gating or tuning, scorer-gradient norm versus scorer/update norm, score
   standard deviation, attention entropy/effective-site count, and max weight
   after the first independent fresh update in both regimes. Require only a
   finite nonzero data gradient and nonuniform post-update attention. Do not
   use those values to add a temperature, scale, or nonzero initialization.

3. **The attention candidate is materially different from both prior zero-init
   failures and the closed SE family, but prior SE evidence is not affirmative
   evidence for this lower-rank selector.**
   EXP-014 zeroed six residual endpoint convolutions, removed their initial
   random residual features, and blocked each branch's upstream path until its
   endpoint moved. Here the complete accepted GAP/head path and its first
   backbone gradient remain intact while the sole new parameter receives a
   direct covariance gradient on backward one. The SE gates in EXP-017--025
   instead modify two signed residual branches, emit 128 channelwise scales,
   and require a dense two-layer channel map. The proposed query performs one
   dense channel dot product but emits only one shared scalar distribution over
   positions. This distinction keeps it outside the closed family; it also
   means the SE result does not show that a one-query rank-one spatial routing
   mechanism has enough output expressivity. Tighten the rationale to
   "preserves global channel mixing in its scorer" rather than implying it
   preserves full SE's dense channel-output interaction.

4. **Exact initial identity is credible only if proved on the actual CPU and
   CUDA kernels; common-gradient bitwise equality is unnecessarily stronger
   than the causal contract.**
   `1/64` is exactly representable in FP32, zero logits should produce exact
   uniform softmax, and a BMM with exact-zero centered weights should return a
   zero correction. The restoring RNG fork and zero overwrite also correctly
   preserve accepted initialization without a new seed. Keep the proposed
   device checks for zero coefficients, correction, pooled values, logits, and
   BN evolution. For common gradients, use a declared tight numerical bound if
   autograd graph accumulation prevents byte equality; the essential semantic
   facts are an unchanged initial function/backbone gradient and a nonzero
   scorer gradient, not verifier-specific byte ordering.

5. **Compute feasibility is the lead candidate's principal operational risk.**
   The attention arithmetic is tiny, but it adds a spatial `1x1` convolution,
   softmax, centered subtraction, BMM, and their backward kernels. The accepted
   run has only 2.536% throughput headroom above 127 passes, and prior H20
   results show small-kernel/MAC estimates are unreliable. Retain the balanced
   complete-update timing gate exactly as written, including both mixup and
   hard regimes. A stable timing miss should close only this execution topology
   and make no accuracy claim.

6. **Spatial standard deviation is the weakest causal test because its
   statistic is likely redundant and its startup is arbitrarily active.**
   The final tensor is BatchNorm then ReLU. For roughly standardized pre-ReLU
   channels, post-ReLU mean and standard deviation are strongly coupled, so
   `sigma` may add little independent information to the accepted mean and MLP.
   `1e-5` is numerically safe, and population variance is the correct statistic
   for all 64 sites, but borrowing BatchNorm epsilon does not calibrate the
   statistic's semantic scale. Likewise, reusing EXP-036's `0.1` does not
   calibrate `0.1*sigma`, and identity initialization immediately changes
   logits and every common gradient. A result would conflate dispersion,
   generic extra linear conditioning, epsilon, scale, and active startup.
   Training-batch correction/mean and logit-delta diagnostics can expose an
   obviously broken implementation but must not be used to tune these choices.
   A miss should reject only the exact `sqrt(var+1e-5)`, identity, scale-0.1
   formulation and deprioritize adjacent rescues; it cannot show second-order
   pooling is useless.

7. **The centered 2x2 proposal contains explicit absolute positional bias and
   a much larger arbitrary active branch.**
   Centering removes each channel's common quadrant component, but flattening
   four ordered quadrant slots into an unconstrained `512 -> 128` matrix gives
   every output separate top-left/top-right/bottom-left/bottom-right weights.
   That is a learned positional parameterization even without a separate
   positional-embedding tensor. It is neither translation invariant nor
   horizontal-flip equivariant, directly conflicting with random crops and
   flips. Kaiming seed 42042 and scale 0.1 immediately perturb the accepted
   path with 65,536 weights, four times the accepted pooled MLP's parameter
   count, so a miss is difficult to attribute and a gain could be dataset
   centering. Centering and contrast-oracle checks are correct, but they do not
   cure this inductive-bias conflict. This candidate should not lead.

8. **Tighten the selected hypothesis and closure before planning.**
   Use: "If a single shared content query can exploit useful nonuniform spatial
   evidence while preserving the accepted endpoint at initialization, then
   this exact zero-started, temperature-one centered-softmax pool will retain
   at least 127 passes and score at least 94.58%." A success supports that exact
   adaptive pooling treatment; it does not establish object localization, GAP
   inferiority in general, or an SE mechanism. A valid normal-exposure miss
   falsifies the exact one-query treatment. It is reasonable to decline
   immediate temperature/init/scale/query-count/cutoff rescues as post-result
   tuning, but record that as experiment-policy closure rather than evidence
   that all learned spatial pooling is ineffective.

## Scored Verdict

### Exact-Neutral Centered Content-Attention Pooling

- **Strength of evidence and reasoning: 4/5.** The mechanism is precisely
  specified, its exact-neutral forward and covariance-open first gradient are
  mathematically correct, it preserves the accepted learner, and it is clearly
  distinct from prior gates; direct evidence that spatial salience limits this
  model is still absent.
- **Potential impact: 4/5.** It can change the sufficient statistic supplied
  to the successful pooled MLP on every example at negligible parameter cost,
  although one shared spatial distribution limits expressivity and makes a
  large gain uncertain.

### Identity-Initialized Spatial-Standard-Deviation Residual

- **Strength of evidence and reasoning: 2/5.** Dispersion is genuinely absent
  from GAP and the formula is valid, but post-BN/ReLU mean/std redundancy and
  the uncalibrated epsilon/scale/identity-active start leave the causal argument
  heavily confounded.
- **Potential impact: 3/5.** A useful second statistic could improve the pooled
  representation cheaply, but the fixed per-channel statistic is limited and
  may mostly duplicate activation magnitude.

### Centered 2x2 Spatial-Contrast Residual Readout

- **Strength of evidence and reasoning: 2/5.** Centering and compute accounting
  are careful, but no error evidence supports absolute quadrant layout, and
  the proposed unconstrained projection conflicts with accepted translation
  and flip augmentation while starting from an arbitrary active seed.
- **Potential impact: 3/5.** It exposes more spatial degrees and substantial
  readout capacity, but that capacity can as readily overfit dataset position
  as improve robust class boundaries.

## Selected Lead

**Exact-Neutral Centered Content-Attention Pooling** is the single strongest
candidate. It wins because it tests the new spatial-information hypothesis
with the smallest intervention, preserves the accepted function and first
backbone gradient exactly, has an analytically open first update, avoids fixed
coordinates, and retains dense channel evidence in the scorer. The standard-
deviation treatment is more confounded by statistic redundancy and arbitrary
active scaling, while the 2x2 treatment adds explicit positional bias and much
larger active capacity. Advance the attention candidate with the tightened
hypothesis, attribution, diagnostics, and closure above.

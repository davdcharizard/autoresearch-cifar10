# Brainstorm EXP-037
**Created**: 2026-08-06

## Web Search & Literature Review

- **Characterizing signal propagation to close the performance gap in unnormalized ResNets** (ICLR 2021; https://openreview.net/forum?id=IX3Nnir2omJ)
  Adapted weight standardization controls channel-mean growth in ResNets; transfer to a shallow BN network is indirect, but it supports mean centering as a representation/conditioning mechanism.
- **Micro-Batch Training with Batch-Channel Normalization and Weight Standardization** (https://arxiv.org/abs/1903.10520; distilled in `proposals/idea-03.md`)
  Mean subtraction supplies much of weight standardization's benefit and BN+WS can improve ImageNet, but full all-layer normalization is unnecessary and expensive here.
- **NormFace / L2-constrained Softmax** (https://arxiv.org/abs/1704.06369; https://arxiv.org/abs/1703.09507; distilled in `proposals/idea-02.md`)
  Feature/weight normalization needs an explicit logit scale; scale 8 clears a ten-class theoretical fitting bound but face-recognition transfer is weak.
- **Batch-statistics calibration literature** (ICLR/OpenReview records; installed PyTorch BN semantics)
  Running-stat adaptation can matter under domain shift, but fixed momentum0.1 analytically erases inherited strong statistics by the first 390-batch weak evaluation.

## Experimental History Review

- EXP010 remains 94.15% with width2, N1/M7+p0.5 CutMix through 80%, then a hard weak cosine tail. Data-policy, optimizer-path, residual-order, transition, pooling, initialization, and activation changes have either regressed or failed safety gates.
- EXP035/036 established that controls must qualify denominator-safe global gates before candidate authority. Reflection then failed candidate-specifically at 20.72x logits; future work should preserve constant padding and avoid per-site/zero-norm ratios.
- EXP029 showed all-Conv gradient centering is active and trajectory-safe but costs 1.97%; a single stem projection is a distinct representation mechanism with much lower expected overhead. EXP034 warns against scale-dividing reparameterizations.
- The weak tail is nearly monotonic and EXP018/030/032 all failed; a pure BN-buffer reset is likely erased before the first legal look. New candidates need either a bounded representation mechanism or an independently plausible conditioning effect.

## Objective Limiter Diagnosis

The frontier is limited by generalization under a short strong phase, not memory, host overhead, or terminal iterate noise. A useful candidate should preserve the accepted curriculum and ordinary SGD while changing representation in an intrinsically bounded way. Stem-only mean centering removes image-facing DC response with a non-expansive projection and one-site cost. A cosine head provides a stronger angular inductive bias and absolute logit bound, but scale8 is an unvalidated hyperparameter and inverse-norm Jacobians change classifier geometry globally.

## Collected Ideas

- **Stem-only mean-centered convolution** — subtract each stem output filter's coefficient mean on every forward; targets low-frequency crop/illumination common mode with a non-expansive projection and one-site cost.
- **Fixed-scale cosine classifier** — normalize pooled features and classifier rows, omit bias in-function, and multiply cosine logits by fixed scale8; bounds logits but changes the readout immediately.
- **One-time BN-stat reset at 80%** — discard strong-view running moments before weak training; analytically almost certainly erased before the first legal evaluation.
- **Full weight standardization** — center/divide every Conv filter; literature-backed but repeats scale/overhead risks and is too broad after EXP029/034.
- **Biasless classifier symmetry** — zero/omit class offsets; rejected in EXP036 review as likely sub-threshold.
- **Weak-tail BN freeze** — retain broad-view moments; likely wrong because evaluation is clean weak data.
- **Channels-last execution** — attacks backward cost but still lacks an exposure-to-accuracy mechanism.
- **Moonshot angular margin head** — add a fixed cosine margin; label-dependent margin handling conflicts with CutMix probability targets and needs extra tuning.

## Combinations

- **Stem centering + cosine head** could remove low-frequency nuisance before enforcing angular separation, but combines two optimizer geometries and destroys attribution.
- **Stem centering + channels-last** might subsidize projection overhead, but layout has no established speedup and should not mask the representation test.
- **BN reset + cosine head** cannot rescue a buffer effect analytically gone by the first look and would only add complexity.

## Candidate Ideas

### Reset BN Running Statistics at the 80% Boundary
**Summary**: Reset all 19 running means/variances/counters once after the boundary evaluation and weak-loader rebuild, preserving affine parameters, weights, optimizer, and data. Full specification: `proposals/idea-01.md`.

**What it targets**: Evaluation normalization mismatch between strong augmented and weak clean views.

**Reasoning**: The state-only isolation is excellent, but momentum0.1 leaves only `0.9**390 = 1.43e-18` inherited contribution by the first weak look. A survival preflight should reject it as an effective no-op rather than score it.

**Sources**: installed PyTorch BN semantics; EXP010/018/032; `proposals/idea-01.md`.

**Estimated Effort**: medium.

**Risk Assessment**: Very high null-effect risk; adding an early evaluation would game the metric.

### Mean-Centered Stem Convolution
**Summary**: Replace only `ResNet.conv1` with a Conv2d subclass that uses `weight - weight.mean((1,2,3), keepdim=True)` in forward, without variance division or stored-parameter mutation. Full specification: `proposals/idea-03.md`.

**What it targets**: Image-facing DC/common-mode nuisance and stem conditioning while retaining edges, color contrast, constant crop padding, and the accepted residual network.

**Reasoning**: Weight-standardization literature isolates mean subtraction as useful; the projection cannot expand effective weight or data-gradient norm. Stem-only scope differs from EXP029's all-layer gradient helper and EXP034's norm-shrinking fan-out. BN may make it redundant, and removal of low-frequency color evidence or >1% overhead remain real risks.

**Sources**: Qiao et al.; Brock et al.; EXP029/034/036; `proposals/idea-03.md`.

**Estimated Effort**: high due to projection/momentum proofs and paired timing.

**Risk Assessment**: Medium-high; cleaner geometry than cosine, but effect size and BN redundancy are uncertain.

### Fixed-Scale Cosine Classifier
**Summary**: Retain the Linear module/state but compute bias-free scale8 cosine logits from normalized pooled features and normalized classifier rows. Full specification: `proposals/idea-02.md`.

**What it targets**: Class separation and confidence generalization through angular rather than radial geometry, with logits intrinsically bounded to [-8,8].

**Reasoning**: NormFace/L2-softmax provide a scale mechanism and scale8 exceeds the ten-class theoretical fitting floor. The absolute bound directly addresses prior unbounded heads, but face-recognition evidence is indirect, scale dominates the outcome, and inverse-norm Jacobians/coupled decay alter SGD geometry.

**Sources**: NormFace; L2-constrained Softmax; EXP014/031/034/035; `proposals/idea-02.md`.

**Estimated Effort**: high.

**Risk Assessment**: High accuracy and optimization risk; recurring normalization kernels may also miss the 99% exposure gate.

## Review

Claude's independent review (`01-idea-review.md`) selected **Mean-Centered Stem Convolution**, scoring evidence/reasoning 7/10 and impact 5/10. It rejected BN reset because default momentum erases the effect before any legal look, and judged cosine's higher ceiling outweighed by unvalidated scale8, CutMix magnitude loss, and the goal's recurring readout-geometry collapse pattern. Its main stem concern is BN redundancy: only 3.7% expected stem-filter energy is removed and the following BN may wash out the effect.

I adopt the pick and add the reviewer's null-detector: real hard/CutMix pre/post-BN and pooled/logit divergence must survive initialization and a preregistered short strong replay before timing. This is a mechanism-survival floor, not an accuracy proxy; controls must qualify it and lower loss cannot select the candidate. Mean-only, stem-only scope and unchanged initialization/decay remain fixed.

## Idea Evaluation

- **Mean-centered stem** — Advance. Best local evidence-to-risk ratio, non-expansive geometry, one-site cost, and clean distinction from prior centering/initialization failures.
- **Fixed-scale cosine classifier** — Defer. Intrinsically bounded but scale-sensitive, globally changes readout geometry, and may erase CutMix ambiguity encoded by feature magnitude.
- **BN running-stat reset** — Reject. Analytically erased before the first legal evaluation and lacks an optimizer/representation mechanism.

## Chosen Idea
**Selected**: Mean-Centered Stem Convolution

**Why this idea**:
It projects only the image-facing stem filters onto a zero-mean basis, suppressing local DC/common mode while preserving accepted constant padding, data, residual graph, and SGD. The orthogonal projection cannot expand effective weight or data-gradient norm, and one-site scope avoids EXP029's all-Conv cost. Unlike EXP034, it performs no variance division or stored-norm shrink.

**Hypothesis**:
Differentiable mean centering of only `ResNet.conv1` will produce measurable post-BN/pooled representation change without candidate-specific class or update instability, retain at least 99% fixed-budget exposure, preserve at least 89.0% switch accuracy, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. A mechanism-null, safety/timing veto, or valid miss retires this exact mean-only stem point without all-layer, scaled, phase, or optimizer rescue.

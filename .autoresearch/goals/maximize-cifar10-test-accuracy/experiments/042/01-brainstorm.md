# Brainstorm EXP-042
**Created**: 2026-07-27

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the accepted compute-effective spatial backbone and place new readout work after its expensive 8x8 transformations.
- **RandAugment** (`knowledge/papers/randaugment.md`): accepted crop/flip and early random transformations encourage invariance, so any spatial readout must avoid rigid positional dependence or keep it subordinate.
- **mixup** (`knowledge/papers/mixup.md`): preserve the accepted input/target interpolation and sole CE; EXP041 showed that adding another representation objective weakens the pooled-head frontier.

No network, remote source, or new retrieval was used. This thorough pass is offline and uses only the persistent knowledge base, accepted source, measured system understanding, and 41 completed experiment records.

## Experimental History Review

- EXP036 remains the 94.48% frontier with 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Its post-GAP residual MLP proves cheap nonlinear channel interaction can improve the readout, but it receives only 128 global means and cannot recover spatial information discarded by pooling.
- EXP037-041 close local classifier decay, hard-tail LR, equal classifier radii, and direct-path auxiliary CE around the accepted head. Preserve sole refined-path CE, ordinary affine classifier freedom, `5e-4` matrix decay, and the global cosine.
- The final post-BN/ReLU map is `128 x 8 x 8`, but accepted adaptive average pooling collapses each channel to one scalar. Spatial dispersion, coarse arrangement, and content-specific salience are therefore structurally absent from the successful pooled head; this is an untested information bottleneck, not a test-label diagnosis.
- Learned spatial processing must remain cheap: backward dominates counted time and new feature-map convolutions have failed or sacrificed exposure. EXP012's rank-64 spatial bottleneck scored 93.74, while EXP017-025 close stage-3 SE gating. A fixed reduction plus at most pooled-scale parameters is materially distinct.
- Memory and wall time are not binding; a new readout must retain at least 127 passes and improve generalization rather than fitting the already near-zero training tail more aggressively.
- The goal is not saturated: 94.48% is not near a CIFAR-10 ceiling and pooling still discards a quantified 63 of 64 spatial degrees per channel. The relevant uncertainty is whether those degrees contain robust class evidence under accepted augmentation.

## Collected Ideas

- **Exact-neutral centered content-attention pooling** - learn one 128-element channel scoring vector over final spatial positions. Add the weighted residual using `softmax(scores) - 1/64`, so zero initialization is bitwise global average pooling yet receives nonzero gradient. This targets salient-location information with minimal capacity and no fixed coordinates.
- **Spatial-standard-deviation residual** - compute per-channel population standard deviation of the final 8x8 map and add `0.1 * Linear_128(std)` before the accepted pooled head. It exposes dispersion discarded by the mean with only 16,384 parameters, but epsilon, identity projection, and scale define an arbitrary non-neutral startup.
- **Centered 2x2 spatial-contrast readout** - subtract each channel's global mean from four quadrant means, project the resulting 512-vector to 128 dimensions, and add it at scale 0.1. It exposes coarse layout at low MAC cost, but adds 65,536 weights and fixed-grid translation/flip sensitivity.
- **Fixed generalized-mean pooling** - replace GAP by nonnegative GeM with prospectively fixed power three. It emphasizes strong activations without new state, but the power is uncalibrated, changes accepted startup globally, and cannot adapt which locations matter by content.
- **Average-plus-max pooling simplification** - add a fixed small max-minus-mean correction to GAP. This exposes peak presence cheaply, but the coefficient is arbitrary and max gradients are sparse/unstable; learned centered attention subsumes it more smoothly.
- **Gradient centralization** - subtract non-output-axis means from convolution/linear data gradients before accepted Nesterov. It has zero inference cost and may regularize filters, but no measured mean-gradient failure exists and its coupled-decay semantics plus reduction overhead are underdiagnosed.
- **One-time hard-boundary momentum reset** - clear accepted Nesterov buffers immediately before the first hard-label update. It cleanly targets the mixup transition but inherited velocity decays below 1% in about 44 updates, giving low expected impact.
- **Spatial covariance pooling moonshot** - summarize cross-channel covariance over 64 locations and low-rank project it before classification. It directly captures second-order co-occurrence but requires a large 128x128 statistic, stabilization/normalization choices, and costly backward reductions unsupported by local evidence.

## Combinations

- **Centered attention pooling + accepted pooled MLP**: content-dependent spatial selection supplies information before the already validated nonlinear channel remap. This is stronger than attention or an alternate head alone because it preserves the successful head and changes only its sufficient statistic; it is the natural isolated treatment, not a compound optimizer change.
- **Standard deviation + global mean**: keeping the accepted mean direct path while adding dispersion can distinguish concentrated and diffuse channel evidence. The combination is stronger than replacing GAP by standard deviation, but its fixed projection/scale still confound statistic value with added capacity.
- **2x2 contrasts + early RandAugment**: accepted transformation diversity could teach the coarse readout to ignore unstable quadrant cues while using robust arrangement. That interaction could beat either alone, but a fixed grid also conflicts directly with the invariance objective and may overfit dataset centering.

## Candidate Ideas

### Identity-Initialized Spatial-Standard-Deviation Residual
**Summary**: Preserve accepted GAP, compute per-channel population standard deviation as `sqrt(var + 1e-5)`, and add a fixed scale-0.1 identity-initialized bias-free `128 -> 128` projection before the accepted pooled MLP. This adds 16,384 parameters and keeps all learned work after a fixed statistic reduction.

**What it targets**: Per-channel spatial dispersion that GAP provably discards, allowing the accepted nonlinear head to combine activation presence with concentration/extent information.

**Reasoning**: EXP036 validates cheap post-pooling projection capacity, while this statistic is distinct from EXP012's learned spatial bottleneck and EXP017-025's multiplicative stage-3 gates. Population variance has a clear semantic role, but the branch changes the initial function and its epsilon/scale/identity start lack local calibration. Full contract: `proposals/idea-02.md`.

**Sources**: `experiments/012/04-analysis.md`; `experiments/017/04-analysis.md`; `experiments/036/04-analysis.md`; `experiments/041/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Mean and standard deviation after BN/ReLU may be redundant; variance backward can be launch-bound, and a miss cannot separate statistic quality from the arbitrary active projection startup.

### Exact-Neutral Centered Content-Attention Pooling
**Summary**: Preserve accepted GAP exactly, then add a centered attention correction from one zero-initialized bias-free `Conv2d(128,1,1)` scorer: `mean + sum((softmax(q^T x) - 1/64) * x)`. It adds 128 decayed parameters, no temperature/scale/position state, and is mathematically a single-query content-weighted pool before the accepted pooled MLP.

**What it targets**: The structural information loss at final global-average pooling, learning content-specific spatial salience before the accepted pooled residual head without fixed coordinates.

**Reasoning**: A zero score vector produces uniform softmax weights; subtracting exact `1/64` makes the attention residual identically zero and leaves accepted pooled bytes unchanged. A synthetic FP32 check gives bitwise initial identity and nonzero score-weight gradient, distinguishing it from failed delayed-opening zero residual endpoints. Full contract: `proposals/idea-01.md`.

**Sources**: `experiments/014/04-analysis.md`; `experiments/017/04-analysis.md`; `experiments/036/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: Uniform averaging may be the beneficial invariant, batch covariance gradients may cancel, and softmax may later concentrate on augmentation artifacts. A miss closes this exact one-query centered-softmax pool and immediate temperature/init/query/cutoff rescues, not other independently motivated pooling statistics.

### Centered 2x2 Spatial-Contrast Residual Readout
**Summary**: Compute four quadrant means, subtract the broadcast global mean per channel, flatten the 512 contrast values, and add a scale-0.1 bias-free `512 -> 128` Kaiming projection alongside the accepted pooled head. It adds 65,536 parameters and no learned spatial-grid operation.

**What it targets**: Coarse final-map arrangement erased by GAP while retaining the accepted global-mean path and nonlinear channel head.

**Reasoning**: Centering prevents simple duplication of GAP and reduces the statistic to spatial contrasts. EXP027 supports capacity/invariance interactions, and the readout is about 52 times cheaper than EXP012's failed spatial bottleneck. Fixed quadrants nevertheless conflict with accepted translation/flip invariance and add much more capacity than attention pooling. Full contract: `proposals/idea-03.md`.

**Sources**: `experiments/012/04-analysis.md`; `experiments/017/04-analysis.md`; `experiments/027/04-analysis.md`; `experiments/036/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: The branch may overfit object centering or quadrant boundaries, and a gain could reflect generic linear capacity rather than spatial layout. A miss closes only this centered 2x2 projection, not translation-equivariant attention.

## Review

The offline adversarial reviewer selected exact-neutral centered content-attention pooling at 4/5 evidence and 4/5 impact. I adopted its corrections: discarded spatial degrees are a structural opportunity rather than a diagnosed error mode; the covariance-gradient formula is correct, but batch cancellation and rapid softmax concentration must be reported as non-tuning diagnostics; the scorer preserves global channel mixing in one query but not full SE's dense channel-output interaction; and common initial gradients may use tight FP32 bounds if autograd accumulation prevents byte equality. Success supports only this exact adaptive pool, not object localization or GAP inferiority. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Attention pooling tests the spatial-information hypothesis with only 128 zero-started weights, no fixed coordinates, exact accepted forward startup, and an analytically open first update. Standard deviation is confounded by post-BN/ReLU redundancy and arbitrary active scaling; 2x2 contrasts add explicit position bias and 65,536 actively initialized weights.

## Chosen Idea
**Selected**: Exact-Neutral Centered Content-Attention Pooling

**Why this idea**:
Replace only the final pooling sufficient statistic with one shared content query. Preserve the accepted GAP kernel and add `sum((softmax(q^T x) - 1/64) * x)` before the accepted pooled MLP, with a zero bias-free scorer. At initialization the correction and common functional change are exactly zero, while the scorer receives `sum_b Cov_spatial(X_b) * dL/dz_b` on backward one. This isolates adaptive spatial selection at minimal parameter cost and without positional embeddings, loss changes, or classifier constraints.

**Hypothesis**:
If one shared content query can exploit useful nonuniform spatial evidence while preserving the accepted endpoint at initialization, the exact zero-started, temperature-one centered-softmax pool will retain at least 127 projected and realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final accuracy at least 94.45% and loss at most 0.2456 are corroboration only. A valid normal-exposure miss closes this exact one-query treatment and immediate temperature/init/scale/query-count/cutoff rescues as experiment policy, not all learned pooling mechanisms.

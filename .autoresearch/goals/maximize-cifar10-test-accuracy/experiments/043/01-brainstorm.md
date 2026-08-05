# Brainstorm EXP-043
**Created**: 2026-07-27

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the accepted compute-effective residual backbone; a new optimizer rule must improve the same function rather than consume the narrow exposure margin with more spatial work.
- **mixup** (`knowledge/papers/mixup.md`): preserve the accepted batch-shared interpolation through 65%; optimizer state spans a real objective transition, but the inherited momentum component decays rapidly.
- **RandAugment** (`knowledge/papers/randaugment.md`): preserve the proven early image-invariance interaction; additional destructive image regularization has a weak local prior.

No network, remote source, or new retrieval was used. This thorough pass is offline and uses only the persistent knowledge base, accepted source, measured system understanding, and 42 completed experiment records.

## Experimental History Review

- EXP036 remains the 94.48% frontier with 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Its cheap post-GAP residual MLP improves decision quality while the hard tail nearly interpolates training, so generalization and boundary quality, not memory, I/O, or convergence wall time, remain limiting.
- Backward consumes about 74% of an isolated step and only 2.536% throughput can be lost before projected exposure falls below 127 passes. New spatial forward/backward work is therefore disfavored; an optimizer intervention can be attractive only if its reductions and launch latency pass direct H20 timing.
- EXP037-041 close adjacent classifier decay, hard-tail LR, equal classifier radii, and direct-path auxiliary supervision around the accepted head. EXP042 further shows that a mathematically neutral learned pooling addition can still harm the well-calibrated invariant readout. Preserve the exact model, sole refined-path CE, classifier freedom, decay, schedule, and GAP.
- EXP021-022 reject expensive and sparse final-window SAM, but no prior experiment applies a one-backward algebraic projection to convolution data gradients. Convolution-only gradient centralization is thus an orthogonal optimization test, not a retry of SAM, classifier normalization, or decay tuning.
- EXP003, EXP006, and EXP030 all make additive masking a low-prior direction: CutMix, broad residual dropout, and narrow drop-path each hurt at normal exposure. A smaller input Cutout is mechanically distinct but opposed by this recurring information-removal evidence.
- The mixup-to-hard-label switch is causally real, but no transition instability is measured. At momentum 0.9, inherited buffer memory falls below 1% in 44 updates, only about 0.48% of the accepted hard tail, so a one-time reset is clean and nearly free but unlikely to move top-1 by 0.10 points.
- The remaining untested gap is whether the dominant convolution-weight updates contain harmful common-mode filter motion. The proposed projector removes only 1,392 directions from 983,472 convolution weights and leaves inference, parameters, the pooled head, and classifier exact, but local history does not diagnose those directions as the error source.

## Collected Ideas

- **Convolution-only gradient centralization** - after backward and before accepted SGD adds coupled decay, subtract each output filter's mean data gradient over input and spatial axes for all 18 convolutions. It targets noisy common-mode filter motion at zero inference cost while deliberately preserving the successful linear readout gradients; its 36 small CUDA operations are the leading exposure risk.
- **One-time hard-boundary momentum reset** - delete every live Nesterov `momentum_buffer` immediately before the first hard-label update. It isolates whether mixed-target velocity briefly conflicts with hard-label refinement and costs essentially nothing, but the affected inherited component naturally decays below 1% within 44 updates.
- **Early per-example 8x8 Cutout** - place one normalized-zero square after accepted mixup through the exact 65% cutoff using an isolated device generator. It adds a local occlusion invariance distinct from area-labeled CutMix, yet three prior masking interventions make additional early information deletion unlikely to help.
- **Canonical per-channel input standardization** - divide the already mean-centered CIFAR inputs by fixed training-set standard deviations before the unchanged WRN. It could equalize stem conditioning at negligible cost, but most scale effects are absorbed by learned stem weights and downstream BatchNorm, while selecting statistics introduces an untested fixed preprocessing change.
- **Decoupled matrix weight decay** - replace coupled SGD decay with an equivalent-looking SGDW shrink on the accepted matrices. It separates regularization from momentum geometry and could prevent decay accumulation in velocity, but it is a broad optimizer rewrite whose effective shrink is not matched by the same `5e-4`; nearby classifier decay experiments give no supporting diagnosis.
- **Higher Nesterov momentum** - raise momentum from 0.9 to 0.95 for more temporal averaging without additional compute. It may reduce stochastic noise over 300 seconds, but it is a bare hyperparameter move with no local bracket, changes the entire trajectory, and risks sluggish late boundary refinement under the fixed cosine.
- **Filter-wise tangential gradient projection** - remove from each convolution gradient the component parallel to its current filter, separating directional learning from norm control while leaving weight decay to govern norms. This imports normalized-geometry reasoning without changing inference, but it is more restrictive and costly than centralization and conflicts with evidence that ordinary affine geometry matters.
- **Bias-free classifier simplification** - remove the ten classifier biases and rely on feature geometry plus weight vectors. It is exact in cost and may reduce class-prior fitting, but it changes the accepted function at initialization and prior classifier constraints substantially worsened accuracy; ten parameters are not a diagnosed limiter.
- **Low-rank covariance pooling moonshot** - summarize second-order channel co-occurrence over the final 8x8 map and project a fixed low-rank statistic into the accepted pooled head. It could expose information GAP discards, but EXP042 favors preserving uniform pooling and covariance backward reductions would consume scarce exposure while adding several arbitrary normalization choices.

## Combinations

- **Gradient centralization + accepted matrix decay**: centralize only the raw convolution data gradient before PyTorch adds the unchanged coupled decay. This is stronger and more attributable than projecting the optimizer's effective direction because it conditions stochastic feature learning while retaining ordinary weight-norm shrink and Nesterov semantics.
- **Momentum reset + accepted hard-label tail**: align the single buffer deletion with the existing objective transition and leave the global cosine untouched. This is stronger than an arbitrary epoch reset because its timing has a causal meaning, though the short 44-update memory window still limits expected impact.
- **Early Cutout + accepted mixup/RandAugment**: post-mixup local erasure could demand distributed evidence while the accepted transformations build global invariance. The combination is potentially richer than any component alone, but the local record says a third early regularizer is more likely to over-regularize than complement the stack.

## Candidate Ideas

### One-Time Full Nesterov Momentum Reset at the Hard-Label Boundary
**Summary**: In the existing one-way transition from mixup to hard labels, delete all 52 live optimizer `momentum_buffer` keys before the first hard-label forward/backward/update. Preserve the accepted LR, decay, Nesterov coefficient, data path, model, and later RandAugment transition exactly.

**What it targets**: A possible optimizer-state mismatch when the objective changes from mixed soft targets to hard-label CE at 65% counted time.

**Reasoning**: Deleting the buffer removes exactly `0.81*b_prev` from the first Nesterov direction while retaining the current hard gradient's fresh-state factor 1.9. It is causal, auditable, and nearly free, but at momentum 0.9 the isolated inherited state falls below 1% in 44 steps out of roughly 9,114 hard-tail updates, so expected impact is near zero. Full contract: `proposals/idea-02.md`.

**Sources**: accepted `train.py`; `experiments/002/04-analysis.md`; `experiments/004/04-analysis.md`; `experiments/020/04-analysis.md`; `experiments/036/03-execute.md`; `proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: The accepted run shows no measured loss spike or instability at the transition, and resetting useful velocity may briefly harm rather than help. A miss closes immediate subset, scaling, timing, LR-restart, and combined-rephase rescues as post-result tuning.

### Convolution-Only Gradient Centralization
**Summary**: Precompute all 18 `Conv2d.weight` tensors, then after every accepted backward subtract each output filter's data-gradient mean across input-channel and spatial axes before the unchanged PyTorch SGD step. Exclude BatchNorm, the pooled residual MLP, classifier weight/bias, and activation gradients. The projection runs for the whole scored trajectory, adds no state or parameters, and occurs before coupled `5e-4` decay.

**What it targets**: The remaining generalization/boundary gap under a compute-bound 130-pass learner. It tests whether correlated common-mode convolution updates are a poorly generalizing direction while preserving the exact accepted function and readout.

**Reasoning**: The rule removes 1,392 scalar directions from 983,472 convolution weights without rescaling the remaining gradient. It is materially cheaper than SAM and avoids the linear-layer geometry implicated by EXP040-042. A local microbenchmark measured about 0.260 ms per step for a broader all-matrix Python projection loop, approximately 2.24% of an 11.6 ms step, so the narrower convolution-only rule is feasible enough for a strict full-body timing gate but not presumed free. Full contract: `proposals/idea-01.md`.

**Sources**: `02-system-understanding.md`; `experiments/021/04-analysis.md`; `experiments/022/04-analysis.md`; `experiments/036/04-analysis.md`; `experiments/040/04-analysis.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: Positive ReLU features make uniform filter directions nonredundant; 1x1 shortcuts have no spatial redundancy; coupled decay will erode projected-away mean components; and roughly 36 sequential reduction/subtraction kernels may breach the 127-pass exposure floor. A normal-exposure miss closes this exact all-convolution rule without layer exclusions or schedules.

### Early Per-Example 8x8 Post-Mixup Cutout
**Summary**: Through exactly 65% counted time, apply one independently located 8x8 normalized-zero square per example after the accepted batch-shared mixup and before the model. Use a fixed private device generator so accepted mixup and global RNG streams remain unchanged; the hard-label tail is an exact bypass.

**What it targets**: Remaining generalization error through local occlusion robustness, a property not directly imposed by crop/flip, mild RandAugment, or convex whole-image mixup.

**Reasoning**: The hole covers only 6.25% of each image and does not alter target mass, making it distinct from EXP003 CutMix and feature-space masks. Nevertheless, EXP003, EXP006, and EXP030 all show that information removal compounds the accepted regularizers poorly, so this is a low-confidence alternative retained for breadth rather than the lead. Full contract: `proposals/idea-03.md`.

**Sources**: `experiments/003/04-analysis.md`; `experiments/006/04-analysis.md`; `experiments/026/04-analysis.md`; `experiments/027/04-analysis.md`; `experiments/030/04-analysis.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: The third early regularizer may weaken feature learning, normalized-zero holes may become a synthetic cue, and mask generation adds counted input writes. A normal-exposure miss closes additive early Cutout/Random Erasing and immediate size/probability/fill/order rescues on this stack.

## Review

The offline adversarial reviewer selected convolution-only gradient centralization at 3/5 evidence and 3/5 potential impact. I adopt its corrections: the projection operator is rigorous but harmful common-mode motion is not a measured diagnosis; the stem, positive ReLU features, and 1x1 shortcuts make the removed directions nonredundant; and the earlier 2.24%-overhead microbenchmark leaves a borderline rather than affirmative timing case. Per-tensor removed-gradient fractions will be non-tuning diagnostics, the counterbalanced complete-step H20 gate remains authoritative, and a miss rejects only this exact global convolution rule. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Gradient centralization is the only finalist that acts persistently on the dominant convolutional representation while preserving inference, model state, the successful pooled head, classifier freedom, and the accepted data/loss/schedule recipe. Cutout compounds a repeatedly negative masking family, while momentum reset directly changes less than 0.5% of the hard tail. Selection remains conditional on semantic and timing qualification; a timing failure is infeasibility, not an accuracy result and not evidence for either lower-ranked candidate.

## Chosen Idea
**Selected**: Convolution-Only Gradient Centralization

**Why this idea**:
Apply one global, convolution-only rule after every backward: subtract each output filter's mean raw data gradient over its input and spatial axes, then let unchanged coupled decay and Nesterov operate normally. This probes an untested optimizer geometry throughout training, removes only 1,392 of 983,472 convolution directions, and leaves accepted forward behavior plus all pooled-head and classifier gradients untouched. The premise is exploratory rather than diagnosed, so exactly one qualified score is warranted and adjacent layer/schedule variants are not rescues.

**Hypothesis**:
If persistent common-mode convolution data-gradient motion impairs fixed-budget generalization, the exact full-run projection over all 18 convolution weights will retain at least 127 projected and realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final accuracy at least 94.45% and final loss at most 0.2456 are corroboration only. A valid normal-exposure miss closes this exact convolution-only centralization rule without exclusions, partial strengths, schedules, alternate axes, or linear-layer additions.

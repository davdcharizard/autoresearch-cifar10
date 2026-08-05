# Brainstorm EXP-039
**Created**: 2026-07-27

## Web Search & Literature Review

- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can be removed while retaining benefit, but the optimizer schedule must still support late fitting after the objective changes.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): soft targets can improve calibration, while stacking them with another soft-target mechanism without diagnosis is risky.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): moderate width/depth is effective on CIFAR, consistent with preserving the accepted spatial learner and spending new complexity outside expensive feature maps.

No network, remote source, or new literature retrieval was used. This offline thorough pass used the persistent knowledge base, accepted source, measured system understanding, and all 38 indexed experiments.

## Experimental History Review

- The accepted frontier is EXP036 at 94.48%, with a 94.45% endpoint, 0.2456 loss, and 130.304 passes. Its `128 -> 64 -> 128` scale-0.1 pooled residual MLP is the only recent mechanism that improved both accuracy and loss.
- The learner nearly interpolates the clean training tail, while test loss remains 0.2456. Compute rather than memory or I/O is binding: forward/backward dominates counted time, so any new mechanism should preserve at least 127 passes and avoid spatial work.
- Classifier decay is now locally bracketed: zero and `1e-3` both lose at normal exposure around accepted `5e-4`. Mixup strength/duration, residual masking, averaging, SAM, batch scaling, precision/layout, SE, late prefix freezing, and adjacent capacity points are closed or infeasible in tested forms.
- The 65% boundary is a real nonstationarity: mixed targets and RandAugment give way to hard labels and clean crop/flip, but the accepted global cosine and Nesterov buffers continue without rephasing. At the boundary LR is 0.06123; the global cosine falls to 0.03395/0.01812/0.00736 at 75/82.5/90%, while a continuous tail-local cosine would give 0.05008/0.03162/0.01315.
- This objective is not saturated: 94.48% is below known CIFAR capacity and prior composition produced a 0.25-point gain after standalone near misses. The search is narrow but still has untested loss-geometry and regime-transition mechanisms.

## Collected Ideas

- **Continuous hard-tail cosine rephase** - at the exact 65% objective boundary, preserve LR continuity at 0.06123 but restart cosine phase over the remaining 35% to the accepted 0.002 floor. It targets under-adaptation after mixed-to-hard target transition, increases only mid-tail update amplitude, introduces no free scalar, and differs from EXP008's harmful move to a zero floor.
- **Hard-tail momentum reset** - clear every SGD momentum buffer exactly when mixup disables, before the first hard-label update. This discards stale mixed-target velocity without changing gradients, LR, graph, or throughput. The effect is sharply attributable but likely lasts only tens of steps under momentum 0.9, limiting upside.
- **Regime-aligned optimizer restart** - combine the continuous tail-local cosine with a one-time momentum reset at the same boundary. The pair treats the clean tail as a second optimization phase and is more coherent than either isolated control, but it compounds state and schedule interventions and weakens causal diagnosis.
- **Gamma-1 focal hard tail** - preserve accepted paired mixup CE early, then replace only clean-tail CE with focal loss at gamma 1. This suppresses already-correct easy examples after near interpolation and concentrates the fixed decision budget on hard examples without an extra coefficient, but hard examples may be mislabeled or unrepresentative.
- **Direct-path auxiliary CE** - during training only, apply a scale-0.1 auxiliary CE to accepted classifier logits from the unrefined pooled vector while retaining the refined path as the sole evaluator output. This regularizes the successful pooled head to preserve a linearly useful direct representation at negligible spatial cost, but may suppress the nonlinear remapping that supplied EXP036's gain.
- **Class-vector orthogonality penalty** - penalize off-diagonal cosine similarity among the ten classifier rows using a coefficient tied prospectively to accepted weight decay. It targets boundary diversity without changing inference or feature norms, yet the coefficient's effective scale is uncertain and CIFAR balance alone does not diagnose collapsed class directions.
- **Mean-norm equalized classifier** - normalize each classifier row and restore a differentiable common mean row norm, removing class-specific radial bias while preserving feature norms and avoiding a fixed cosine temperature. Initialization perturbation should be small and inference overhead negligible, but normalized-weight SGD and decay semantics become indirect.
- **Worker-side random erasing moonshot** - add a mild early tensor-space erasing policy under the existing worker-private temporal gate. This imports an occlusion invariance distinct from color/geometric RandAugment and preserves GPU compute, but CutMix and feature masking failures warn that additional occlusion may overregularize the accepted recipe.

## Combinations

- **Tail cosine rephase + momentum reset**: both align optimizer state to the same known objective discontinuity. The rephase sustains clean-tail learning while the reset prevents the first clean updates from inheriting mixed-target velocity, plausibly producing a truer second phase than either alone; the cost is weaker attribution.
- **Direct-path auxiliary CE + class-vector orthogonality**: the auxiliary objective keeps raw pooled features linearly separable while orthogonality diversifies class directions. Together they could regularize both sides of the affine boundary more completely, but two coefficients and no local geometry diagnosis make the combination premature.
- **Focal hard tail + momentum reset**: reset velocity before switching to a hard-example-weighted objective. This avoids carrying mixup momentum into the focal phase, but creates a larger objective discontinuity and should follow isolated evidence rather than serve as a first test.

## Candidate Ideas

### Training-Only Direct-Path Auxiliary CE
**Summary**: During training, blend 90% accepted refined-path CE with 10% CE from `fc(z)` on the raw pooled vector, reusing `POOLED_HEAD_SCALE=0.1` rather than adding a coefficient. Evaluation remains exactly the accepted refined logits; both paths share the classifier and mixup targets.

**What it targets**: Boundary robustness of the accepted pooled residual head by preserving linear usefulness of the dominant direct representation at negligible spatial cost.

**Reasoning**: EXP036 proves nonlinear pooled refinement is useful, but does not establish whether its raw direct path remains independently discriminative. Convex 90/10 loss mixing preserves approximate gradient scale and supplies cheap deep supervision without adding an inference parameter. Full contract: `proposals/idea-03.md`.

**Sources**: `experiments/036/04-analysis.md`; `02-system-understanding.md`; `03-experiment-learnings.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: It reallocates 10% of supervision away from the only newly validated head mechanism, may force the shared classifier to compromise between feature geometries, and adds small launch-bound classifier/loss work. The coefficient is structurally reused rather than empirically justified.

### Gamma-1 Focal Loss Only in the Hard Tail
**Summary**: Keep accepted paired mixup CE through 65%, then use unnormalized multiclass focal loss `mean((1-p_t) * CE)` with exactly gamma 1 on hard targets. Add no alpha, blending, detachment, normalization, ramp, or schedule compensation.

**What it targets**: Fixed clean-tail decision budget currently spent on examples whose training CE is already near zero. Gamma-1 focal reweights gradients toward ambiguous examples without changing model inference or spatial work.

**Reasoning**: The learner nearly interpolates training while test loss remains material, so confidence-based example weighting is a plausible distinct loss geometry. Restricting it to the hard tail avoids stacking another soft-target mechanism with mixup; gamma 1 is the minimal standard focal exponent and introduces no continuous coefficient. Full contract: `proposals/idea-02.md`.

**Sources**: `02-system-understanding.md`; `knowledge/papers/label-smoothing.md`; `knowledge/papers/time-matters-regularization.md`; `experiments/036/04-analysis.md`; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Hard examples may be noisy or atypical crops, and unnormalized focal loss suppresses aggregate gradient scale near interpolation, potentially undoing the accepted nonzero LR floor. There is no local focal evidence.

### Regime-Aligned Hard-Tail Cosine Rephase
**Summary**: Preserve the accepted LR function exactly through 65%, then start a new cosine at the accepted boundary value `0.06123215` over the remaining 35%, ending at the protected `0.002` floor. Change no momentum state or other behavior. The boundary, start, duration, and endpoint all derive from accepted constants, so the treatment adds no tunable scalar.

**What it targets**: The known mixed-target-to-hard-label objective transition, which currently inherits a cosine already 70% below peak. The candidate increases interior clean-tail LR area by 39.46% while adding only scalar host arithmetic and preserving the measured >=127-pass compute envelope.

**Reasoning**: EXP008 showed that reducing late update amplitude by moving the endpoint to zero harms both accuracy and loss. This proposal keeps the useful endpoint but allocates more motion after early regularization leaves, consistent with the temporal-phase motivation in the local Time Matters note. Momentum reset is deferred because its inherited contribution falls below 1% in about 44 updates, whereas rephasing acts over roughly 9,000 tail steps. Full contract: `proposals/idea-01.md`.

**Sources**: `experiments/008/04-analysis.md`; `experiments/036/04-analysis.md`; `knowledge/papers/time-matters-regularization.md`; `02-system-understanding.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: The accepted tail already nearly interpolates and finishes near its best, so 39.46% more LR area may cause stochastic wandering or stronger effective coupled decay. A miss closes immediate tail-rephase shapes but leaves the separately attributable momentum reset open.

## Review

The offline adversarial critic selected the continuous hard-tail cosine rephase at 7/10 evidence and 7/10 impact. I adopted its causal limits: larger LR also increases coupled-decay integration; EXP008 supports preserving motion but not necessarily adding more; a success validates only the complete fixed-seed package; and a miss falsifies the exact curve while merely deprioritizing nearby schedule tuning. Exposure, endpoint, and loss are interpretation checks rather than alternate success paths, and the isolated momentum reset remains open. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. The rephase has the strongest parameter-free construction, sustained effect, and systems isolation. Focal loss risks nearly extinguishing already-small tail gradients without local evidence, while auxiliary CE weakens the only recently validated head and lacks a diagnosed direct-path failure.

## Chosen Idea
**Selected**: Regime-Aligned Hard-Tail Cosine Rephase

**Why this idea**:
Keep the exact accepted LR through the 65% mixed-target boundary, then use the accepted boundary LR as the start of a new cosine over the remaining accepted tail duration to the accepted 0.002 floor. This directly tests whether the hard-label phase benefits from a schedule aligned to its own duration, changes no graph or optimizer state, adds no free scalar, and should preserve the 130-pass operating regime. The intervention is a package of larger data-gradient and coupled-decay updates; it is not uniquely attributable to either.

**Hypothesis**:
If the accepted fixed-seed learner benefits from the exact 65%-anchored tail rephase package, the candidate will retain at least 127 projected and realized passes and raise `best_test_acc` from 94.48% to at least 94.58%. Final accuracy >=94.45% and loss <=0.2456 are corroboration only. A normal-exposure miss falsifies this exact curve with 39.46% greater tail LR area and deprioritizes nearby rephase tuning without formally rejecting every tail schedule; isolated momentum reset remains untested.

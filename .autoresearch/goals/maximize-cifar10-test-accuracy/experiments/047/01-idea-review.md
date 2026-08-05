# Adversarial Idea Review EXP-047

## Prioritized Feedback

1. **Advance Early Post-GAP Pooled-Feature Mixup Replacement, but state the hypothesis as the complete replacement rather than as evidence for manifold linearity alone.** Its source contract is coherent: the accepted Beta draw and permutation move from `256x3x32x32` inputs to the captured `128`-D post-GAP vector, the paired targets and strict 65% cutoff remain aligned, the accepted nonlinear pooled head stays downstream, and the hard/evaluation path uses `feature_mix=None`. This is the only finalist that opens a materially new training geometry at negligible spatial cost. However, proposal idea-02 lines 56-69 correctly acknowledge that the score jointly removes mixed-pixel backbone/BN training and adds pre-MLP feature interpolation. Preserve that narrow bundled interpretation and do not claim either component independently caused an outcome.

2. **The feature-mix candidate silently depends on a small post-GAP nonlinearity being enough to replace the strongest validated spatial regularizer.** EXP002 gained `+0.69` points from early input mixup, whereas accepted `train.py` maps the mixed pooled vector mostly through the affine direct path `fc(z_mix)`; only the scale-`0.1` residual MLP processes the interpolation nonlinearly. Thus a late mixture may be too easy, may regularize mostly the small head, and may fail to impose the input-space invariance that later composed successfully in EXP027. The saved `mixup.md` supports input interpolation, not this representation placement. Retain proposal idea-02's preregistered training-only head Jensen-gap, grouped backbone-gradient, and clean-versus-mixed-BN diagnostics, but keep them non-gating and do not use them to restore input mixup or move the site after seeing results.

3. **A miss for pooled-feature mixup cannot empirically close all representation placements listed in its no-rescue section.** Stage-1/2/3 mixing sends synthetic maps through different downstream nonlinearities and BatchNorms, while post-head/logit mixing has different or degenerate semantics; they are not variants of the exact post-GAP/pre-MLP mechanism. It is reasonable to decline those follow-ups as search policy, but the report must say the score falsifies only the complete fixed post-GAP replacement. Otherwise proposal idea-02 lines 321-326 overstate what one bundled score establishes.

4. **Default SiLU repeats an unchanged candidate previously disqualified by the local review record and still lacks the missing diagnosis.** EXP036 explicitly preserves the exact successful ReLU head and calls activation changes an unjustified adjacent sweep; EXP045 `01-idea-review.md` then rejected this same default-SiLU substitution because signed `W2` already permits inhibitory corrections, the direct GAP path preserves `z`, and SiLU bundles a negative lobe with positive attenuation, changed zero derivative, and far-negative nonmonotonicity. EXP045-046 add no positive activation evidence. Proposal idea-01's scalar algebra and byte/RNG contract are sound, but its post-selection diagnostics cannot retroactively justify consuming the score. A future activation experiment should first have an independently motivated training-only diagnosis of dead/negative hidden regions tied to boundary quality; do not advance SiLU in EXP047.

5. **Early-only epsilon-0.05 label smoothing is semantically valid but repeats another twice-deprioritized candidate without new calibration evidence.** Proposal idea-03 correctly derives `q=0.95*y_mix+0.05*u`, the same-class mass `0.955`, mean-reduced gradient `(p-q)/B`, RNG neutrality, and the exact hard tail. The causal case remains weak: accepted mixup already supplies example-aware softness, EXP004/020 and EXP005/035 protect its timing and strength, the saved label-smoothing note warns against stacking soft-target methods, and EXP041 shows extra CE-derived pressure can weaken the pooled-head frontier. Neither near-zero late loss nor failures of unrelated EXP042-046 treatments diagnose harmful early overconfidence, and `0.05` remains conventional. Keep it closed unless a prospective training-only calibration diagnosis supplies a nonredundant target.

6. **No finalist violates the literal goal constraints or presents an evaluator/seed reward hack, but the score contract must remain conjunctive and unique.** Each proposal is `train.py`-only, dependency-free, fixed-seed, single-H20, retains the frozen 300-second budget and at-most-once-per-epoch cadence, and uses one redirected score killed at 600 seconds. Reconfirm `94.48% @ a7c42dc` by source/index audit rather than another baseline score; require both `best_test_acc >=94.58%` and realized exposure `>=127`, and never let endpoint loss, diagnostics, or a low-exposure high score select a rescue or second run.

## Scored Verdict

### Early Post-GAP Pooled-Feature Mixup Replacement

- **Strength of evidence and reasoning: 3/5.** Placement, label/pairing algebra, Jacobian, RNG, hard-path identity, and cost reasoning are rigorous, while EXP036 supports nonlinear capacity at the site; direct evidence supports input mixup rather than moving it, and the BN/removal/placement effects remain inseparable.
- **Potential impact: 4/5.** Replacing the full early convex-training law can reshape backbone and head gradients throughout 65% of training at near-zero added cost, giving it the largest credible upside even though its direction is uncertain.

### Default SiLU in the Accepted Pooled Residual MLP

- **Strength of evidence and reasoning: 2/5.** The activation math and exact-state contract are correct, but no hidden-region diagnosis overcomes EXP036's preserve-the-head guidance or EXP045's prior rejection of this unchanged candidate.
- **Potential impact: 3/5.** It can materially change the successful bottleneck's representation and gradients nearly for free, but it also attenuates every positive activation and risks erasing rather than extending the sole recent gain.

### Early-Only Epsilon-0.05 Uniform Label Smoothing

- **Strength of evidence and reasoning: 2/5.** Dense-target semantics and temporal isolation are exact, but uniform entropy is redundant with accepted mixup, epsilon lacks a local basis, and no calibration evidence identifies the claimed limiter.
- **Potential impact: 2/5.** The treatment is cheap and could modestly change confidence, but it adds no example information and mainly weakens an already bracketed early target trajectory.

## Selected Lead

**Early Post-GAP Pooled-Feature Mixup Replacement** is the single strongest idea. It wins over SiLU because it tests a distinct training-geometry hypothesis rather than reopening an activation substitution already rejected by EXP036/045 closure discipline. It wins over label smoothing because it changes where the validated convex example prior acts instead of stacking undiagnosed uniform entropy onto that prior. Advance exactly the fixed post-GAP/pre-MLP replacement with one shared draw/permutation, no input interpolation, the accepted hard/evaluation path, the semantic and paired-exposure gates, one score, and bundled-result interpretation.

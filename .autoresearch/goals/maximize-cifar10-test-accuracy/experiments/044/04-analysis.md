# Report EXP-044: Exact-Neutral Spatial-Dispersion Input
- **Created**: 2026-07-27

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the accepted94.48% frontier under the unchanged300-second single-H20 budget. The experiment tested whether fixed per-channel spatial dispersion adds boundary information absent from global means.

## Idea & Hypothesis

Preserve GAP as the complete direct path and add final-map population standard deviation through one zero-start bias-free `128->64` adapter into the accepted pooled MLP hidden preactivation. The hypothesis predicted at least94.58% best accuracy and127 realized passes if dispersion supplied useful invariant information not recoverable from the mean.

## Approach

Registered an all-zero8,192-parameter adapter after accepted pooled-head initialization inside the restoring RNG fork. The forward retained exact adaptive-average pooling, computed `sqrt(var_population+1e-5)`, and added its adapter output before the accepted ReLU/output matrix/scale0.1. An ignored harness captured the real production statistic, proved exact common startup state/logits/gradients, independently qualified nonzero statistic backward and CE/mixup adapter gradients, verified Nesterov updates, and interleaved complete-step timing.

## Execution

Semantic qualification passed directly. Mean/std correlation was0.835-0.854, the epsilon floor only0.68-0.70% of median std, and adapter gradients were nonzero. Timing passed at median retention0.984150 and projected128.239 passes. The sole score completed normally in342.5 wall seconds with correct transitions,26 unique evaluations, one H20, and no runtime error.

## Results

- **Primary metric**: 93.95% (baseline:94.48%, delta:-0.53 points,-0.56%)
- **Observations**: Exposure remained normal at25,139 steps/128.71168 passes. The branch was mathematically open and affordable, but final93.83%/0.2637 worsened both accepted endpoint accuracy94.45% and loss0.2456. Training still nearly interpolated the hard tail.
- **Analysis**: The intended statistic and optimization path worked, so this is not a null-adapter or compute failure. High post-BN/ReLU mean/std correlation supports the counter-hypothesis that dispersion mostly duplicated activation magnitude/occupancy already available to GAP and the accepted nonlinear head. Added correlated capacity and statistic gradients changed the boundary adversely. This rejects the complete fixed statistic-plus-adapter, not every second-order representation in principle; immediate statistic/epsilon/width/scale/startup/placement variants lack a new diagnosis and are closed by policy.
- **Key Learning**: Per-channel spatial std is highly correlated with post-ReLU means and does not improve the accepted pooled representation despite exact-neutral opening.

## Verification

- **Conditions**: Completion/resource passed; primary improvement and hypothesis accuracy failed.
- **Review Notes**: Results confirmed trustworthy: source scope frozen, common startup exact, analytic gradients and updates passed, timing/exposure normal, and the score was unique.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid93.95% score failed baseline94.48% and threshold94.58% at128.71168 passes.

## Unexplored Avenues

- Cross-channel covariance could add information beyond correlated marginal std, but its arbitrary rank/normalization and backward cost require independent motivation before a score.
- Variance, RMS, max, alternate epsilon, adapter width, scale, nonzero initialization, or output placement are immediate post-result rescues and are deliberately declined.
- A statistic diagnosed on training-only failure modes could justify a distinct readout later; no such diagnosis exists under the frozen evaluator contract.

## Next Steps

- **High confidence**: Preserve exact uniform GAP and the accepted pooled MLP; stop adding undiagnosed spatial summary statistics.
- **Medium confidence**: Seek a new low-cost mechanism outside closed readout, masking, classifier, and gradient-projection families.
- **Low confidence**: Early-only label smoothing remains executable but redundant with mixup and should lead only if no stronger causal candidate emerges.

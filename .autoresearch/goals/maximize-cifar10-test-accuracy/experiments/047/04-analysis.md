# Report EXP-047: Post-GAP Feature Mixup Replacement
- **Created**: 2026-07-27

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the accepted94.48% frontier under the unchanged300-second single-H20 budget. The experiment tested whether the accepted convex-label prior works better between pooled decision representations than between raw pixels.

## Idea & Hypothesis

During exactly the first65%, replace input interpolation with one batch-shared interpolation of the128-dimensional post-GAP vector immediately before the accepted nonlinear residual MLP. Preserve the same Beta law, permutation, paired labels, cutoff, exact ordinary hard/evaluation path, and all other training components. Success predicted at least94.58% best accuracy and127 passes.

## Approach

Added a default-`None` model argument and one post-GAP conditional; replaced the input-blending helper with an identical-draw pairing helper; called the candidate feature path only in the early branch. The semantic harness bound the actual functional pooled tensor and actual MLP input, used coefficient0.3 plus non-self-inverse cycles, proved default-path/state/RNG identity, forward/Jacobian equations, and Nesterov updates, then qualified full early/hard exposure. Interpretation remained bundled: spatial BNs saw ordinary augmented inputs while the pooled head saw interpolated features.

## Execution

Static review corrected an optional-argument patch initially placed on `PreActBlock` before any preflight. Semantic and timing gates then passed: actual Jacobian error was zero, the nonlinear Jensen gap was0.1045, and median projected exposure was129.453 passes. The sole score completed normally in342.5 wall seconds with correct transitions,27 evaluations, one H20, and no errors.

## Results

- **Primary metric**: 94.20% (baseline:94.48%, delta:-0.28 points,-0.30%)
- **Observations**: Exposure was normal at25,409 steps/130.09408 passes. The fixed feature pairs were highly similar initially (mean cosine0.97095), mixing reduced pooled norm to0.97528, and the downstream MLP was measurably nonlinear. Yet final94.20%/0.2619 worsened both accepted endpoint94.45% and loss0.2456.
- **Analysis**: The intervention was active, nonlinear, and affordable, so the miss is not a null path or compute artifact. Moving interpolation this late leaves most of the mapping from mixed pooled vectors to logits affine, while the spatial backbone no longer learns invariance from mixed pixels. Clean spatial BN plus post-GAP interpolation did not replace the accepted input-level mechanism. This falsifies only the complete fixed bundled replacement; other feature sites were not tested and are merely declined under no-rescue search policy.
- **Key Learning**: Pooled feature interpolation cannot replace mixed-pixel training; input-level interpolation supplies useful spatial-backbone regularization absent from a late decision mixture.

## Verification

- **Conditions**: Completion/resource passed; primary improvement and bundled hypothesis accuracy failed.
- **Review Notes**: Results confirmed trustworthy: source/state/default path, actual-tensor algebra/Jacobian, draw RNG, updates, exposure, and unique score all passed.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid94.20% score failed baseline94.48% and threshold94.58% at130.09408 passes.

## Unexplored Avenues

- Earlier spatial feature mixup would expose more downstream nonlinearity but changes compute and BN semantics; it is not empirically refuted, yet is declined as an immediate placement rescue.
- Combining input and feature mixup could preserve spatial invariance but compounds regularization and violates the fixed replacement interpretation; no local evidence supports it.
- Alternate placement, detach, normalization, coefficient, pairing, cutoff, or auxiliary variants are post-result search and remain closed by policy.

## Next Steps

- **High confidence**: Preserve accepted input mixup and its spatial-backbone exposure.
- **Medium confidence**: Seek a genuinely orthogonal GPU-local mechanism that does not replace or add supervision around accepted mixup/head paths.
- **Low confidence**: SiLU and label smoothing remain executable but lack the independent diagnoses repeatedly requested by reviews.

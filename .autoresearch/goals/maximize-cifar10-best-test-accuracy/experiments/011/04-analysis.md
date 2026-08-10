# Report EXP-011: CutMix Probability 0.75
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` from the 94.15% baseline at `7c1e7d8`; improvement required at least 94.25% under the fixed protocol.

## Idea & Hypothesis

Increase only plateau `CUTMIX_PROBABILITY` from 0.5 to 0.75. Claude selected the stronger direction because EXP-010 recovered its small strong deficit, improved immediately after the switch, and finished rising. The hypothesis predicted stronger regional invariance would clear 94.25%; an 80% checkpoint below 87.08% would diagnose excessive compounded regularization.

## Approach

Changed exactly one literal in `train.py`. Alpha 1.0, worker RNG isolation, target semantics, N1/M7, width 2, optimizer, 80% loader transition, hard weak tail, timer, seed, and evaluator remained byte-identical to EXP-010.

## Execution

Mandatory external Claude idea and plan reviews completed with exit code 0; no fallback reviewer was used. Static checks passed. A focused real-loader preflight delivered 179.02 batches/s, 75.10% mixed targets, valid target formats, and eight clean worker exits. One fixed-seed H20 run exited 0 without retry in 332.9 seconds total.

## Results

- **Primary metric**: `94.00%` (baseline: `94.15%`, delta: `-0.15` percentage points, `-0.16%` relative)
- **Observations**: Realized mixing was 16,151/21,502 strong batches (75.11%). Early clean accuracy was 81.17% at 20%, 3.18 below EXP-010, and the final strong checkpoint reached only 86.82%, crossing the 87.08% underfit marker and trailing EXP-010 by 2.91 points. The first weak checkpoint nevertheless reached 93.40%, 0.24 above EXP-010, showing rapid hard-label conversion. The tail peaked at 94.00% by epoch 63 and repeatedly returned to that ceiling through the final epoch. Final NLL was 0.1933, essentially identical to EXP-010's 0.1934. It completed 26,919 steps, 100.08% of EXP-010 exposure, with unchanged 598.7 MB VRAM.
- **Analysis**: The stronger probability achieved more regional mixing and preserved compute, but overshot the short-horizon regularization balance. Its improved first weak checkpoint suggests the composite features remained useful, yet the 2.91-point clean strong deficit left too much representation/classifier mismatch for the fixed tail. Equivalent steps and NLL rule out throughput or simple confidence calibration as the limiter. The tail plateau rather than continued climbing, so additional time is not clearly the missing factor. P=0.75 is discredited at this horizon; the successful p=0.5 point should remain accepted, and interpolation above it has weak expected value without evidence of a non-monotonic peak.
- **Key Learning**: CutMix p=0.75 crossed the strong-underfit marker and lost 0.15 points despite equal exposure; preserve the validated p=0.50 point.

## Verification

- **Conditions**: Completion, summary, hardware, scope, timing, target provenance, lifecycle, unique evaluations, parameter count, and exposure passed. Primary accuracy failed: 94.00% <94.25%.
- **Review Notes**: Results are trustworthy. Only the reviewed scalar changed; 75.11% realized mixing matches configuration; seed/evaluator and all accepted mechanics stayed fixed; no reroll occurred. The top-1 delta is modest, but the large strong-checkpoint separation and registered underfit crossing support the mechanism diagnosis.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run finished 0.15 points below baseline and 0.25 below the threshold.

## Unexplored Avenues

- P=0.25 remains unmeasured, but Claude judged its direction contrary to EXP-010's evidence and it likely dilutes the winning mechanism.
- An intermediate p between 0.5 and 0.75 may soften underfit, but the measured endpoints do not support a peak above p=0.5; avoid low-ceiling interpolation.
- Stopping CutMix before N1/M7 is still distinct, but adds lifecycle/RNG complexity and lacks direct evidence that hard-label strong views reproduce weak-tail recovery.
- A new architecture or representation lever on the accepted p=0.5 recipe now has a stronger rationale than more CutMix scalar tuning.

## Next Steps

- **High confidence**: restore p=0.5 and preserve the complete EXP-010 recipe.
- **Medium confidence**: return to a thorough, orthogonal brainstorm for a representation mechanism capable of clearing 94.25%.
- **Low confidence**: revisit CutMix timing only if a concrete lifecycle-safe mechanism beats probability interpolation in adversarial review.

## Exit Action Results

- None defined.

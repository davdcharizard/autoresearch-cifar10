# Report EXP-023: Selective Width with Full Two-Gate SE
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with a required threshold of 94.17%, by testing whether conditional channel routing makes added low-resolution capacity more useful within the fixed 300-second budget.

## Idea & Hypothesis

Compose the two strongest standalone architecture signals: a `[32,64,160]` final-stage width and two full exact-neutral ratio-16 SE gates. The hypothesis required diagnostic-free production timing to project at least 127 passes before scoring, then predicted super-additive conditional use of the extra channels.

## Approach

The implementation used explicit widths and attached two `160->10->160` gates to stage-3 residual branches before shortcut addition. New gates used fixed seed 23017 inside a restored CPU RNG fork, with zero second projections for exact identity. Preflight compared accepted, width-only, and composed models and timed accepted versus composed mixup/hard-label production paths.

## Execution

Semantic checks passed exact counts 691,674/961,562/968,302, composed-versus-width-only common state/RNG, identity logits, two-step gradient opening, placement, and optimizer groups. The initial plan's accepted-versus-width tensor equality check was corrected because EXP-010's shape-dependent constructor consumes a different RNG sequence before model-wide initialization. Stable timing then failed the preregistered exposure gate, so no scored run or `run.log` was launched.

## Results

- **Primary metric**: unavailable (baseline: 94.07%; no scored run)
- **Observations**: Accepted/composed medians were 13.2050/14.8494 ms for mixup and 12.9943/14.6057 ms for hard labels. Weighted retention was 0.889403, projecting 126.206224 passes; every timing-window CV was below 0.72%.
- **Analysis**: The composition was semantically sound and diagnostic removal kept overhead measurable, but width plus full conditional routing was still 11.06% slower than accepted. It missed the tightened 127-pass gate by 0.79 pass and landed near the multiplicative overhead predicted from the standalone experiments. Lowering the threshold after measurement would defeat the reviewer-requested protection against scoring an exposure-starved composition. This rejects the exact `[32,64,160]` plus two ratio-16 gate operating point as feasible under its preregistered premise; it does not test accuracy or establish whether the mechanisms are redundant.
- **Key Learning**: Full two-gate routing on a 160-channel final stage stacks to 11.06% overhead and misses the 127-pass feasibility floor.

## Verification

- **Conditions**: Verification not run; the execution stopped at the preregistered exposure gate.
- **Review Notes**: Preflight evidence is trustworthy and stable, but no accuracy claim can be made.
- **Verdict**: crash
- **Verdict Basis**: No primary metric was produced because the planned candidate failed feasibility before scoring.

## Unexplored Avenues

- **Two diagonal conditional gates**: retain both placements and per-example channel response while removing global MLP interaction; this is a distinct cheaper mechanism, not a rescue of the composed width model.
- **Smaller width with full gates**: could fit the envelope, but adjacent width tuning after a result-selected composition has weak justification and should not be prioritized.

## Next Steps

- **Medium confidence - two diagonal conditional gates on accepted width**: directly test whether per-channel input dependence retains the attention signal at near-baseline cost.
- **Low confidence - alpha-0.1 mixup closure**: run the remaining isolated strength-side probe despite the under-regularization prior.
- **Low confidence - non-compounding input invariance**: develop an early schedule that substitutes rather than stacks augmentation with mixup and preserves the clean tail.

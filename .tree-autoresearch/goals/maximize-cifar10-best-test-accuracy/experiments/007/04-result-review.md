# Claude Adversarial Result Review: EXP-007

Claude Opus first audited the exact run evidence, then issued this corrected review after being given the parent configuration omitted from the first prompt: EXP-004 used Euclidean SAM with `rho=0.05`. The correction supersedes the first review's claim that the parent radius was unspecified.

## Verdict Audit

**Run is valid; result is a reject.** EXP-007 executed as preregistered, integrity checks passed, and the step count (25,575 vs 25,560, +0.06%) shows no compute-budget confound. Best 95.34 misses the 95.50 threshold by 0.16 and sits 0.06 below parent 95.40, roughly one tail standard deviation, so it is indistinguishable from the parent on one fixed-seed run. The reject stands on the threshold alone.

With parent `rho=0.05` confirmed, EXP-007's first pulse had Euclidean `||epsilon||=0.450053` against the parent's scalar 0.05, an approximately 9x larger actual perturbation. Every mechanism claim must account for that difference.

## Integrity Findings

- Instrumentation is self-consistent: `||epsilon/scale||=0.5` exactly matches the configured ASAM radius, with no implementation bug indicated.
- The design is internally valid, preregistered, and free of post-hoc selection.
- There is no time or step confound; adaptive scaling did not cost measurable throughput.
- Attribution is package-level: norm geometry, effective Euclidean radius, and treatment of BatchNorm/bias parameters all differ from the parent.

## Mechanism Diagnosis

Calling this a literature-package comparison licenses the claim that published-default ASAM at this cadence did not beat the validated SAM package. It does not license the narrower claim that adaptive geometry is worse than Euclidean geometry.

The first-pulse category shares are a real diagnostic snapshot, but they are insufficient to claim persistent bias domination throughout training because both parameter scales and group gradient norms evolve. The lower final loss is suggestive, but it also cannot establish a mechanism from one seed.

## Safe Learning

1. ASAM with `p=2`, `rho=0.5`, `eta=0.01`, all tensors, start 0.75, and period two does not clear the improvement bar on EXP-004.
2. Adaptive and Euclidean radius values are not numerically comparable: adaptive `rho=0.5` produced a measured Euclidean norm near 0.45 here.
3. The pulse instrumentation is validated and reusable.
4. The result does not show that adaptive geometry hurts, that bias concentration caused the miss, or that any observed-accuracy-driven tuning is justified.

## Next Experiment

Claude's same-axis proposal is a matched-Euclidean-radius adaptive-geometry test branched from EXP-004, with group-share time-series instrumentation. Its own stated prior is low because the target effect is close to the noise floor; an unrelated mechanism with a larger prior effect is the preferred allocation, with matched-radius ASAM retained as a fallback.

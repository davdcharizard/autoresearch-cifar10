# Plan Review EXP-039

Offline fallback adversarial review found no blocking issue and required these clarifications:

1. The first real hard-label step normally has observed progress slightly above 65%, so audit its formula-derived LR; reserve exact `0.06123215295935604` equality for a synthetic 65% probe.
2. Replace abbreviated verification commands with directly runnable project-root commands and explicit TSV/script paths.
3. Declare float64 scalar tolerances for schedule values, continuity, monotonicity, and tail-area ratio.
4. Pin timing schedule probes to 50% mixup and 75% hard-tail progress so the hard benchmark exercises the changed curve.
5. Describe pre-boundary returned values as bitwise accepted; the `learning_rate()` source necessarily changes while non-schedule source/state remains byte-identical.

The schedule math, independent optimizer oracle, timing weighting, scope, sole-score threshold, exposure handling, and restrained closure claims are otherwise sound.

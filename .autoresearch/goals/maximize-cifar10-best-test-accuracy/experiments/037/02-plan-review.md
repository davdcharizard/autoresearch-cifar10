# Adversarial Plan Review — EXP-037

## Concerns

1. A small stem effect immediately before BN is noise-limited under a one-seed +0.10 protocol; preflight cannot establish causality.
2. Many hand-set gates risk another invalid outcome; EXP035 showed non-specific ratios can false-veto.
3. A 1% timing veto may reject a legitimate accuracy tradeoff and repeats EXP029's information-limited outcome.
4. Production step/evaluation floors can turn the intervention's own cost into process invalidity rather than letting the primary metric price it.
5. Ignored harness code needs identity/oracle/source-hash self-tests before it can authorize or veto.

## Resolution

- Retained the user-approved single-seed rule but explicitly limits a bare pass/miss to protocol evidence.
- Made controls load-bearing before candidate, removed per-site/zero-denominator gates, and defined persistent concentration only.
- Changed timing from a 1% exposure veto to catastrophic feasibility: ratio<=1.05, wall<540s, bounded memory/stability. Actual accuracy prices ordinary overhead.
- Removed production step floors; steps/evaluation count below the 19-look ceiling are informational.
- Made mechanism survival exceed both `1e-4` and 5x accepted control/control divergence; controller source hashes, identity telemetry, known-array math, and production-helper binding are mandatory.
- Corrected review context: the plan reuses registered EXP022/028 corpora, not EXP033 files.

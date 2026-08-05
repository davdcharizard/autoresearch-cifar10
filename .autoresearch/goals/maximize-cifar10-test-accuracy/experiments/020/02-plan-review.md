# EXP-020 Plan Review

1. **Fixed-time execution is not step-deterministic at a narrow margin.** This is an inherent property of the user-approved benchmark. Preserve the single-run fixed-seed protocol; do not add reruns or remeasure the baseline because both would violate the established constraints.
2. **Exact transition-time equality can false-fail.** Require 225.0 <= logged transition time < 226.0 and final training_seconds >=300.0, since checks occur after a completed step.
3. **The baseline is a stored single run.** Keep index baseline 94.07 as required by the autoresearch goal; same-session remeasurement would add an unauthorized scored rerun.
4. **The acceptance boundary is exact and discrete.** CIFAR-10 has 10,000 test cases, so 94.17 is exactly ten additional correct predictions over 94.07 and satisfies at least +0.10.
5. **The semantic preflight must mirror production strict inequality.** Import candidate constants/functions with a dummy evaluator and explicitly evaluate progress < cutoff at below/equal/above probes; source diff remains the primary scope oracle.


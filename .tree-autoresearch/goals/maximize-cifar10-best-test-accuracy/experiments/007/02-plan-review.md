# Claude Adversarial Plan Review - EXP-007

## Prioritized Concerns

1. The planned production radius `rho*D/(D+eps)` is tautological and does not touch constructed epsilon buffers; it cannot detect a missing second scale or wrong perturbation.
2. Aggregate geometry can hide perturbation mass concentrating in BatchNorm gamma or `fc.weight`; first-pulse denominator and epsilon-energy contributions need explicit parameter groups.
3. A merely positive Euclidean norm permits a near-zero no-op to be recorded as a valid ASAM result; first-pulse actual geometry needs a non-metric magnitude gate.
4. Fewer than 25,000 steps is an evidentiary miss, not a hard protocol violation; 24,000-24,999 must remain a valid exposure-degraded result with no retry.
5. The repair clause permits run selection after evaluation output exists; any retry must be justified from a non-accuracy failure before reading the first eval line and preserve the failed evidence.
6. Several protocol checks are structural consequences of the branch predicate, not discriminating ASAM validation; label them accordingly and center actual epsilon checks.
7. Precommit the interpretation of 95.50-95.69 and prohibit informational metrics from upgrading a noise-sized gain into mechanism evidence.
8. Record GPU co-tenancy/utilization, ensure benchmark processes exit, and verify adequate free memory before launch.
9. Require foreach elementwise operations where possible to avoid hundreds of per-tensor kernel launches eroding the step horizon.
10. Specify actual-parent harness isolation (state copy and RNG restore), inline smoke residence, and final untracked-file checks.

## Resolution

- The plan now measures `||epsilon/s||`, `max|epsilon/s|`, Euclidean norm, and four group-energy shares from the actual first production epsilon buffers, with a preregistered implementation-only magnitude band.
- Formula-derived cadence/radius facts are labeled structural; actual first-pulse geometry is the discriminating gate.
- Runtime projection still requires 25,000 steps before launch, but an actual 24,000-24,999-step run is valid, final, and exposure-degraded rather than retryable.
- Retry, GPU-state, foreach, parent-harness, interpretation, and cleanup requirements are tightened below.

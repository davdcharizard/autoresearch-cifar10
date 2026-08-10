# Claude Adversarial Plan Review - EXP-006

## Prioritized Concerns

1. Compare the refactored CutMix RNG behavior against the actual parent source at commit `1a8d0de`, not only a newly written parent-policy simulator.
2. Use exact precomputed fixed-seed policy counts rather than statistical bands in the pre-run smoke; never change the expected counts or seeds after observing a failure.
3. Claim parent RNG parity only over a shared step prefix, because the time-based cutoff can produce different total step counts.
4. Advanced indexing can lose channels-last layout; explicitly restore and assert channels-last after hidden mixing.
5. Derive the acceptance threshold from the stored parent metric with decimal arithmetic rather than a hand-entered float.
6. Resolve the tree helper through `CLAUDE_PLUGIN_ROOT` with a verified installed-path fallback.
7. Describe the generator inventory precisely: seed 43 is CPU-only, while seed 44 has CPU and CUDA generators.
8. Mean lambda alone cannot distinguish `Beta(2,2)` from a uniform distribution; also audit `mean(min(lambda, 1-lambda))`.

## Resolution

- The plan now loads and compares against the actual parent implementation, uses exact 100,000-step counts, and limits parity claims to shared prefixes.
- Hidden mixing explicitly restores channels-last layout and verifies it in semantic and GPU smokes.
- Threshold derivation, plugin-root resolution, generator inventory, and deterministic Beta-distribution shape checks are preregistered below.

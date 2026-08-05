# Adversarial Review — EXP-018 Finalists

## Prioritized Feedback

1. **RandAugment's always-on design conflicts with the validated clean tail.** EXP-002 and later schedule results support clean hard-label refinement, while additive regularization has repeatedly regressed. If revisited, RandAugment should stop with mixup and must pass a worker-throughput preflight.
2. **Efficient Channel Gate confounds placement and mechanism.** Changing two gates to one while replacing the ratio-16 MLP with a 1D channel convolution prevents useful attribution. Its low-FLOP latency premise is also unreliable on the H20, where kernel shape and launch cost have already produced large timing differences.
3. **Channel adjacency lacks a strong semantic basis.** A local 1D channel kernel may not reproduce the global cross-channel selection learned by EXP-017's MLP.
4. **Final-Block-Only SE relies on observational evidence.** Gate 1 was far more example-dependent, but gate 0's mean scale of 0.6468 was a substantial mostly static attenuation and may have contributed to the improved loss. The next run must state that causal uncertainty explicitly.
5. **Recover as much exposure as possible.** EXP-017 already established feature-dependent gate behavior, so the scored candidate does not need runtime diagnostic accumulation. Verify exact neutrality and two-step opening in an evaluator-free preflight, then remove observational diagnostics from scored code.
6. **The margin remains narrow.** Prior positive candidates cluster between 94.10 and 94.16. Final-block-only SE nevertheless has the most direct quantitative path because it retains a measured positive mechanism and targets its measured overhead.

## Scored Verdict

### Final-Block-Only Neutral SE

- **Evidence/reasoning: 4.5/5** — direct 94.16% precedent, quantified conditionality difference, and a controlled placement change; the gate-0 causal inference remains uncertain.
- **Potential impact: 4/5** — a realistic path to clear 94.17 by recovering exposure, though the expected margin is modest.

### Final-Block Efficient Channel Gate

- **Evidence/reasoning: 2.5/5** — plausible lightweight attention analogy, but it confounds two changes, relies on unverified H20 latency, and assumes useful channel adjacency.
- **Potential impact: 3/5** — upside exists if both the kernel and conditioning work, but both premises are weak.

### One-Operation Mild RandAugment

- **Evidence/reasoning: 2/5** — general CIFAR evidence is strong, but the exact always-on treatment conflicts with local clean-tail and additive-regularization evidence.
- **Potential impact: 2.5/5** — orthogonal upside in principle, with substantial over-regularization and CPU-feed risks here.

## Pick

**Final-Block-Only Neutral SE.** It attacks the diagnosed quality/overhead gap with the strongest direct evidence and isolates placement while preserving the proven ratio-16 selector. Strip scored diagnostics, verify opening before the run, and preregister gate 0's strong static attenuation as the central failure risk.


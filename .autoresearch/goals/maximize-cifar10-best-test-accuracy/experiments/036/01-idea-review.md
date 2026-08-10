# Adversarial Review — EXP-036 Finalists

## Prioritized Feedback

1. **Channels-last has two unproven links.** Tiny FP32 CIFAR kernels may not speed up, and even a 1-3% exposure gain has no established accuracy value. First prove exposure matters before investing in layout complexity.
2. **Zero classifier bias is self-admittedly sub-threshold.** Ten initial offsets have no demonstrated connection to terminal generalization and ordinary SGD can relearn them immediately.
3. **Reflection's primary risk is effect size.** The prior proposal predicts 94.27% against a 94.25% gate and admits flatness/regression is most likely. Its counter-mechanisms—useful missing-context regularization and duplicated edge objects—are real.
4. **Right-size reflection preflight.** Loader throughput is the main implementation risk; do not let optimizer-path-style per-site/relative gates dominate a data-only diff. Preserve paired corpus semantics and a bounded candidate-specific safety screen, applying EXP035's control-qualification rule.
5. **Reflection is distinct from failed data policies.** It deletes no interior pixels, changes no target/mixing rate, and preserves the simultaneous 80% transition.

## Scored Verdict

- **Reflection-Padded Strong and Weak Crops — evidence/reasoning 6.5/10; potential impact 5.5/10.** Clean boundary-prior mechanism and orthogonal scope, but no direct proof of a +0.10 gain.
- **End-to-End FP32 Channels-Last — evidence/reasoning 4/10; potential impact 3.5/10.** It targets measured backward cost but depends on both a shape-specific speedup and an unproven exposure-to-accuracy link.
- **Explicitly Zero the Final Classifier Bias — evidence/reasoning 2.5/10; potential impact 1.5/10.** No credible terminal mechanism and likely below the formal margin.

## Pick

**Reflection-Padded Strong and Weak Crops.** It is the only finalist directly aimed at generalization under the protected strong phase, preserves every validated EXP010 component, adds no GPU-model work, and is cleanly distinct from prior failed data-policy mechanisms. Treat it as a low-margin orthogonal probe and make paired loader throughput the load-bearing preflight.

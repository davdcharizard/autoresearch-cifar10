# Adversarial Review — EXP-020 Finalists

## Prioritized Feedback

1. **Early-only RandAugment repeats additive regularization during mixup.** EXP-003 and EXP-006 already show compounding transformations/feature masking with mixup regresses, while shared worker-state and CPU throughput add implementation risk.
2. **Alpha 0.1 relies on a weak monotonic inference.** Beta(0.1,0.1) is more concentrated near clean/swapped extremes, effectively weakening regularization when the 50% cutoff already showed under-regularization risk.
3. **A 75% cutoff has the only measured directional support.** EXP-004's 50% cutoff lost 0.16 relative to 65%, indicating useful regularization in that interval. The main risk is an inverted-U optimum at 65% and a shorter clean tail.
4. **All candidates have modest expected margin.** Mixup variants have not beaten 94.07, so this is a controlled gap-closing probe, not evidence of a large ceiling increase.

## Scored Verdict

- **Extend Mixup to 75%**: evidence 7/10, impact 5/10. Direct 50-to-65 directional evidence and a clean one-line treatment; overshoot remains plausible.
- **Weaker Alpha-0.1**: evidence 4/10, impact 4/10. Untested but likely under-regularizes and lacks a favorable local gradient.
- **Early-Only RandAugment**: evidence 3/10, impact 5/10. Higher conceptual ceiling, but repeats a failed compounding pattern and adds worker/throughput risk.

## Pick

**Extend Mixup to Seventy-Five Percent.** It is the only candidate supported by a measured direction and preserves the accepted mechanism with negligible implementation risk. A 70% hedge would reduce overshoot risk, but 75% offers the clearer test and a larger plausible effect against the +0.10-point acceptance margin.


# Adversarial Review — EXP-019 Finalists

## Prioritized Feedback

1. **Two-gate SE without diagnostics overstates likely exposure recovery.** The observed 98.58% versus 95.36% retention comparison confounds removing one gate with removing scalar diagnostics. Require exact matched timing if pursued; history also shows more passes alone does not reliably improve accuracy.
2. **Static first-block scale is the strongest mechanistic follow-up, but gate 0 was not literally static.** EXP-017 found feature logit RMS above bias RMS even for gate 0. Its output variance was nevertheless small relative to its mean attenuation, so a per-channel static scale can capture the dominant effect while accepting that weak input dependence may matter.
3. **The central risk is interaction rather than mean attenuation.** If two conditional gates co-adapted, static scaling will reproduce EXP-018's failure. State this as the primary hypothesis risk; do not hide it behind throughput.
4. **Early-only RandAugment compounds regularization and has a worker-propagation hazard.** Persistent workers will not observe ordinary transform mutation. A shared-memory flag inside sample loading would be required, and local additive-regularization evidence remains unfavorable.

## Scored Verdict

### Static First-Block Channel Scale Plus Final SE

- **Evidence/reasoning: 4/5** — directly restores the attenuation EXP-018 removed while preserving final conditional selection; uncertainty remains about weak gate-0 input dependence.
- **Potential impact: 4/5** — can preserve the two-gate mechanism's useful division of labor with much less runtime cost.

### Exact Two-Gate SE Without Diagnostics

- **Evidence/reasoning: 3/5** — exact 94.16 precursor, but diagnostic cost is unmeasured and likely smaller than claimed.
- **Potential impact: 2/5** — likely a narrow one-image, noise-level opportunity.

### Early-Only Mild RandAugment

- **Evidence/reasoning: 2/5** — general CIFAR support conflicts with repeated local additive-regularization failures and persistent-worker complexity.
- **Potential impact: 3/5** — opens a new axis, but the immediate failure risks are substantial.

## Pick

**Static First-Block Channel Scale Plus Final SE.** It most directly repairs the causal failure exposed by EXP-018: restore cheap learned first-block attenuation while retaining final input-dependent selection. The plan must foreground that conditional two-gate interaction, rather than attenuation alone, may have been essential.


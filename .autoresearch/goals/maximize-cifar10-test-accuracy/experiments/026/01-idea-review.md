# Adversarial Review - EXP-026 Candidate Ideas

## Prioritized Feedback

1. **Width 144 is a near-fatal closed-family retry.** Its capacity/exposure point lies between accepted and width 160, which scored only 94.11, and EXP-010 explicitly rejected adjacent width tuning. Drop it; there is no evidence for a non-monotonic interior optimum.
2. **Batch 128 holds approximate LR/batch noise scale constant.** Halving both LR and batch changes update granularity rather than clearly increasing optimizer noise. Reframe it as a granularity operating point, with halved peak/floor under-update as the principal risk.
3. **Early-only RandAugment has the strongest mechanism match.** CPU augmentation is outside counted training time in `train.py`, so it can add missing photometric/geometric invariance without paying the optimizer-exposure tax that undermined many prior ideas. Its <=500-second wall preflight is essential.
4. **RandAugment's residual risks are bounded but real.** The epoch cutoff trails mixup by under one epoch; magnitude-independent operations are not uniformly mild; worker RNG consumption changes later crop/flip trajectories. Keep all fixed and report transition/passes rather than attempting RNG compensation.
5. **Additive regularization is the main accuracy risk.** Unlike CutMix, stronger mixup, or dropout, RandAugment targets a missing invariance axis and is removed for the terminal phase, making this a distinct high-risk bet rather than an unchanged retry.

## Scored Verdict

| Candidate | Evidence / reasoning | Potential impact |
|---|---|---|
| Worker-safe early-only RandAugment | **4.5/5** - direct limiter match, relevant local literature, exposure-neutral counted path; additive-regularization escape remains unproven. | **4.5/5** - highest ceiling among genuinely untried orthogonal levers without sacrificing optimizer passes. |
| Batch 128 with scaled LR | **2.5/5** - defensible operating point, but noise-scale rationale is muddled and evidence for granularity is weak. | **2.5/5** - plausible but bounded, with material under-update risk. |
| Selective width 144 | **1/5** - adjacent retry explicitly closed by EXP-010 and not cleanly comparable under a new tail seed. | **1/5** - the local width anchor provides no evidence for reaching 94.17. |

## Pick

**Worker-Safe Early-Only One-Operation RandAugment.** It is the only finalist that directly addresses the missing image-invariance gap without reducing counted optimizer exposure. The temporal removal and fixed standard policy make its additive-regularization risk a clean one-shot hypothesis.

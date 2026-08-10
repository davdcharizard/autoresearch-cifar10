# Claude Adversarial Idea Review — EXP-005

## Prioritized Feedback

1. **All candidates predict effects inside the documented noise band.** EXP-003 measured 0.14-0.29-point selection/confirmation swings. Under the frozen one-run protocol, choose the largest defensible expected true effect rather than the cleanest implementation. HMix's stated 0.10-0.30 range and ECA's acknowledged below-gate risk make both likely no-improvement outcomes even if their mechanisms work.

2. **DLB halves evaluation opportunities on a max-statistic metric.** Its 389-step epochs imply roughly 65 evaluations rather than the parent's 132. EXP-004's final equaled its best, which mitigates the concern but does not remove it. The plan must preregister final-versus-best reporting and explicitly prohibit redefining an epoch merely to recover evaluation cadence.

3. **DLB's data-order change is an attribution confound.** A custom sampler changes the example order, which can matter near a 0.10-point gate. This is inherent rather than seed rerolling, but it must be acknowledged. Fix the sampler permutation at seed 42 and leave it untunable.

4. **DLB receives a diluted dose relative to the paper.** Clean-clean gating under 0.5 early CutMix yields about 25% early coverage, and the current parent is much more regularized than the paper baselines. The matched CutMix+DLB WRN-20-8 gain of 0.60 points is the right anchor, but the hypothesis must explicitly discount it to a plausible 0.10-0.30-point effect here.

5. **DLB changes the successful SAM tail objective.** SAM will perturb on CE+KL rather than the validated CE-only gradient. That is principled but unproven and may stack consistency/flatness regularizers too strongly. Preregister that flat accuracy with final loss worse than 0.1654 indicates an over-regularized tail.

6. **HMix is well implemented but tests a weak hypothesis.** Its clipped-label math and RNG parity are sound, yet its CIFAR-100-only evidence and 0.10-0.30-point expected effect sit in EXP-003's demonstrated saturation/noise region.

7. **ECA has the weakest evidence and likely smallest lever.** The identity-preserving `2*sigmoid` gate is good experimental engineering, but it differs from the paper's gate; the cited result is ImageNet-only, and 18 zero-initialized, weight-decayed parameters may remain near identity in the short run.

8. **DLB epoch boundaries may add uncharged worker startup.** This is not a correctness issue because charged timing remains per batch, but the plan should report it so total-runtime changes are not confused with charged throughput.

No candidate violates a hard constraint, smells like reward hacking, or retries a failed approach unchanged. DLB's evaluation and data-order issues are mandatory plan items rather than fatal flaws.

## Scored Verdict

| Idea | Evidence & reasoning | Potential impact |
|---|---|---|
| **Clean-Gated DLB** | **4/5** — only candidate with matched CIFAR-10 and CutMix-composition evidence; docked for dose dilution and the data-order confound. | **4/5** — highest ceiling after discounting; tail full coverage plus early partial coverage can plausibly clear +0.10. |
| **Front-Loaded HMix** | **3/5** — correct math and strongest attribution hygiene, but CIFAR-100-only evidence in a locally saturated mechanism family. | **2/5** — its central expected effect sits at the gate before transfer discounting. |
| **Identity-Initialized ECA** | **2/5** — excellent seed/RNG engineering but only indirect ImageNet evidence for a modified gate. | **2/5** — 18 weight-decayed parameters under a short horizon have a high probability of a null result. |

## Pick

**Clean-Gated Last-Mini-Batch Self-Distillation (DLB).** It is the sole candidate with evidence on the target dataset, composed with the parent's augmentation, at a magnitude large enough to remain above the 0.10-point gate after substantial discounting. HMix's expected effect already straddles the gate, and ECA's expected effect is closest to zero. Adopt DLB only with the sampler/cache verification burden and the three preregistered safeguards above.

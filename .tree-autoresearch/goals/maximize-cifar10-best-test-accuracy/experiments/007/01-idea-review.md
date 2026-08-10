# Claude Adversarial Idea Review - EXP-007

## Prioritized Feedback

1. **ASAM duty-cycle transfer is the main uncertainty.** Published 0.20-0.46-point ASAM-over-SAM gains come from full-run use, while EXP-007 would apply ASAM on about one eighth of all steps. Treat +0.30 as an exploratory target, not a literature-predicted sparse-dose effect. Audit adaptive and Euclidean perturbation norms at activation and preserve near-parent step exposure.
2. **EMA's stated limiter conflicts with the parent.** EXP-004 finished at its best 95.40%, so failed-child tail oscillation does not establish parent iterate variance. EMA also overlaps SAM, replaces the only late evaluation source, and averaging BatchNorm buffers is weaker than recalibration.
3. **ECA has the weakest transfer.** Its evidence is from much deeper ImageNet residual models, and 18 identity-started parameters may remain near identity while small-kernel launches reduce steps. Tighten any future latency gate substantially.
4. **ECA construction concern is already addressed.** The proposal uses standalone zero `nn.Parameter` tensors, not initialized `nn.Conv1d` modules, so it consumes no parent initialization RNG. This critic concern does not require a design change.
5. **Small gains remain unresolved.** Precommit that a gain below about +0.30 is formally judged by the 95.50 tree gate but is not persuasive mechanism evidence under observed 0.14-0.29-point variability.

## Scored Verdict

| Candidate | Evidence and Reasoning | Potential Impact |
|---|---:|---:|
| Literature-scale ASAM | 7/10 - closest dataset, architecture family, and validated component; discounted for sparse duty-cycle transfer and the combined radius/geometry package. | 6/10 - published ceiling reaches +0.30, with a bounded null-like downside because data and model-pass cadence remain fixed. |
| Time-constant EMA | 5/10 - real CIFAR averaging evidence and coherent horizon design, but the parent does not show the claimed tail oscillation and BatchNorm/evaluation ownership is risky. | 5/10 - low cost and plausible 0.1-0.4 upside, but SAM redundancy and EMA-only late evaluation create asymmetric regression risk. |
| Identity-centered ECA | 3/10 - elegant parent-equivalent design but weak transfer to a shallow tuned CIFAR WRN. | 4/10 - potentially additive, but likely small learned movement and nontrivial launch overhead. |

## Pick

**Literature-Scale ASAM in the Validated Clean Tail.** It asks whether the already-paid second pass uses the right perturbation geometry while keeping CutMix, independent images, cadence, evaluator, and optimizer-update count conceptually fixed. Refine the plan with a strict geometry audit, runtime exposure gate, honest sparse-dose attribution, and a preregistered +0.30 evidentiary bar.

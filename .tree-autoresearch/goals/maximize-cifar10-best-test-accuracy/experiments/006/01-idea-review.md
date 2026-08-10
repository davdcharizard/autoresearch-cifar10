# Claude Adversarial Idea Review — EXP-006

## Prioritized Feedback

1. **ASAM's full-run gain does not survive dose scaling cleanly.** The cited 0.20-0.30-point ASAM-over-SAM WRN gains use full-run ASAM. Applying only about 2,450 late pulses plausibly compresses the incremental effect to 0.05-0.10 points, at or below the gate and below EXP-003's 0.14-0.29 variability.
2. **ASAM rho 0.5 is doubly transferred.** It was tuned across the full learning-rate trajectory, whereas this proposal applies it only in the low-LR tail. A non-metric Euclidean perturbation safety bound would be needed, but even correct calibration does not fix the low expected effect.
3. **The initial 50/50 CutMix/manifold split cannibalizes too much validated CutMix.** Preserve 75% of selected mixing as CutMix and allocate 25% to manifold mixup. Marginal early probabilities become 0.50 clean, 0.375 CutMix, and 0.0625 at each hidden boundary.
4. **Manifold evidence remains transferable after discounting.** Direct CIFAR gains come from weaker baselines and longer schedules, but even retaining 20-30% of the reported 1.0-1.9-point improvement leaves 0.2-0.5 points. The hybrid does not reproduce the paper's strongest `{0,1,2}` policy because its input-space method is CutMix rather than linear Mixup; this caveat must be explicit.
5. **Compact ResNeXt stacks too many extrapolations.** It is smaller than the parent, far below the paper's 34M-parameter scale, uses narrower groups, and changes block type. The frozen WRN-tuned optimizer makes a negative-centered outcome likely even if its throughput microbenchmark passes.
6. **The acceptance gate is below observed variability.** A single-run candidate needs a plausible true effect around 0.3 points, not merely 0.10. Manifold mixup is the only finalist whose evidence-discounted ceiling comfortably meets that standard.

No finalist violates a hard constraint or repeats a failed experiment unchanged. All three proposals have adequate scope and integrity controls; expected effect size decides the pick.

## Scored Verdict

| Idea | Evidence & reasoning | Potential impact |
|---|---|---|
| **Adaptive clean-tail ASAM** | **6/10** — direct mechanism match, but full-run evidence and missing dose arithmetic weaken transfer. | **3/10** — plausible 0.05-0.10-point effect is inside noise and may not clear the gate. |
| **Shared-budget CutMix + manifold mixup** | **6/10** — strong direct CIFAR evidence and exact preservation of images/steps, discounted for weak-baseline transfer and CutMix cannibalization. | **7/10** — discounted 0.2-0.5-point effect is the only ceiling above both gate and noise band. |
| **Compact PreAct ResNeXt** | **3/10** — four stacked extrapolations from much larger matched-complexity evidence. | **6/10** — high raw architecture ceiling, but likely optimization mismatch under the frozen parent recipe. |

## Pick

**Shared-Budget CutMix and Manifold Mixup**, refined to a 75/25 split among selected mixed batches. It preserves EXP-005's central lesson, adds no model pass, leaves half of early batches clean, retains most validated CutMix exposure, and has the only defensible effect ceiling above the goal's practical noise floor.

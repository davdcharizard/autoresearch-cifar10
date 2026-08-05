**Prioritized Feedback**

1. **CutMix: LS=0.2 is the largest hidden risk.** idea-01 explicitly admits CutMix and label smoothing stack soft targets (`idea-01.md:95-104`), while idea-02 independently argues LS=0.2 is stale after EXP-008 (`idea-02.md:18`). Holding LS=0.2 for attribution may test a deliberately weakened CutMix. If maximizing accuracy is the goal, pre-register a CutMix+LS0.1 companion or use LS0.1 only on CutMix batches.

2. **CutMix: the proposed helper may not be throughput-free as written.** `torch.randint(..., device=inputs.device).item()` for bbox centers (`idea-01.md:46-47`) forces CUDA scalar syncs inside the timed step; the current loop counts step time and synchronizes at `train.py:312`. Generate bbox scalars on CPU with the already-seeded torch CPU RNG or Python integers, then verify `num_epochs/num_steps` against EXP-008.

3. **SAM: the under-anneal threshold is too forgiving.** The proposal treats ~133 epochs as safely above a ~110 threshold (`idea-04.md:196-203`), but learnings show EXP-005 lost at 131 vs 142 epochs and EXP-007 failed at 94 (`03-experiment-learnings.md:56-58`). Tail-gated SAM may spend the whole +0.1pp target on fewer low-LR updates. Consider a later gate, periodic tail SAM, or a same-session baseline before trusting a near miss.

4. **SAM: perturbing every trainable parameter is under-argued.** `sam_params` includes BN affine params and `GatedResidual.alpha` (`idea-04.md:89-91`, `train.py:134`), even though idea-02 identifies those 1-D params as special and harmful to decay (`idea-02.md:14-16`). A SAM perturbation through the ReZero gate could test gate fragility rather than useful flatness. Safer first variant: perturb only conv/fc weights or at least log alpha behavior.

5. **SAM: tail-only SAM is plausible but weakly evidenced.** The cited SAM result supports full-training flatness optimization, not “last 25% only” (`idea-04.md:170-183`). The proposal’s own prior is only 25-30% to clear the bar (`idea-04.md:281-288`), and EXP-010 says optimizer-like changes already tie SGD (`03-experiment-learnings.md:60-62`). Keep the kill criterion; do not let a sub-baseline run become a rho sweep.

6. **Recipe-scalar refresh: the sweep design is not the cleanest test.** If LS retune is the highest-prior stale scalar, the table should include LS0.1-only; instead B bundles WD-shaping+LS and C adds an 8e-4 conv-WD confound (`idea-02.md:83-91`). Replace C with LS-only, or run A/B plus same-session baseline. Otherwise a small win is hard to attribute.

7. **Recipe-scalar refresh: expected effect sits on the noise floor.** The proposal estimates each knob at 0.05-0.15pp and the bundle at 0.1-0.25pp (`idea-02.md:104-109`), while the benchmark has ~0.1pp run-to-run noise (`03-experiment-learnings.md:32-34`). This is a reasonable maintenance sweep, not the strongest single bet unless the loop allows multiple matched runs.

8. **No fatal hard-constraint violation found.** All three can stay inside `train.py`, use no new deps, keep one eval per epoch, and avoid seed hacking. The main risks are measurement ambiguity and method dilution, not constraint breach.

**Scored Verdict**

- **CutMix:** Evidence/reasoning **8/10**: best aligned with the proven EXP-008 throughput-free augmentation lever and supported by CutMix literature, but direct evidence on this already heavily regularized recipe is missing. Potential impact **7/10**: credible +0.10-0.30pp if LS/throughput issues are handled.

- **Recipe-scalar refresh / WD-shaping+LS:** Evidence/reasoning **6.5/10**: mechanistically sensible and code-specific, especially ReZero alpha no-decay, but the proposal admits mixed literature and sub-noise individual effects. Potential impact **5/10**: low downside, but likely a few hundredths unless the stale-LS hypothesis is exactly right.

- **Tail-gated SAM:** Evidence/reasoning **5.5/10**: SAM is real, but this tail-gated variant is mostly extrapolation and fights the known low-LR-step-count limiter. Potential impact **6.5/10**: upside is real if flatness is still untapped, but diluted by fewer tail updates and EXP-010’s optimizer-axis null.

**Pick: CutMix data-mixing regularization.** It attacks the diagnosed limiter most directly, is untried, has the strongest external and local support, and avoids the known capacity/epoch trap. Run it only after fixing the CUDA scalar sync and treating LS=0.2 as the main design risk, not an afterthought.

External sources checked: CutMix https://arxiv.org/abs/1905.04899, SAM https://arxiv.org/abs/2010.01412, Bag of Tricks https://arxiv.org/abs/1812.01187.

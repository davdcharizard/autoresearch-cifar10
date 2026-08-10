# Claude Adversarial Review — EXP-014 Finalists

Claude Opus independently recomputed proposal arithmetic from `train.py` and the experiment reports. It used read-only tools with no fallback model.

## Prioritized Feedback

1. **Uniform SWA's parent-kernel statistics are wrong.** EXP-011 copy-initializes the first EMA state, then applies exponential lerps. The actual oldest-state weight is about `6.30%`, not `0.12%`; newest weight is about `1.72%`, ESS about `79.2`, and mean age about 25 seconds. This makes removal of the stale anchor more interesting, but invalidates the proposal's quantitative story. A bias-corrected EMA would isolate the anchor more cleanly than swapping the whole kernel.
2. **Uniform averaging of BatchNorm buffers is not neutral.** The tail LR falls by about 17x, so early high-LR running statistics receive much more influence under SWA. Parameter-uniform/buffer-EMA or latest-buffer semantics would isolate the parameter kernel better; at minimum BN distance and variance-ratio diagnostics are mandatory.
3. **Positive Poly-1 upweights already-correct examples.** The exact hard multiplier `1+0.25*p_y` approaches 1.25 on confident examples and 1.0 on residual errors, which may be the wrong direction for boundary error. A positive result must be interpreted against an effective-LR/reweighting null.
4. **Poly-1 epsilon 0.25 has no external accuracy anchor.** It is justified only by an optimizer-safety ceiling, while the cited paper says coefficients are task-dependent. A null falsifies only this operating point.
5. **Soft-target Poly-1 partly undoes CutMix.** Clipped boxes bias adjusted lambda toward the original label; positive epsilon sharpens unequal mixed targets toward that majority, potentially weakening the lineage's `+0.61`-point mechanism. Hard-target-only Poly-1 would preserve CutMix semantics better.
6. **The width proposal has not established capacity limitation.** The only durable train-fit observation is EXP-001's near-zero late loss, which weakens a raw-capacity story. Add accuracy-blind train-fit diagnostics, but do not overstate them when no contemporaneous full parent run exists.
7. **Width reduces max-selection opportunities.** At the `1.15x` gate, about 115 evaluations are expected versus the parent's 133, so the candidate needs extra plateau lift to overcome fewer draws. Dose evidence is mixed: EXP-010 gained no accuracy from `+9.3%` steps, while EXP-012's smaller dose miss limited causal interpretation.
8. **The width latency gate is tighter than its MAC ratio.** Exact candidate MAC ratio is `1.1756045x`, while the gate is `1.15x`; low-resolution H20 efficiency makes a pass plausible but not assured. Claude suggested a preregistered width-288 fallback, which the main agent rejects because it would convert one isolated package into a timing-selected architecture sweep. A valid width-320 gate failure will be recorded without fallback.
9. **Width changes relative SAM dose.** Global `rho=0.05` stays fixed while parameter norm grows, so preflight should report `||epsilon||/||w||` for both arms without retuning rho.
10. **SWA replaces a validated mechanism.** The once-per-epoch limit prevents same-run EMA/SWA comparison; bias-correction or hybrid buffer semantics would retain more validated machinery.
11. **SWA's 7.5% parent-drift gate is too loose.** Use the sibling-standard `<=3%` if this proposal is revisited.
12. **All scientific tail bars were too low.** Keep `95.71%` as the formal goal threshold, but require final-16 mean at least `95.69%` as the scientific plateau readout and report max-minus-tail-mean premium.
13. **The group-equivariant paper supports the only moonshot representation idea, but none of the finalists implements it.** Width is therefore the only finalist with a capacity/representation ceiling; Poly-1 and SWA are tail-polish bets.

## Confirmed Strengths

- No finalist violates the goal's file, GPU, seed, budget, dependency, or evaluation constraints.
- None retries a failed approach unchanged. EXP-010 rejected block relocation, not added final-stage width.
- The width proposal's parameter/MAC arithmetic is exact: `3,827,290` parameters, `461,556,864` MACs/image, and `1.1756045x` parent MAC ratio.
- The Poly-1 hard/CutMix gradient calculus and its dose arithmetic are exact.

## Scored Verdict

| Idea | Evidence & reasoning | Potential impact |
|---|---:|---:|
| Calibrated Stage-3 Width-5 Expansion | **7/10** — exact systems arithmetic, a historically productive width lever, and defensible low-resolution allocation; capacity limitation remains unverified. | **8/10** — the only finalist whose plausible ceiling reaches the diagnosed `0.25-0.30` stable lift, despite fewer steps/evaluations and weaker relative SAM dose. |
| Uniform Full-State Clean-Tail SWA | **5/10** — targets a real stale-anchor defect, but its kernel statistics are wrong and its BN treatment is weak. | **5/10** — removing a 6.3% anchor and roughly doubling ESS is real, but literature suggests mild generalization gains near the noise floor. |
| Bounded Soft-Target Poly-1 | **5/10** — exact derivation and clean implementation, but epsilon has no effect-size evidence and positive weighting may work against the limiter and CutMix. | **4/10** — preserves dose, yet likely acts as mild tail rescaling and has less credible upside than the required plateau lift. |

## Pick

**Calibrated Stage-3 Width-5 Expansion.** It is the only finalist with a ceiling commensurate with the diagnosed stable-gain requirement. Its arithmetic and integration surface are clean, and a null is informative if train-fit, dose, evaluation-count, and relative-SAM diagnostics are reported honestly. Claude conditioned execution on a latency fallback, train-fit diagnostics, relative SAM perturbation reporting, and a `95.69%` tail-mean scientific bar. The fallback is not adopted for isolation reasons; the other diagnostic and scientific-bar conditions are adopted with the limitation that no second full parent metric run is allowed.

## Follow-up Review

- **Verdict**: PASS
- Claude confirmed the capacity premise is scoped as unverified, arithmetic is exact, `95.69%` is separated from the formal `95.71%` gate, evaluation-count/max-premium and relative-SAM diagnostics are report-only, and width 320 is fixed with no conditional resize.
- It judged rejection of the width-288 fallback methodologically coherent: a timing-selected fallback would weaken package falsifiability, while width 288 remains available as a separately preregistered future experiment.

# Claude Adversarial Result Review - EXP-006

## Verdict Audit

- `no-improvement` is correct: parent 95.40% plus the 0.10-point gate requires 95.50%; observed best/final accuracy was 95.41%.
- The +0.01-point delta is below the run's own tail variation: the final four evaluations span 95.26-95.41%, so it is not evidence of improvement.
- Raw log values reconcile exactly with the execution record. Policy dose, Beta moments, boundary split, SAM cadence, runtime, evaluations, steps, and parameter count are plausible and within preregistered bounds.

## Integrity Findings

1. The `late=0` counter is vacuous: `mix_boundary` is reset to `None` before the cutoff branch and can only be assigned inside it, so the late counter cannot fire. The cutoff is structurally enforced but not independently measured by that counter.
2. Wall-clock progress means fixed seeds do not fix step count, phase boundaries, or per-step learning rates. The child ran 25,644 steps versus 25,560 and received 2,488 SAM pulses versus 2,449, so the nominal +0.01 cannot be assigned to manifold mixing.
3. The frozen protocol's max over 132 uncharged test evaluations inflates small best-accuracy deltas. This is a goal-level limitation shared with the parent, not an EXP-006 validity violation.
4. RNG parity, private generator separation, discarded parent specs, channels-last restoration, label pairing, and Beta(2,2) shape all check out.
5. Hidden mixing occurs under BF16 autocast rather than FP32. This is a low-severity fidelity difference, unlikely to explain the null result.

## Mechanism Diagnosis

- EXP-006 is a substitution experiment: approximately 2,557 parent CutMix batches were replaced by 2,561 manifold batches while selected/clean exposure, image stream, forward count, and total work stayed nearly fixed. The two marginal effects were indistinguishable in accuracy at this dose.
- Final test loss worsened from 0.1654 to 0.1749, far beyond the child's 0.0011 final-four-epoch loss range, while accuracy was flat within noise. Replacing CutMix's input-space, area-adjusted soft-label dose with hidden interpolation did not preserve confidence/NLL behavior.
- Possible contributors are lower effective soft-label pressure and downstream BatchNorm statistics updated on interpolated hidden features absent at test time.
- Discredited: this specific 75/25 CutMix-to-manifold reallocation and the premise that validated CutMix exposure is cheap funding for a new augmentation.
- Not discredited: manifold mixup generally, because its realized marginal dose was only about 10% of all steps and the design measures a difference between mechanisms rather than its isolated effect.

## Safe Learning

- Sub-0.30-point candidates are difficult to distinguish under this single-run max-test-accuracy protocol; EXP-006 adds a 0.15-point within-run tail spread to EXP-003's confirmation reversals.
- Substitution experiments are low-information at this gate because they identify only the difference between the removed and added mechanisms. Prefer additive or orthogonal changes that retain validated mechanisms at full dose.
- Report final test loss and exact tail-phase dose as first-class diagnostics when wall-clock scheduling makes fixed-seed step exposure variable.
- Verification counters must have a reachable failure path; otherwise present the invariant as a structural code argument rather than measured evidence.

## Next Experiment

- Recommended: additive model-weight EMA, preregistered without scalar search, to average the visible late-iterate oscillation while leaving full CutMix and SAM exposure intact. Specify decay and BatchNorm handling before execution and require an effect near 0.3 points to treat a single run as persuasive.
- Runner-up: apply manifold mixing additively to a fraction of the parent's clean half while retaining full CutMix, which isolates manifold's contribution but risks excess regularization given the loss regression.

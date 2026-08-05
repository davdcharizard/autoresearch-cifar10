# Plan Critique - EXP-035

Cross-model review was intentionally unavailable because this session is
strictly offline/local. The required fallback local plan critic reviewed the
goal, brainstorm, full proposal, `TASK.md`, accepted source, and written plan.

1. **[Verification Procedure 10] Valid transition ordering can be falsely classified as a crash.** Mixup is checked before each step, while RandAugment is disabled after iterator exhaustion. If the epoch's final batch crosses 195 counted seconds, RandAugment can legitimately log first and mixup can log at the next epoch with the same completed-step count. Accept both this boundary ordering and the usual mixup-first ordering, while validating times, steps, and epoch boundary semantics.

2. **[Verification Procedure 7-8] The timing estimator needs exact window and counterbalancing definitions.** Three alternating windows cannot fully counterbalance arm order. Define each window as synchronized elapsed time divided by measured steps, use four windows with `accepted/candidate, candidate/accepted` repeated, and compute the CV and regime medians over those four window means. Give each pair a distinct preregistered deterministic CUDA fixture/RNG state.

3. **[Verification Procedure 10] The final evaluation need not be a distinct partial-epoch call.** A valid run can end at iterator exhaustion or on an epoch divisible by five. Validate the evaluation epoch set as every fifth epoch union the final epoch, with at most one call per epoch, rather than requiring a distinct extra final evaluation.

## Verdict

**Revise, then proceed.** The one-line treatment, distribution bounds,
fixed-time retention formula, source-scope audit, sole-score rule, and 94.42%
decision threshold are otherwise sound.

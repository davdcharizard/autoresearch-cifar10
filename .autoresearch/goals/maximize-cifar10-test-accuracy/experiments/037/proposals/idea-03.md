# Proposal: Reject Direct/Residual Logit Consistency

## Verdict

Do not advance this idea to planning or a scored experiment. There is no
defensible single coefficient, temporal window, or teacher direction under the
one-fixed-seed protocol, and the intervention directly suppresses the only new
mechanism with positive local evidence.

## Proposed Mechanism

The accepted `a7c42dc` model forms pooled features `p`, residual features
`r = 0.1 * pooled_head(p)`, and scored logits `z = fc(p + r)`. A consistency
regularizer could obtain direct logits `z_direct = fc(p)` from the same pooled
tensor, so it needs no second backbone forward. A detached-teacher version
would add `lambda * KL(softmax(z_direct.detach()) || softmax(z))`; symmetric KL
or squared centered-logit distance are nearby variants.

This is cheap, but its exact action is to penalize the class-dependent logit
correction `z - z_direct = fc.weight @ r` (the classifier bias cancels). EXP-036
showed that this correction was initially nondegenerate (logit RMS perturbation
`0.069719`, residual/direct feature-norm ratio `0.120864`) and then improved
best accuracy from 94.32% to 94.48% and final loss from 0.2523 to 0.2456 at
130.304 passes. The direct path is therefore a superseded component, not an
independently validated teacher.

## Why No Exact Treatment Is Defensible

- `lambda` has no locally measured scale. KL magnitude changes with confidence
  over the cosine schedule and across the mixup-to-hard transition, so generic
  values such as 0.1 or 1.0 do not define a stable pressure relative to CE.
- An early-only window would constrain the very co-adaptation by which the new
  head learns while mixup and RandAugment are active. The successful run gives
  no evidence that this co-adaptation is unstable.
- A hard-tail-only or late-only window would pull the accepted classifier back
  toward the old direct boundary precisely when hard labels refine top-1.
- Detaching `z_direct` arbitrarily privileges an unvalidated teacher. Not
  detaching it lets both paths move together and adds poorly attributable
  gradients to the shared classifier and backbone. Symmetric KL does not solve
  that identification problem.
- Temperature, centered logits, MSE versus KL, and confidence masking are each
  additional uncalibrated choices. Selecting among them after observing a run
  would become forbidden adjacent head tuning.

## Decision Contract

Make no production change and spend no score on this candidate. A future
proposal may reconsider consistency only after a preregistered, training-only
diagnostic establishes a concrete failure mode, such as growing direct/full
disagreement coincident with worse training-objective generalization proxies,
and derives both coefficient and window without test-set feedback. Diagnostics
must not inspect evaluator data or alter the accepted training trajectory.

This rejection closes generic direct/full-logit KL, symmetric KL, and
centered-logit MSE as unsupported immediate add-ons to `a7c42dc`; it does not
close consistency mechanisms backed by a genuinely independent teacher or
augmentation-paired views, which would be distinct proposals and may incur
additional backbone compute.

## Constraints and Evidence

- Preserve accepted commit `a7c42dc`, the exact scale-0.1 bias-free
  `128 -> 64 -> 128` pooled residual head, and all optimizer/data/evaluator
  behavior.
- Primary threshold for any future scored successor is 94.58%, at least 0.10
  points above the accepted 94.48% baseline.
- Sources: `experiments/036/04-analysis.md`, `experiments/036/02-plan.md`,
  `03-experiment-learnings.md`, `02-system-understanding.md`, and accepted
  `train.py`.

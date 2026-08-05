# Brainstorm EXP-003
**Created**: 2026-07-24

## Web Search & Literature Review

This local-only loop reused the goal's curated literature rather than performing network retrieval.

- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): a carefully selected late averaging window can improve generalization cheaply, but early or nonstationary iterates can bias the result.
- **mixup** (`knowledge/papers/mixup.md`): convex image and target interpolation improves CIFAR generalization with one forward pass; EXP-002 now confirms the mechanism on this exact baseline.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can retain its benefit after removal, supporting the validated hard-label tail and other time-gated regularizers.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): width can be a compute-efficient CIFAR capacity lever, while the current H20 memory result leaves substantial headroom.
- **RandAugment** (`knowledge/papers/randaugment.md`): diverse image transformations can improve CIFAR accuracy, though CPU transform overhead is risky in this fixed-time, fast-GPU regime.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild target smoothing combats overconfidence, but it overlaps mechanistically with the already-successful mixup phase.

## Experimental History Review

- The initial baseline reached 91.54%. EXP-001 established WRN-16-2, batch 256, selective weight decay, Nesterov SGD, persistent workers, and a time-aligned cosine schedule, improving the metric to 93.38% in about 146 data passes.
- EXP-002 retained that baseline and added alpha-0.2 mixup for the first 65% of counted time. It improved 93.38% to 94.07%, ended at its best checkpoint, and preserved 141.9 data passes, validating early regularization plus a long clean-label tail.
- No accepted approach has failed yet. The main protocol failure was non-persistent loader workers, which caused a wall-time timeout despite valid training progress; persistent workers and sparse evaluation must remain fixed.
- Still untried on the current baseline are spatial mixed-sample regularization, late parameter averaging, regularized extra width, canonical channel standardization, block dropout, Cutout, and controlled mixup-strength/cutoff refinement.

## Objective Diagnosis

The current model remains generalization-limited: after mixup switches off, its smoothed hard-label loss falls near zero while test accuracy converges around 94%, and the final evaluation equals the best. More of the same low-LR fitting is therefore unlikely to yield a large gain. EXP-002 also shows that a scheduled regularizer can move the ceiling without materially sacrificing exposure. The best next ideas either improve the geometry of the final solution, introduce complementary spatial invariance, or add capacity only together with the regularization that now controls memorization. The acceptance threshold is 94.17%, so an intervention needs more than a noise-scale checkpoint fluctuation.

## Collected Ideas

- **Short-window late EMA** — maintain a close-tracking parameter average only in the final portion of the hard-label tail and evaluate that view at the existing cadence. It targets residual solution variance at negligible compute cost, but must avoid the trajectory lag and BN mismatch identified in EXP-002's prior review.
- **Early CutMix with a clean tail** — replace early convex whole-image interpolation with rectangular patch replacement and area-corrected mixed targets, then retain the final 35% hard-label phase. It targets spatially localized evidence and occlusion robustness while preserving one forward pass.
- **Regularized WRN-16-3** — widen the model one step while retaining the validated early mixup schedule, using a throughput-aware batch and LR. Unlike the rejected unregularized width idea, this tests whether mixup can unlock useful capacity without simply widening the train/test gap.
- **Canonical per-channel standardization** — replace the current unit standard deviations with CIFAR-10 channel standard deviations while holding the model and training path fixed. This simplifies input conditioning and may improve optimizer balance, but first-layer BN makes the likely effect modest and direct evidence here is weak.
- **Time-gated Cutout** — apply a small GPU-side square mask during the same early critical period as mixup, either instead of or probabilistically alongside it, and disable both for the hard tail. This imports spatial occlusion invariance with little arithmetic but risks over-regularizing already mixed images.
- **WRN block dropout** — add the dropout placement used in wide residual networks between block convolutions, active only during the early/middle phase. It targets feature co-adaptation rather than label interpolation, though stochastic activations may slow convergence in the 300-second budget.
- **Mixup schedule refinement** — retain the successful algorithm and tune one exposed value, such as alpha 0.1 or a 50% cutoff. This is the cleanest attribution and lowest implementation risk, but a single manual point has smaller expected upside than a qualitatively new mechanism.
- **Stochastic-depth moonshot** — randomly bypass residual branches early and progressively restore the full model before the hard-label tail. It could regularize an implicit ensemble at low average compute, but WRN-16-2 is shallow and branch dropping may remove too much representation depth.

## Combinations

- **Early mixup + late EMA**: mixup shapes a smoother representation during high-LR learning, while a short-window EMA could stabilize the final hard-label basin. The mechanisms are complementary, but an EMA-only evaluation can conceal the live model's peak and weaken attribution.
- **WRN-16-3 + validated mixup**: extra capacity raises the representation ceiling while the already-proven mixup schedule counters the wider model's memorization risk. This is materially stronger than the previously reviewed unregularized width proposal, though reduced exposure remains the main downside.
- **Early CutMix + hard-label tail**: preserve the successful temporal regularization recipe but substitute localized patch mixing for global interpolation. The combination may learn stronger spatial evidence than either ordinary crop/flip or whole-image mixup while retaining late clean-label calibration.

## Candidate Ideas

### Early-Only CutMix With an Area-Corrected Hard-Label Tail
**Summary**: Replace mixup during the first 65% of counted time with one shared rectangular CutMix patch per batch using `Beta(1,1)`, a device-local permutation, and a target coefficient recomputed from the actual clipped patch area. Preserve the validated final 35% hard-label path and every other EXP-002 setting.

**What it targets**: The remaining spatial generalization gap by retaining natural pixels while discouraging reliance on one contiguous discriminative region.

**Reasoning**: EXP-002 validates mixed targets and early removal on this model. CutMix transfers that successful temporal recipe to a localized composition mechanism with one forward pass and similar tensor overhead, potentially learning stronger occlusion and localization invariance than dense pixel interpolation.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002 report.

**Estimated Effort**: medium

**Risk Assessment**: The local literature does not directly validate CutMix, and area fraction can be a poor semantic-label proxy on 32x32 images. Incorrect clipping or aliased in-place assignment would invalidate the implementation; throughput and exact area correction need explicit tests.

### Short-Horizon Post-Mixup EMA
**Summary**: Preserve the complete EXP-002 path, initialize FP32 parameter EMA shadows at 75% counted time, update every ten steps with decay 0.99, and use the EMA view only for the budget-exhausted final evaluation. Ordinary sparse evaluations remain live, retaining their best scores while giving one endpoint-smoothed model a valid final measurement.

**What it targets**: Residual minibatch-order and SGD-iterate noise around the mature post-mixup solution, while limiting the lag and BN mismatch risks identified for the earlier EMA design.

**Reasoning**: Averaging literature supports carefully windowed late averages at low cost. A roughly 1,000-step effective horizon is short enough to follow EXP-002's improving tail, and final-only EMA evaluation preserves the compatible live-model history under the once-per-epoch limit.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/weight-averaging.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002 report and prior idea review.

**Estimated Effort**: medium

**Risk Assessment**: The cosine endpoint may already suppress nearly all useful iterate variance, so upside could be below 0.10 points. Live BN buffers may mismatch averaged affine parameters, and safe parameter swap/restoration requires careful smoke testing.

### Mixup-Regularized WRN-16-3 With Batch-Scaled SGD
**Summary**: Widen the accepted model to WRN-16-3, use batch 384 and a linearly scaled 0.3-to-0.003 LR range, and retain alpha-0.2 mixup through 65% plus the hard-label tail. A matched synthetic gate requires at least 80 projected passes and 60% of WRN-16-2 image throughput before the full run.

**What it targets**: Representation capacity after the validated mixup phase changes the width tradeoff by controlling memorization; the H20's roughly 97 GiB memory leaves capacity headroom, while the gate bounds compute loss.

**Reasoning**: WRN width already produced the largest gain in EXP-001, and mixup added another 0.69 points in EXP-002. Combining the two addresses the earlier review's fatal objection to unregularized width, with batch scaling intended to improve H20 utilization.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/wide-residual-networks.md`; `knowledge/papers/mixup.md`; EXP-001 and EXP-002 reports; project insights.

**Estimated Effort**: low

**Risk Assessment**: Width increases convolutional work roughly quadratically and may yield too few updates or data passes; batch 384 can also reduce useful SGD noise. The LR scaling and lower exposure could swamp any capacity gain despite ample memory.

## Review

The reviewer selected early-only CutMix because it targets generalization without sacrificing the update and exposure budget, while changing only the early regularizer. Significant concerns adopted from `01-idea-review.md`: local evidence supports mixed targets and temporal removal but not CutMix directly; on 32x32 images, pasted area can be a weak proxy for semantic content; and replacing the validated mixup path creates a regression risk. The experiment therefore keeps the proven 65/35 schedule, recomputes target mass from exact clipped area, logs mean pasted fraction, tests donor alias safety, and pre-registers a null as a clean return to mixup rather than evidence against early regularization. The suggested mixup/CutMix Bernoulli mixture was not adopted because it would weaken attribution in this first CutMix test.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`: **Early-Only CutMix With an Area-Corrected Hard-Label Tail** scored 6/10 for both evidence/reasoning and potential impact and was the recommended balance of mechanism fit, exposure preservation, and attribution. Mixup-regularized WRN-16-3 had the highest theoretical impact score but was downgraded for its large update tax and four coupled changes. Short-horizon EMA was judged unlikely to clear the threshold because the low-LR endpoint already exhibits little useful iterate variance.

## Chosen Idea
**Selected**: Early-Only CutMix With an Area-Corrected Hard-Label Tail

**Why this idea**:
EXP-002 proves that early mixed-target regularization plus a 35% clean tail improves this exact WRN. CutMix tests a complementary spatial inductive bias while retaining one forward pass, essentially all image exposure, the accepted architecture and optimizer, and a clean one-variable comparison against mixup. Its main risks are measurable through exact area accounting, deterministic pixel tests, throughput comparison, and the final hard-label recovery curve.

**Hypothesis**:
Replacing alpha-0.2 mixup with area-corrected `Beta(1,1)` CutMix during the first 65% of counted training, then returning to the unchanged hard-label path, will raise `best_test_acc` from 94.07% to at least 94.17% while retaining at least 95% of EXP-002's matched synthetic throughput and full wall-time compliance.

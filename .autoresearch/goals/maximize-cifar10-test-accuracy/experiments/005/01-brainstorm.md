# Brainstorm EXP-005
**Created**: 2026-07-24

## Web Search & Literature Review

This local-only quick pass reused the curated mixup, regularization-timing, and WRN sources.

- **mixup** (`knowledge/papers/mixup.md`): whole-image interpolation is the only added generalizer that has improved this exact WRN.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization may be removed late, but EXP-004 now bounds the useful local critical period beyond 50%.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): width remains a higher-capacity alternative if mixup timing saturates.

## Experimental History Review

- EXP-001 established the 93.38% WRN-16-2 baseline; EXP-002 reached the current 94.07% with alpha-0.2 mixup through 65% and a 35% hard-label tail.
- EXP-003 replaced mixup with shared CutMix and fell to 93.72% at normal exposure, favoring whole-image interpolation.
- EXP-004 moved the mixup cutoff to 50%, gained 0.9% exposure, but fell to 93.91% with worse final test loss. The 50-65% mixed-target window is therefore useful and the pre-registered opposite 75% arm is the clean next timing probe.
- Mixup alpha above 0.2 and mixup-regularized width 3 remain untested. Alpha below 0.2 and late EMA have weaker mechanistic support.

## Objective Diagnosis

Generalization remains limiting, and the timing evidence is now directional: shortening the accepted regularization window hurts even when it increases exposure. The immediate question is whether the 65% cutoff is near optimal or whether keeping alpha-0.2 mixup through 75% further improves representation quality while leaving 75 seconds for margin recovery. Stronger alpha and regularized width offer distinct alternatives but change regularization severity or compute exposure rather than isolating duration.

## Collected Ideas

Quick pass; omitted.

## Combinations

Quick pass; omitted.

## Candidate Ideas

### Stronger Alpha-0.4 Mixup at 65%
**Summary**: Retain the validated 65% cutoff and change only `MIXUP_ALPHA` from 0.2 to 0.4, making interpolation less endpoint-heavy and increasing the frequency of materially mixed examples.

**What it targets**: Insufficient regularization severity while preserving the accepted timing balance.

**Reasoning**: Alpha 0.2 improved the non-mixup baseline by 0.69 points, and EXP-004 indicates less duration is harmful. A moderate upward alpha move is mechanistically coherent and isolates strength from duration.

**Sources**: `knowledge/papers/mixup.md`; EXP-002 and EXP-004 reports; EXP-004 idea review.

**Estimated Effort**: low

**Risk Assessment**: Stronger interpolation can slow representation fitting or leave margins too soft for the 105-second tail. There is no local evidence alpha 0.2 is under-strength, so the direction remains exploratory.

### Mixup-Regularized WRN-16-3
**Summary**: Increase width to 3 with batch 384 and batch-scaled LR while retaining the accepted alpha-0.2, 65/35 mixup schedule. Require a strict matched throughput gate before the full run.

**What it targets**: Representation capacity once mixup controls the wider model's overfitting risk; the H20 has over 96 GiB unused memory.

**Reasoning**: Width and mixup are the two validated mechanisms behind the current baseline. Their combination answers the earlier fatal objection to unregularized width and offers more upside than another small timing adjustment.

**Sources**: `experiments/003/proposals/idea-03.md`; `knowledge/papers/wide-residual-networks.md`; `knowledge/papers/mixup.md`; EXP-001/002 reports.

**Estimated Effort**: low

**Risk Assessment**: Width roughly doubles convolutional work, sharply reducing updates and exposure; batch/LR changes confound attribution and may hurt generalization despite higher capacity.

### Later 75% Mixup Cutoff
**Summary**: Preserve every accepted EXP-002 setting but change `MIXUP_END_FRACTION` from 0.65 to 0.75, extending alpha-0.2 interpolation by 30 seconds and leaving a 75-second hard-label cosine tail.

**What it targets**: The remaining generalization gap and the local evidence that the 50-65% regularization window is valuable.

**Reasoning**: EXP-004 is a clean directional result: more hard-label exposure did not replace mixup. The opposite 75% arm tests whether that trend continues without changing strength, architecture, or optimizer, and the retained tail still spans roughly 35 epochs at low LR.

**Sources**: EXP-002 and EXP-004 reports; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; goal learnings.

**Estimated Effort**: low

**Risk Assessment**: Seventy-five seconds may be insufficient for hard-label margins to recover from soft targets; accuracy can remain depressed at the budget boundary. The likely effect is modest and near the single-run noise scale.

## Review

The reviewer selected alpha 0.4 because it probes the only untested regularization axis while preserving the validated 65/35 timing and full exposure. Significant concerns adopted: there is no local evidence alpha 0.2 is under-strength, so this is a genuine two-sided exploratory test; the batchwise scalar means the change affects whole batches coherently; and a successful mechanism should reduce late test loss relative to EXP-002's 0.2432 while clearing the metric threshold. The 75% cutoff was rejected because it sacrifices a tail that was still improving, and width 3 was deferred because its batch/LR pairing and exposure loss confound the result.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`: **Stronger Alpha-0.4 Mixup at 65%** scored 6/10 for evidence/reasoning and potential impact. WRN-16-3 has a higher theoretical ceiling but a much larger probability of an exposure-confounded run. The later cutoff had the weakest support because EXP-004 does not imply mirror-symmetric benefit and EXP-002 still needed its clean tail.

## Chosen Idea
**Selected**: Stronger Alpha-0.4 Mixup at 65%

**Why this idea**:
This keeps the successful timing, architecture, optimizer, and throughput fixed while testing whether materially mixed examples are still too rare under the endpoint-heavy alpha-0.2 distribution. It is a single-constant intervention with direct attribution and no new runtime path.

**Hypothesis**:
Changing `MIXUP_ALPHA` from 0.2 to 0.4 while retaining the 65% cutoff will strengthen early interpolation, reduce final test loss below 0.2432, and raise `best_test_acc` from 94.07% to at least 94.17% without materially reducing exposure.

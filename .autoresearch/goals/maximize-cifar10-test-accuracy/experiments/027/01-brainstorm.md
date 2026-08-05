# Brainstorm EXP-027
**Created**: 2026-07-26

## Web Search & Literature Review

- **RandAugment** (`knowledge/papers/randaugment.md`): standard image-space transformations can add CIFAR invariances without model FLOPs; EXP-026 now supplies a local positive but insufficient signal.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early augmentation can shape generalization while later removal preserves terminal fitting, matching the worker-safe policy already validated locally.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): low-resolution transformation capacity is a plausible CIFAR lever; EXP-011 supplies the direct local depth result.

No network source was consulted; this offline quick pass uses the local knowledge base and completed experiment evidence.

## Experimental History Review

- The baseline remains 94.07%. Twenty-four rejected or feasibility-aborted follow-ups now show a stable plateau, but three distinct mechanisms moved directionally: full SE 94.16, extra final block 94.15, and early RandAugment 94.12.
- EXP-011's extra 8x8 block retained 132.92 passes and gained 0.08 points, but final loss worsened to 0.2782, indicating added capacity without enough generalization. EXP-026 retained 142.45 passes and gained 0.05 points under exact clean-tail RNG isolation, with 0.2574 loss and negligible counted cost.
- The composition has a concrete interaction rationale rather than an additive-score assumption: early image invariance may regularize the overconfident deeper tail, while the extra block supplies representation capacity missing from standalone RandAugment. RandAugment's CPU work does not reduce the extra block's counted GPU throughput, though wall time must be rechecked.
- Both exact standalone treatments are closed. A composition is justified only as one preregistered interaction, with EXP-011 model state and EXP-026 transform/RNG/cutoff semantics reproduced exactly; no depth, policy, seed, strength, or timing tuning is allowed.
- Batch 128 with a fully scaled LR curve remains the strongest standalone fallback but changes update granularity, BatchNorm groups, and mixup refresh frequency with weak direct evidence. Alpha 0.1 is the remaining isolated mixup-strength gap but has an adverse under-regularization prior.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2 -> 0.1`, retaining batch-shared sampling, the validated 65% cutoff, WRN-16-2, optimizer/schedule, seed, transforms, and evaluation.

**What it targets**: The unmeasured weaker side of mixup strength at the locally optimal duration.

**Reasoning**: Alpha 0.4 over-regularized, while duration 50/65/75 is bracketed. Alpha 0.1 cleanly completes the strength map with essentially unchanged throughput and no implementation ambiguity.

**Sources**: `knowledge/papers/mixup.md`; EXP-002/004/005/015/020; experiment learnings.

**Estimated Effort**: low

**Risk Assessment**: Beta(0.1,0.1) is more endpoint-heavy and likely under-regularizes; every prior mixup perturbation regressed. It is map completion with low expected ceiling.

### Batch-128 With a Proportionally Scaled LR Curve
**Summary**: Change batch 256 to 128, scale the complete LR curve from `0.2->0.002` to `0.1->0.001`, and double only the safety step cap. Retain accepted architecture, data, mixup, optimizer family, time schedule, seed, and evaluator. Gate on at least 120 projected passes and 46,875 projected updates.

**What it targets**: Finer optimizer-update granularity plus smaller BatchNorm and batch-shared mixup groups while holding approximate LR/batch noise scale constant.

**Reasoning**: This is the strongest remaining standalone operating-point change and keeps epoch example counts aligned because both batch sizes drop 80 images. It is distinct from failed exposure-only treatments, but lacks direct evidence that more lower-amplitude decisions improve the current boundary.

**Sources**: EXP-026 `proposals/idea-02.md` and `01-idea-review.md`; EXP-001/002; EXP-008/009/016; experiment learnings.

**Estimated Effort**: low

**Risk Assessment**: Halving peak and floor may under-update within 300 seconds, and BatchNorm/mixup grouping co-vary with update granularity. A null cannot isolate these effects and no LR or batch retry is allowed.

### Extra 8x8 Block Plus Fixed Early RandAugment
**Summary**: Reproduce EXP-011's exact `[2,2,3]` WRN with 987,098 parameters and EXP-026's exact worker-safe, RNG-isolated `RandAugment(num_ops=1,magnitude=5)` policy through the first exhausted epoch ending at or after 65%. Preserve accepted batch-256 FP32 SGD, alpha-0.2 mixup cutoff, seed, schedule, and evaluator. Require semantic identity to both component oracles, >=130 projected and realized passes, and <=500 projected wall seconds before one score.

**What it targets**: The complementary representation/generalization gap: EXP-011 added useful low-resolution transformation capacity but became overconfident, while EXP-026 added image invariance at no counted throughput cost but lacked enough representational gain.

**Reasoning**: The components independently gained +0.08 and +0.05 and act on different axes. The claim is not arithmetic additivity; it is that early invariance can improve the deeper model's high-loss generalization while the extra block retains its positive capacity signal. EXP-026's private worker RNG makes the data intervention attributable, and its CPU-only cost should leave EXP-011's 132.92-pass regime intact.

**Sources**: EXP-011 and EXP-026 plans/reports; `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; `knowledge/papers/wide-residual-networks.md`; protocol learnings.

**Estimated Effort**: high

**Risk Assessment**: Two individually sub-threshold treatments may interact negatively or merely add noise; both worsened final loss relative to accepted. Reconstructing exact model initialization plus worker RNG/cutoff semantics is demanding. A miss closes this exact composition and cannot trigger depth, policy, seed, cutoff, or exposure changes.

## Review

The reviewer selected the extra-block plus early-RandAugment composition at 3.5/5 evidence and 4/5 impact, while rejecting any claim that RandAugment is already shown to cure overconfidence. I adopt the honest interaction framing: success needs near-additive top-1 benefit despite both components' worse loss. Planning must reproduce both component oracles exactly, retain >=130 passes, and report whether final loss improves over EXP-011's 0.2782 as an informative mechanism check. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The composition is high variance but is the only candidate with two local positive signals on distinct axes and a plausible path above 94.17. Batch 128 remains the next fallback but has multiple inseparable operating-point effects; alpha 0.1 is low-ceiling map completion.

## Chosen Idea
**Selected**: Extra 8x8 Block Plus Fixed Early RandAugment

**Why this idea**:
EXP-011's extra low-resolution block reached 94.15% but generalized with high loss, while EXP-026's early image-invariance policy reached 94.12% without reducing counted exposure or perturbing accepted crop/flip RNG. Their mechanisms are distinct and operationally composable: the extra block supplies representation capacity, and worker-side RandAugment supplies early input diversity behind the slower GPU path. This is a one-shot interaction test, not permission to tune either closed component.

**Hypothesis**:
Exact composition of EXP-011's `[2,2,3]` model and EXP-026's fixed RNG-isolated early `N=1,M=5` RandAugment will retain at least 130 data passes and raise both fixed-seed best and final test accuracy from 94.07% to at least 94.17% if early invariance usefully shapes the deeper model and their top-1 benefits interact near-additively. Final loss below EXP-011's 0.2782 would support the intended interaction; higher loss would falsify that mechanism even if top-1 moves by noise.

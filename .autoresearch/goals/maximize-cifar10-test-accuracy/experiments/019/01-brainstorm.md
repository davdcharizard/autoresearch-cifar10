# Brainstorm EXP-019
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): retain the accepted WRN backbone and both high-resolution transforms; local architecture exchanges have consistently hurt.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): fixed-time interventions must account for when compute is spent, supporting removal of observational work that reduces training updates.
- **EXP-017 and EXP-018 reports** (`experiments/017/04-analysis.md`, `experiments/018/04-analysis.md`): two gates with diagnostics scored 94.16 at 133.64 passes, while final-only without diagnostics recovered 137.83 passes but fell to 93.67.

No network source was consulted; this offline loop uses the persistent local knowledge base and completed experiment evidence.

## Experimental History Review

- The accepted 94.07% WRN plus early batch-shared mixup remains unbeaten. Most schedule, regularization, precision, averaging, initialization, and topology changes regress.
- Dense late width/depth and two-gate SE are the only repeated near-positive neighborhood. EXP-017 produced the strongest 94.16 result and lower 0.2321 final loss, but its gates plus preregistered diagnostic accumulation retained only 95.36% matched throughput.
- EXP-018 recovered exposure to 137.83 passes with a single uninstrumented final gate but scored 93.67. This falsifies gate conditionality as a component-selection rule and shows that the first gate's strong attenuation or two-gate interaction matters.
- The narrow unresolved gap is whether EXP-017's model mechanism can retain its +0.09 signal when observational overhead is removed. This differs from a rerun: the fixed-time treatment executes more optimizer steps while preserving exact gate placement, initialization seed, and model semantics.

## Collected Ideas

## Combinations

## Candidate Ideas

### Early-Only Mild RandAugment
**Summary**: Apply `RandAugment(num_ops=1, magnitude=5)` only while mixup is active, then revert to the accepted crop/flip transform for the final 35% clean-label tail using a worker-safe shared progress signal.

**What it targets**: Missing image invariances while preserving the locally validated clean tail, unlike the rejected always-on proposal.

**Reasoning**: RandAugment has direct CIFAR evidence and is orthogonal to model capacity. Restricting it to the regularized phase addresses the main review objection from EXP-018.

**Sources**: `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/004/008.

**Estimated Effort**: medium

**Risk Assessment**: Mutating worker transforms by progress is complex and may not propagate safely to persistent processes. It also stacks regularizers in a history where additive regularization usually regressed.

### Exact Two-Gate SE Without Diagnostics
**Summary**: Recreate EXP-017's two ratio-16 stage-3 gates with the same exact-neutral initialization and the same preregistered seed 17017, but remove every training-time diagnostic accumulator and terminal gate summary. Keep all model, data, optimizer, and evaluation behavior identical.

**What it targets**: EXP-017's measured quality-versus-exposure gap: preserve the only attention interaction that reached 94.16 while recovering steps consumed by purely observational GPU work.

**Reasoning**: EXP-017 improved both top-1 and final loss; EXP-018 shows the first gate cannot simply be deleted. One uninstrumented gate retained 98.58% versus 95.36% for two instrumented gates, so instrumentation plus the second gate plausibly cost several passes. Removing observation is a clean fixed-budget implementation change and reuses, rather than rerolls, the exact scored gate seed.

**Sources**: `experiments/017/04-analysis.md`; `experiments/018/04-analysis.md`; `03-experiment-learnings.md` medium-importance SE entry.

**Estimated Effort**: medium

**Risk Assessment**: The two MLP gates themselves may account for nearly all overhead, leaving little exposure recovery. EXP-017's 94.16 may also be a narrow trajectory result that extra steps do not improve. This must use seed 17017 exactly and may not tune or rerun.

### Static First-Block Channel Scale Plus Final SE
**Summary**: Put an exact-neutral learned 128-channel residual multiplier on `layer3[0]` and the ratio-16 conditional SE gate on `layer3[1]`. Initialize the scale vector to one and initialize the final gate from the project's fixed seed 42 inside a restored CPU RNG fork, exactly as the scored EXP-018 final gate.

**What it targets**: The interaction exposed by EXP-018: retain gate 0's largely static attenuation and gate 1's conditional behavior with less work than two pooled MLP gates.

**Reasoning**: EXP-017 measured gate 0 mean 0.6468 but only 0.00312 across-example variance, whereas gate 1 was much more conditional. A learned per-channel vector matches that division of labor without hard-coding the post-hoc 0.65 value.

**Sources**: `experiments/017/04-analysis.md`; `experiments/018/04-analysis.md`; H20 shape-sensitivity note in `.autoresearch/project-notes/project-insights.md`.

**Estimated Effort**: medium

**Risk Assessment**: Gate 0 may need input dependence despite low aggregate variance, or two-gate co-adaptation may be inseparable. The new parameterization changes mechanism and initialization trajectory, reducing attribution.

## Review

The blind review selected **Static First-Block Channel Scale Plus Final SE** at 4/5 evidence and 4/5 impact. I adopt its central warning: EXP-017's gate 0 logits were feature-driven despite low output variance, so the experiment tests whether dominant per-channel attenuation is sufficient; conditional co-adaptation may still be essential. Plan review removed the proposed seed-17017 reproduction because reusing a known near-positive initialization is an avoidable seed-selection concern; the final gate instead matches EXP-018's fixed-seed-42 initialization. I do not adopt a within-experiment fallback because a second parameterization after seeing the score would violate single-treatment discipline. Full review: `01-idea-review.md` and `02-plan-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Static first-block scaling plus final SE addresses EXP-018's missing attenuation while preserving most of EXP-017's model interaction at lower runtime cost. Two gates without diagnostics is too close to an unmeasured one-image replay, and early RandAugment retains strong local regularization and worker-state risks.

## Chosen Idea
**Selected**: Static First-Block Channel Scale Plus Final SE

**Why this idea**:
Relative to EXP-018, this treatment adds only a learned, exact-neutral first-block channel vector while preserving its fixed-seed-42 final gate. It directly tests whether cheap learned attenuation restores the missing function without adding another conditional MLP.

**Hypothesis**:
An exact-neutral learned channel scale on `layer3[0]` plus the fixed-seed-42 conditional gate on `layer3[1]` will retain at least 97% matched throughput and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by adding cheap first-block attenuation to EXP-018's final-block selection.

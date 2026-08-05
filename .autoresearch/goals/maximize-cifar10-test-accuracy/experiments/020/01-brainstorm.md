# Brainstorm EXP-020
**Created**: 2026-07-26

## Web Search & Literature Review

- **Mixup** (`knowledge/papers/mixup.md`): convex image/label interpolation improves CIFAR generalization at minimal GPU cost; local EXP-002 validates alpha 0.2 with a timed clean tail.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization duration is an independent lever; late removal can preserve early benefits while enabling clean convergence.
- **RandAugment** (`knowledge/papers/randaugment.md`): mild automated augmentation opens an orthogonal invariance axis but must be reconciled with persistent workers and the clean tail.

No network source was consulted; the loop uses the persistent offline knowledge base and completed experiment history.

## Experimental History Review

- The 94.07% accepted WRN uses batch-shared alpha-0.2 mixup through 65% counted time, followed by a 35% hard-label tail. This is the only accepted refinement after architecture modernization.
- Shortening mixup to 50% scored 93.91 despite slightly more exposure, proving the 50-65% window is useful. Stronger alpha 0.4 scored 93.57, while per-example coefficients scored 93.79; batch-shared alpha 0.2 remains calibrated.
- Attention/capacity, precision, averaging, decay, initialization, and topology variants are now exhausted or repeatedly below margin. The opposite side of mixup duration (75%) and weaker alpha (0.1) remain controlled gaps in the only validated regularization mechanism.
- The limiting error mode is a narrow generalization/convergence balance: training loss reaches near zero, but modifications that add or remove too much regularization worsen test accuracy. A duration extension preserves the validated mechanism while shortening, but not eliminating, clean refinement.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Mixup
**Summary**: Keep the accepted 65% cutoff but reduce the batch-shared beta parameter from 0.2 to 0.1, producing more near-clean examples during the regularized phase.

**What it targets**: Excess target/input softness while preserving batch-level coefficient coherence and the validated temporal structure.

**Reasoning**: Alpha 0.4 was decisively too strong; alpha 0.1 tests the opposite direction without changing duration or throughput. It may preserve mixup's regularization while allowing earlier class-specific fitting.

**Sources**: EXP-002, EXP-005, EXP-015; `knowledge/papers/mixup.md`.

**Estimated Effort**: low

**Risk Assessment**: The alpha-0.4 failure does not imply monotonic improvement toward weaker mixing. Alpha 0.2 may be the balanced point, and weaker mixing may under-regularize like the 50% cutoff.

### Early-Only Mild RandAugment
**Summary**: Apply one magnitude-5 RandAugment operation only while mixup is active, using a `multiprocessing.Value` inherited by persistent workers and polled by a wrapper transform; disable it at the accepted 65% transition.

**What it targets**: Image invariances beyond crop/flip while preserving the clean final phase and proving worker-side phase propagation.

**Reasoning**: It is the main orthogonal data avenue left and has direct CIFAR evidence. Shared-memory state addresses the persistent-worker flaw identified in EXP-019 review.

**Sources**: `knowledge/papers/randaugment.md`; EXP-002/003/005/006; EXP-019 idea review.

**Estimated Effort**: high

**Risk Assessment**: It compounds mixup during the early phase despite repeated additive-regularization failures, adds CPU cost, and introduces a shared-worker control path that requires strong semantic preflight.

### Extend Mixup to Seventy-Five Percent
**Summary**: Change only `MIXUP_END_FRACTION` from 0.65 to 0.75, retaining batch-shared alpha 0.2 and a 25% hard-label tail.

**What it targets**: The duration balance of the sole validated generalizer. EXP-004 established that 50% is too short, but the longer side has never been tested.

**Reasoning**: The 50% result lost 0.16 points relative to 65%, consistent with useful regularization late in training. Extending by a smaller 10-point window tests whether 65% still under-regularizes while retaining 75 counted seconds for clean margin refinement.

**Sources**: EXP-002 and EXP-004 reports; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`.

**Estimated Effort**: low

**Risk Assessment**: The accepted cutoff may already be optimal; shortening the clean tail can delay class-boundary refinement and repeat over-regularization failures. One fixed 75% treatment closes only this point, not the whole duration curve.

## Review

The blind review selected **Extend Mixup to Seventy-Five Percent** at 7/10 evidence and 5/10 impact. I adopt the inverted-U warning: 65% may already be optimal, and the 25% clean tail may be too short. I retain 75% rather than the suggested 70% hedge because the goal requires a +0.10-point move and 75% is the preregistered, clearer opposite-side probe. The review rejects alpha 0.1 as likely under-regularization and RandAugment as a repeat of additive regularization with worker complexity. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The 75% cutoff is supported by the only measured favorable gradient, changes one exposed constant, preserves batch-shared alpha 0.2, and keeps a meaningful clean tail. The alternatives have weaker local evidence and more compound risks.

## Chosen Idea
**Selected**: Extend Mixup to Seventy-Five Percent

**Why this idea**:
EXP-004 showed that ending mixup at 50% was too early, losing 0.16 points versus 65%. Extending the validated batch-shared alpha-0.2 treatment to 75% is the cleanest test of whether the duration optimum lies later, with no throughput or implementation confound.

**Hypothesis**:
Changing only `MIXUP_END_FRACTION` from 0.65 to 0.75 will preserve accepted exposure and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by providing useful regularization for 30 additional counted seconds while retaining a 75-second hard-label tail.

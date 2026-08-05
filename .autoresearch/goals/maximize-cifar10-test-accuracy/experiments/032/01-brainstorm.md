# Brainstorm EXP-032
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): symmetric Beta input/target interpolation remains the strongest low-cost local generalizer; alpha controls endpoint concentration without extra model compute.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early-only regularization followed by clean refinement supports preserving the validated 65%/35% schedule.

No network source was consulted. This offline quick pass uses the local knowledge base and the developed EXP031 proposals.

## Experimental History Review

- EXP027 remains the 94.32% baseline. Its exact `(2,2,3)` depth, early N1/M5 RandAugment, batch-shared alpha-0.2 mixup through 65%, full FP32 gradients, schedule, decay, and hard tail are the only demonstrated positive composition.
- Exact early residual masking is closed by two normal-exposure regressions; channels-last is closed by its fixed semantic bound; late freezing, small batch, BF16, schedule/floor, late decay removal, SAM, EMA, attention, and nearby capacity allocations are also negative or infeasible.
- Mixup duration is bracketed at 50%/75%, stronger alpha 0.4 failed, and per-example coefficients failed. Alpha 0.1 is the remaining clean batch-shared strength bracket on the accepted deeper-plus-RandAugment learner, but the prior is negative.
- The current limiter is generalization/boundary quality at near-zero tail loss. Compute is binding but exposure-only treatments repeatedly regressed, so an input/target change with near-zero cost is preferable to another pure speed intervention.

## Collected Ideas

## Combinations

## Candidate Ideas

### Reflection-Padded Random Crops
**Summary**: Change only the accepted `RandomCrop(32, padding=4)` to `RandomCrop(32, padding=4, padding_mode="reflect")`, retaining flip, early RandAugment, worker-private RNG, mixup, model, optimizer, and clean-tail timing. The crop offsets remain drawn by the same worker stream; only pixels exposed outside the original image change from zero to reflected content.

**What it targets**: Artificial zero borders introduced by accepted crop augmentation, seeking more natural boundary statistics at no counted GPU cost and without masking internal features.

**Reasoning**: It is orthogonal to failed feature masking and does not consume the clean tail. Reflection can remove high-contrast padding artifacts while preserving spatial displacement. Local evidence is limited, and RandAugment may already supply enough image invariance; exact worker replay must distinguish intended active-image changes from unintended RNG/cutoff drift.

**Sources**: current `train.py`; EXP026-EXP027 worker augmentation evidence; `knowledge/papers/randaugment.md`; `02-system-understanding.md` data-path overlap.

**Estimated Effort**: low

**Risk Assessment**: Reflection can create repeated edge texture and weaken useful crop regularization; active worker outputs intentionally diverge and could interact poorly with RandAugment. A valid miss closes this padding geometry without trying replicate/symmetric modes or padding sizes.

### Batch 512 With a Fully Scaled LR Curve
**Summary**: Change `BATCH_SIZE=512`, `LR=0.4`, `MIN_LR=0.004`, and image-equivalent `MAX_STEPS=32000`, preserving every other accepted component. Require at least 1.10x measured complete-body image rate and 146.308 projected passes before a sole score.

**What it targets**: The 98% forward/backward share and 98.9% H20 memory headroom by seeking better convolution utilization and more images per fixed time.

**Reasoning**: This is the unexplored large-batch counterpart to infeasible batch 128 and has the largest upside. It is not optimization-equivalent: even at the gate it uses about 45% fewer Nesterov/BN/mixup decisions, and exposure-only treatments repeatedly failed, making it an indivisible low-evidence operating point.

**Sources**: EXP031 `proposals/idea-01.md` and `01-idea-review.md`; EXP009, EXP016, EXP027-EXP029; `02-system-understanding.md`; `project-notes/project-insights.md`.

**Estimated Effort**: medium

**Risk Assessment**: Coarser stochastic updates, doubled LR, larger BN estimates, changed mixup refresh, dropped tail, and extra eval opportunities may degrade accuracy despite speed. A valid miss cannot authorize adjacent batch/LR/momentum/warmup repair.

### Weaker Alpha-0.1 Batch-Shared Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2 ->0.1`. Preserve the one scalar coefficient shared by batch 256, 65% cutoff, deeper-plus-early-RandAugment learner, full FP32 Nesterov SGD, seed 42, worker isolation, and evaluator. Beta(0.1,0.1) keeps mean 0.5 while moving 81.28% of draws outside `[0.1,0.9]`, making early batches more endpoint-heavy.

**What it targets**: Boundary quality without adding feature masks or model compute, testing whether the composed deeper/augmented learner needs slightly less interpolation softness.

**Reasoning**: This is the only unmeasured one-axis strength bracket and has exact one-line attribution. It preserves the full residual contribution protected by EXP030. Contrary evidence is strong: alpha 0.4, duration changes, and coefficient decorrelation all failed, while EXP027 shows no direct over-regularization symptom.

**Sources**: EXP031 `proposals/idea-03.md`; `knowledge/papers/mixup.md`; EXP004, EXP005, EXP015, EXP020, EXP027, EXP030.

**Estimated Effort**: low

**Risk Assessment**: Likely under-regularization and alpha-dependent Beta rejection changes the subsequent CUDA permutation trajectory by design. One valid normal-exposure miss must close immediate alpha strength tuning with no adjacent value, seed, cutoff, or coefficient rescue.

## Review

The reviewer selected reflection padding at 3/5 evidence and 3/5 impact, emphasizing that support is mechanistic rather than demonstrated locally. I adopt the required controls: exact one-line production diff; direct proof that crop offsets, flip decisions, sampler order, private RandAugment draws, and cutoff state remain aligned; measurement of how often sampled windows touch padding; proof that pre-normalization differences are confined to padding-derived pixels; real-loader stability; >=130-pass protection; and closure of reflection/symmetric/replicate/padding-size variants after one valid miss. Final accuracy and loss remain corroboration only. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Reflection padding is the only finalist that targets boundary statistics while preserving the accepted model, full-gradient decisions, BN batch size, LR trajectory, mixup process, and counted exposure. Batch 512 has higher upside but changes the optimizer regime; alpha 0.1 is cleaner but sits in a uniformly negative neighborhood.

## Chosen Idea
**Selected**: Reflection-Padded Random Crops

**Why this idea**:
Reflection padding replaces an artificial high-contrast crop boundary with image-derived content at effectively zero counted GPU cost. It is distinct from internal feature masking, keeps every accepted training mechanism, and can be tested with exact worker decision-stream controls and a one-shot closure rule.

**Hypothesis**:
If zero-padded crop boundaries are degrading the deeper-plus-RandAugment learner's boundary quality, then changing only `RandomCrop` to `padding_mode="reflect"` will retain at least 130 passes and raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%. `final_test_acc >=94.32%` and `final_test_loss <=0.2523` will be reported as corroboration only. A valid normal-exposure miss closes reflection, symmetric, replicate, and alternate-padding-size geometry without retry.

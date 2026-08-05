# Brainstorm EXP-029
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): mixup supplies low-cost soft-target regularization, but the local alpha/duration record now favors preserving the accepted alpha-0.2/65% policy.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): temporal regularization is useful, but EXP-028 shows the critical-period argument does not extend to freezing representation gradients.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): the accepted deeper WRN benefits from modest low-resolution capacity; its full gradient path should remain intact.

No network source was consulted; this offline quick pass reuses persistent local literature and completed experiment evidence.

## Experimental History Review

- The baseline remains EXP-027 at 94.32% best / 94.22% final with 133.01 passes. Its exact deeper-plus-early-RandAugment interaction is the accepted base and must remain intact.
- EXP-028 proved that exposure is not the missing ingredient by itself: late prefix freezing delivered 159.10 passes and 1.56x hard-tail speed but lost 0.33 points and worsened loss. Every stage therefore keeps gradients throughout EXP-029.
- The exact batch-128 / fully scaled LR proposal was the reviewed EXP-028 runner-up at 3/5 evidence and 3.5/5 impact. It remains the strongest untested operating point and changes update granularity rather than deleting representation learning.
- Mixup duration is bracketed at 65%, stronger alpha 0.4 failed, and per-example coefficients failed. Alpha 0.1 is the only clean strength-side gap, but its expected ceiling is low because the accepted model still benefits from substantial early regularization.
- Broad dropout failed by 0.55 points; targeted third-block drop-path is materially narrower but still stacks a third early regularizer and has no compute upside.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2 ->0.1`, retaining the accepted batch-shared coefficient, 65% cutoff, deeper-plus-RandAugment model, optimizer, schedule, seed, and evaluation. Beta(0.1,0.1) yields more endpoint-heavy mixtures while preserving the mean coefficient and negligible overhead.

**What it targets**: Possible residual over-regularization in the accepted early phase, using a single-axis target/input interpolation adjustment without changing exposure or the hard tail.

**Reasoning**: Alpha 0.4 failed strongly, but the weaker side has never been scored. This completes the strength bracket with a clean intervention and exact accepted RNG topology aside from the intended Beta distribution.

**Sources**: `knowledge/papers/mixup.md`; EXP-002/004/005/015/020/027; `03-experiment-learnings.md`.

**Estimated Effort**: low

**Risk Assessment**: Every prior perturbation of the accepted mixup mechanism regressed, and the new accepted capacity likely needs rather than rejects early regularization. The likely effect is below the +0.10 margin.

### Batch 128 With Fully Scaled LR
**Summary**: Change only `BATCH_SIZE=256 ->128`, the full time-based LR curve `0.2 ->0.002` to `0.1 ->0.001`, and the nonbinding safety cap `64,000 ->128,000`. Preserve the accepted `(2,2,3)` model, early RandAugment, alpha-0.2 batch-shared mixup, FP32 SGD/Nesterov, weight decay, seed, loader, and evaluator. The exact developed specification is `experiments/028/proposals/idea-02.md`.

**What it targets**: Boundary quality per backward pass. At comparable image exposure, batch 128 makes roughly twice as many smaller-batch optimizer, BN, and batch-shared mixup decisions without freezing any representation.

**Reasoning**: The batch/LR pair follows the original linear-scaling lineage and preserves 49,920 images per epoch. EXP-028 rejects extra upper-only steps, not finer full-model updates. Require >=120 projected/realized passes and >=46,875 updates to keep the treatment in a materially different but viable regime.

**Sources**: `02-system-understanding.md`; `experiments/028/proposals/idea-02.md`; EXP-001/008/009/016/027/028.

**Estimated Effort**: low

**Risk Assessment**: The treatment jointly changes gradient noise, momentum/BN/mixup horizons, weight-decay frequency, and the terminal LR floor. EXP-008 makes the 0.001 floor the clearest negative prior, and a null cannot isolate batch size alone.

### Early Drop-Path on the Added Block
**Summary**: Apply private-RNG per-example inverted stochastic depth at fixed `p=0.05` only to `layer3[2]` through 65%, then use the exact full accepted model for the hard tail and all evaluations. The residual branch remains computed, and all global RNG streams remain accepted. Exact developed specification: `experiments/028/proposals/idea-03.md`.

**What it targets**: Generalization of the specific extra block whose capacity became useful only with early RandAugment, rather than broad network regularization.

**Reasoning**: It differs from EXP-006 by targeting one of seven blocks, halving probability, and dropping a whole branch contribution per example. This is the only remaining targeted model regularizer with a precise local interaction rationale.

**Sources**: `experiments/028/proposals/idea-03.md`; EXP-006/011/026/027; `knowledge/papers/time-matters-regularization.md`.

**Estimated Effort**: medium

**Risk Assessment**: It stacks another early regularizer after broad dropout failed, can damage the accepted block/RandAugment synergy, buys no exposure, and may be too weak to clear +0.10.

## Review

The reviewer selected batch 128 with fully scaled LR at 3.5/5 evidence and 3.5/5 impact. I adopt the central caveat: this is one indivisible operating point, not a clean batch-size or update-granularity ablation, because momentum/BN/mixup horizons, decay frequency, and the 0.001 terminal floor change together. Planning must preserve the exact four constants, fixed exposure/update gates, and no-repair rule, while preregistering final loss and endpoint accuracy as evidence for the weaker-tail failure mode. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Batch 128 is the only candidate that addresses the open full-model boundary-quality question while preserving every stage's gradients and the accepted depth/RandAugment interaction. Alpha 0.1 has a uniformly negative neighborhood and low expected ceiling; drop-path has a speculative mechanism and the strongest contrary prior.

## Chosen Idea
**Selected**: Batch 128 With Fully Scaled LR

**Why this idea**:
The exact batch-128 / `0.1 ->0.001` curve is a coherent lineage-based operating point and the strongest remaining untested lever. EXP-028 shows that more updates help only if every representation continues adapting; this treatment supplies more full-model decisions instead of reallocating gradients. Direct H20 and real-loader gates will prevent a severe image-exposure or wall-time confound.

**Hypothesis**:
If the accepted deeper-plus-invariance model benefits from finer full-model optimizer/BN/mixup decisions under a proportionally scaled LR, then batch 128 with the exact `0.1 ->0.001` curve will retain at least 120 data passes, complete at least 46,875 full-model updates, raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%, and retain `final_test_acc >=94.32%`. A valid miss closes this exact four-constant operating point without isolating any one of its coupled mechanisms.

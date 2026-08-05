# Brainstorm EXP-028
**Created**: 2026-07-26

## Web Search & Literature Review

- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early training can establish durable generalization effects, supporting a clean temporal intervention at the accepted 65% boundary.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): modest depth and width are effective CIFAR representation levers; local evidence now shows low-resolution depth becomes useful when paired with early invariance.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild soft targets can reduce overconfidence, but stacking them with mixup requires careful temporal separation.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): short-window iterate averaging can reduce terminal variance at low compute cost, though EXP-013 rejected one long-lag whole-state EMA design.

No network source was consulted; this offline pass uses the persistent local paper distillations and measured experiment record.

## Experimental History Review

- EXP-027 established the new 94.32% baseline by composing exact `(2,2,3)` depth with exact early RandAugment. It completed 133.01 passes and lowered final loss to 0.2523 relative to depth alone, proving that the accepted treatment is an interaction, not a standalone capacity or augmentation effect.
- The system breakdown at `02-system-understanding.md` measures backward at about 74% of counted step time. Stage 1 costs 39.6% of forward despite only 3.3% of parameters; stage 3 holds 83.1% of parameters for 29.5% of forward. The optimization problem is therefore accuracy per backward pass, not data, optimizer overhead, wall time, or memory.
- High-resolution processing cannot be removed architecturally: EXP-016's `[1,2,3]` redistribution lost 0.50 points versus the new baseline. This does not establish that high-resolution weights still need gradients during the low-LR final 35%.
- Exact mixup duration, stronger mixup, CutMix, broad early dropout, late decay removal, cosine-to-zero, BF16, long-horizon EMA, SAM, adjacent stage-3 capacity, and the full SE family are closed by valid misses or repeated feasibility failures.
- The strongest untested operating-point change remains batch 128 with a fully proportional LR curve. Other gaps are temporal compute allocation, targeted regularization of only the newly useful block, and a very short terminal averaging window distinct from EXP-013.

## Collected Ideas

- **Freeze stem and stage 1 after 65%** - at the exhausted epoch boundary that already ends mixup/RandAugment, set their parameters non-trainable while keeping forward computation and BatchNorm running statistics. This directly attacks the dominant backward cost and tests whether early high-resolution features are already settled before the low-LR hard-label tail; EXP-016 establishes that the features matter, not that their late gradients do.
- **Batch 128 with fully scaled LR** - halve batch size and the complete LR curve to `0.1 -> 0.001`, doubling only the safety step cap. This trades kernel efficiency for roughly twice as many smaller-batch optimizer/BatchNorm/mixup decisions and is the strongest remaining standalone operating-point gap.
- **Targeted early stochastic depth on only the third stage-3 block** - bypass only the new block with small probability through 65%, then train the full accepted model in the hard tail. Unlike EXP-006's p=0.10 dropout on every residual branch, this regularizes the specific capacity whose standalone version overfit and whose RandAugment composition succeeded.
- **Label-smoothed bridge in the hard tail** - after mixup/RandAugment end, use very mild smoothing for a fixed middle slice before a final unsmoothed interval. This targets the remaining near-zero-train/high-test-loss gap without stacking target regularizers simultaneously, but risks consuming the hard-label refinement window already shown to matter.
- **Short terminal parameter EMA** - begin a parameter-only high-decay EMA very late and evaluate it with live BN state, targeting the 0.10-point best-to-final decline. This is distinct from EXP-013's 65%-start whole-state EMA but is weakly identified because one endpoint cannot establish trajectory variance.
- **Freeze only stage 1 convolutions, retain affine BN updates** - a narrower temporal-compute simplification that preserves late channel rescaling while removing convolution gradients. It may retain more adaptability than freezing the whole stage, but complicates attribution and saves less backward work.
- **Auxiliary early-only stage-2 classifier** - attach a discarded classifier during the first 65% to strengthen representation gradients, then remove its loss for the clean tail. This moonshot spends extra early backward compute to improve feature quality rather than exposure; it is plausible only if optimization quality, not step count, dominates.
- **Compute-efficient replacement for the third block** - replace its two dense convolutions with a bottleneck or grouped transformation while retaining the accepted early-invariance policy. This could recover passes, but EXP-012's bottleneck failure and the accepted block's likely dense-interaction value make it low confidence.

## Combinations

- **Late stage-1 freeze + batch 128**: smaller batches supply more optimizer decisions while the late freeze repays part of their GPU-efficiency cost. The cross could preserve pass exposure better than batch 128 alone, but it entangles two optimization mechanisms and should follow isolated evidence rather than lead.
- **Third-block stochastic depth + existing early RandAugment**: the base already contains early RandAugment; selectively regularizing only the added block may make its capacity more robust than either broad dropout or image augmentation alone. The risk is compounding early regularization in the exact window whose current balance succeeded.
- **Late stage-1 freeze + short terminal EMA**: faster tail updates may increase endpoint variance, while a short EMA could stabilize it. This is stronger mechanistically than either alone but makes a null result uninterpretable and is unsuitable before testing the freeze itself.

## Candidate Ideas

### Batch 128 With Fully Scaled LR
**Summary**: Change only batch 256 to 128, the full LR curve `0.2 -> 0.002` to `0.1 -> 0.001`, and the safety cap 64k to 128k. Preserve the accepted deeper-plus-RandAugment architecture, optimizer family, time schedule, seed, transforms, and evaluator. Full specification: `proposals/idea-02.md`.

**What it targets**: The accepted run nearly interpolates yet makes only 25,978 optimizer decisions. Smaller batches create finer, noisier optimizer, BatchNorm, and batch-shared mixup decisions, testing boundary quality rather than raw example exposure.

**Reasoning**: The exact batch/LR pair follows the original linear-scaling lineage and retains equal examples per epoch. Require >=120 projected/realized passes and >=46,875 updates so the run materially changes granularity without collapsing exposure.

**Sources**: `02-system-understanding.md`; `proposals/idea-02.md`; EXP-001/008/009/016/027.

**Estimated Effort**: low

**Risk Assessment**: Halving the 0.002 floor can weaken the proven terminal refinement; momentum, BN, weight-decay, and mixup horizons all change in example units, making a null hard to attribute. Batch-128 kernels may miss the exposure gate.

### Early Drop-Path on the Added Block
**Summary**: Apply per-example inverted stochastic depth with fixed `p=0.05` only to `layer3[2]` through 65% counted time, driven by a private CUDA generator, then use the exact accepted full model for the hard tail and every evaluation. The branch is always computed, so this is a targeted regularizer rather than conditional compute. Full specification: `proposals/idea-03.md`.

**What it targets**: The added block's standalone high test loss suggests brittle capacity, while its composition with early RandAugment succeeded. Narrow masking tests whether the new block specifically benefits from early ensemble-like regularization at near-zero overhead.

**Reasoning**: This differs from EXP-006 by targeting one of seven blocks, dropping a whole residual contribution per example after its convolutions, halving the probability, and operating on the accepted interaction. A private RNG prevents the mask from rerolling mixup or model streams.

**Sources**: `02-system-understanding.md`; `proposals/idea-03.md`; EXP-006/011/026/027; `knowledge/papers/time-matters-regularization.md`.

**Estimated Effort**: medium

**Risk Assessment**: It stacks a third early regularizer in a history where broad dropout and stronger regularization failed, may damage the exact block/RandAugment interaction, and adds overhead without saving compute. The effect may be too small for a +0.10 margin.

### Late Stem and Stage-1 Freeze
**Summary**: Preserve the accepted model and first 65% trajectory exactly, then at the exhausted RandAugment boundary clear gradients and set all 33,424 stem/stage-1 parameters non-trainable once. Keep the full forward path and live BatchNorm running-stat updates, leave optimizer membership/state intact, and spend the saved prefix backward work on more hard-label updates to stages 2/3 and the classifier. Full specification: `proposals/idea-01.md`.

**What it targets**: Backward is about 74% of counted cost and the high-resolution prefix is the largest forward stage, while the late LR is already small. The treatment tests temporal compute allocation: essential early features may no longer need late gradients even though their transformations must remain present.

**Reasoning**: EXP-016 proves removing high-resolution capacity for the whole run is harmful; this proposal instead retains its exact initialization, early training, forward computation, and BN statistics. The accepted 65% boundary is preregistered and literature supports an early critical period. Matched hard-tail timing must show >=1.20x speed and >=145 projected passes before scoring.

**Sources**: `02-system-understanding.md`; `proposals/idea-01.md`; EXP-016/027; `knowledge/papers/time-matters-regularization.md`.

**Estimated Effort**: medium

**Risk Assessment**: Clean hard-label inputs may require continued low-level adaptation precisely after augmentation ends; stopping prefix momentum and decay may also harm. More upper-layer updates can overfit, and speed alone has not predicted accuracy.

## Review

The reviewer selected late stem/stage-1 freezing at 3.5/5 evidence and 4/5 impact. I adopt its caveats: the regularization critical-period literature does not establish that early convolution gradients become dispensable, and forward-stage shares cannot predict backward savings. Planning must treat the idea as a high-risk temporal allocation test, require direct balanced hard-tail timing and >=145 projected passes, prove frozen weights/momentum remain fixed while BN buffers stay live, and require final accuracy to meet the same 94.42 improvement threshold as best accuracy against extra evaluation opportunities. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The freeze is the only finalist that directly converts the measured 74% backward bottleneck into a temporally isolated treatment while preserving the entire accepted early trajectory and inference representation. Batch 128 is the fallback but changes five coupled optimizer/data-group semantics for the whole run. Drop-path has the weakest evidence because it stacks another early regularizer after broad dropout failed.

## Chosen Idea
**Selected**: Late Stem and Stage-1 Freeze

**Why this idea**:
The treatment preserves exactly what EXP-016 showed is essential - both high-resolution blocks and their first 65% learning - while testing a genuinely different question: whether their low-LR late gradients are worth more than additional upper-layer hard-label updates. It has the largest measured upside and a fail-closed timing gate. The interpretation will remain narrow because neither the literature nor prior runs directly establish late-gradient redundancy.

**Hypothesis**:
If the stem and stage-1 parameters are sufficiently established at the exhausted 65% boundary, freezing exactly their 33,424 parameters while keeping the full forward path and live BN statistics will deliver at least 145 data passes and raise both fixed-seed best and final test accuracy from 94.32% to at least 94.42%. Failure at adequate exposure will show that continued late prefix adaptation is more valuable than extra upper-layer decisions.

# Brainstorm EXP-004
**Created**: 2026-07-24

## Web Search & Literature Review

This local-only quick pass reused the goal's curated sources.

- **mixup** (`knowledge/papers/mixup.md`): convex sample/target interpolation improves CIFAR generalization at negligible forward-pass overhead; EXP-002 validates alpha 0.2 locally.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization often matters most in an early critical period and can be removed without losing its generalization benefit.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): carefully windowed late parameter averages can mildly improve generalization, though the current low-LR endpoint leaves limited variance to exploit.

## Experimental History Review

- EXP-001 established WRN-16-2 and a time-aligned cosine schedule, moving the original 91.54% baseline to 93.38%.
- EXP-002 added alpha-0.2 mixup through 65% counted time and reached the current 94.07% best with 141.9 data passes; final equaled best, validating early soft targets plus a long hard-label tail.
- EXP-003 replaced mixup with shared-rectangle CutMix, preserved 142.5 passes, but regressed to 93.72%. The result favors refining the proven whole-image interpolation mechanism rather than trying another spatial label proxy.
- Mixup strength and cutoff remain untuned. Late EMA remains untested but has small estimated headroom and final-evaluation risk.

## Objective Diagnosis

The accepted model is still generalization-limited, but EXP-003 shows that regularizer quality matters more than generic regularization strength. EXP-002 continued improving through its 105-second hard-label tail and ended at its best, suggesting clean-label margin refinement is valuable and may benefit from more time. The narrow next question is whether the alpha-0.2 mixup critical period can end earlier, or whether softer-target strength rather than duration should change. Any candidate must preserve the roughly 142-pass exposure and exceed 94.17%.

## Collected Ideas

Quick pass; omitted.

## Combinations

Quick pass; omitted.

## Candidate Ideas

### Gentler Alpha-0.1 Mixup
**Summary**: Preserve the 65% cutoff and all other settings, but reduce `MIXUP_ALPHA` from 0.2 to 0.1. The more endpoint-heavy beta distribution presents examples closer to natural images while retaining mixed targets throughout the validated critical period.

**What it targets**: Possible over-softening during representation learning without shortening the duration of the successful intervention.

**Reasoning**: EXP-003 shows that stronger spatial label mixing can hurt despite normal throughput. Reducing alpha is a direct, isolated way to retain mixup's linearity bias while lowering average interpolation severity and potentially easing the transition into hard-label refinement.

**Sources**: `knowledge/papers/mixup.md`; EXP-002 and EXP-003 reports; current `train.py` beta implementation.

**Estimated Effort**: low

**Risk Assessment**: Beta(0.1,0.1) often produces nearly unmixed batches, so the change may discard useful regularization and fall back toward the 93.38% non-mixup baseline. There is no local evidence that alpha 0.2 is too strong.

### Short-Horizon Final EMA
**Summary**: Keep the complete EXP-002 trajectory and maintain FP32 parameter EMA shadows from 75% counted time with decay 0.99 every ten steps. Preserve live intermediate evaluations and use the EMA view only for the final evaluation.

**What it targets**: Residual endpoint iterate noise without changing examples, gradients, exposure, or the validated mixup schedule.

**Reasoning**: Weight-averaging evidence supports late, short windows at low overhead. A final-only short horizon addresses the earlier concerns about long-window lag and losing all live late evaluations.

**Sources**: `knowledge/papers/weight-averaging.md`; EXP-002 idea review; `experiments/003/proposals/idea-01.md`; EXP-002 report.

**Estimated Effort**: medium

**Risk Assessment**: The cosine schedule already ends at LR 0.002 and EXP-002 final equaled best, leaving little variance to average. EMA/BN mismatch and replacing the live final evaluation create real downside, while expected upside may be below 0.10 points.

### Earlier 50% Mixup Cutoff
**Summary**: Keep every accepted EXP-002 setting, including alpha 0.2, but disable mixup at 50% rather than 65% counted time. This allocates 150 seconds to mixed-target representation learning and 150 seconds to the unchanged hard-label cosine path, changing one exposed constant.

**What it targets**: The balance between early generalization and late clean-label margin refinement. EXP-002's final-best equality and continued tail improvement suggest the current 105-second hard tail may not fully exploit low-LR hard-label convergence.

**Reasoning**: Critical-period evidence supports removing regularization after early learning, and EXP-002 proves the mechanism works locally. A 50% cutoff remains far beyond warmup and the high-LR early phase while adding 45 seconds of clean-label training at moderate-to-low LR.

**Sources**: `knowledge/papers/time-matters-regularization.md`; `knowledge/papers/mixup.md`; EXP-002 and EXP-003 reports; goal learnings.

**Estimated Effort**: low

**Risk Assessment**: The successful gain may depend on mixup remaining active through 65%; ending at 50% could weaken representation regularization and simply overfit longer. The expected improvement is modest and close to single-run noise.

## Review

The reviewer selected the 50% cutoff as the only candidate anchored directly in EXP-002's continued hard-label-tail improvement. Significant concern adopted: that improvement is confounded with cosine decay, so 50% is a two-sided probe that may under-regularize rather than a presumed win. A null will not trigger a rerun; it routes a later loop toward the opposite 75% arm or a different lever. The alpha-0.1 candidate was downgraded because a more U-shaped beta distribution weakens typical mixing and EXP-003 does not show alpha 0.2 is too strong. EMA was deferred because the stable low-LR endpoint offers little variance to average.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`: **Earlier 50% Mixup Cutoff** scored 6/10 for evidence/reasoning and potential impact, ahead of EMA's low ceiling and alpha 0.1's weak directional support. The experiment remains a single clean cutoff test rather than a multi-run sweep so it stays within the normal full-budget protocol and yields direct attribution.

## Chosen Idea
**Selected**: Earlier 50% Mixup Cutoff

**Why this idea**:
The accepted alpha-0.2 mechanism and every other training choice remain fixed; only the division between mixed-target learning and hard-label refinement changes. This directly tests the strongest unresolved signal from EXP-002 while preserving throughput and avoiding the failed CutMix mechanism.

**Hypothesis**:
Changing `MIXUP_END_FRACTION` from 0.65 to 0.50 will give the converged WRN 45 additional seconds of hard-label cosine refinement and raise `best_test_acc` from 94.07% to at least 94.17% without materially reducing exposure or violating wall-time compliance.

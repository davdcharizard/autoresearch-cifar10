# Brainstorm EXP-025
**Created**: 2026-07-26

## Web Search & Literature Review

- **EXP-017 full-SE result** (`experiments/017/04-analysis.md`): two global cross-channel gates scored 94.16 with 133.64 passes; observation diagnostics were included in its 4.6% overhead.
- **EXP-018/019/024 ablation chain**: removing a placement, removing input dependence, or restricting dependence to the same channel scored 93.67/93.86/93.91. Full two-gate cross-channel interaction is the remaining distinguishing attention mechanism.
- **mixup** (`knowledge/papers/mixup.md`): the accepted alpha-0.2 schedule remains validated; alpha 0.1 is the last isolated strength-side gap but has an adverse under-regularization prior.

No network source was consulted; this offline loop uses persistent knowledge and completed experiment artifacts.

## Experimental History Review

- Accepted accuracy remains 94.07% with a 94.17% threshold after 22 rejected follow-ups. Most standalone changes lose 0.15-0.55 points; only full SE, extra depth, selective width, and EMA reached 94.10-94.16.
- Full two-gate ratio-16 SE is the closest result at 94.16. Its diagnostics performed multiple scalar reductions on every training forward; no diagnostic-free accepted-width score exists.
- Four attention experiments now isolate the mechanism: both placements, input dependence, and global cross-channel mixing distinguish full SE's signal. Final-only, static, and diagonal self-gating are closed.
- Composing full SE with 160-channel width projected only 126.21 passes and was aborted. Accepted width is required for any final attention closure.
- More exposure alone is not a general solution: BF16, EMA, and cheaper attention variants did not convert additional steps into acceptance. A diagnostic-free full-SE run is therefore a narrow efficiency closure, not a broad exposure thesis.
- The remaining limiter is a noise-scale representation/generalization boundary. Any attention candidate must preserve cross-channel conditioning; otherwise it should be judged mainly for mechanism closure, not expected acceptance.

## Collected Ideas

## Combinations

## Candidate Ideas

### Neighbor-Mixing Conditional Gates
**Summary**: Gate both accepted stage-3 residuals using a zero-initialized shared-kernel 1D convolution of width three over the 128-channel pooled vector, followed by `2*sigmoid`. This introduces immediate neighboring-channel interaction with only four parameters per gate and exact-neutral first-step opening.

**What it targets**: The cross-channel dependency absent from EXP-024 while retaining near-baseline exposure, both placements, and per-example conditioning.

**Reasoning**: Diagonal self-gating failed, so some channel interaction is needed. A channel-axis convolution lets each scale depend on adjacent pooled features at negligible parameter cost and can test whether local rather than globally dense interaction is sufficient.

**Sources**: EXP-017/024 mechanism comparison; recurring attention learning.

**Estimated Effort**: medium

**Risk Assessment**: Channel ordering has no known locality, so neighbor mixing may be arbitrary and another destructive simplification. Treat a miss as closure; require exact gradients and >=138 passes.

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA` from 0.2 to 0.1, preserving batch-shared sampling, the 65% cutoff, accepted architecture, schedule, and throughput.

**What it targets**: The only unmeasured mixup-strength direction at the validated duration.

**Reasoning**: Alpha 0.4 was too strong; alpha 0.1 cleanly tests whether accepted alpha lies above the optimum with no implementation ambiguity.

**Sources**: `knowledge/papers/mixup.md`; EXP-002/004/005/015/020.

**Estimated Effort**: low

**Risk Assessment**: Endpoint-heavy Beta(0.1,0.1) likely under-regularizes and all prior mixup perturbations regressed. Expected ceiling is low.

### Diagnostic-Free Full Two-Gate SE Closure
**Summary**: Recreate EXP-017's exact two `128->8->128` ratio-16 gates, fixed seed 17017, exact-neutral initialization, and residual placement, while removing all training-time diagnostic buffers, reductions, and terminal gate reporting. Keep every training/evaluation setting unchanged.

**What it targets**: The 94.16 near miss using the only attention function class that survived all ablations, while recovering fixed-time work spent solely on observation.

**Reasoning**: EXP-018/019/024 now show that simplifying placement, conditionality, or cross-channel mixing loses the signal. Removing diagnostics changes no loss or gate semantics and is the only remaining efficiency treatment that preserves full interaction. Reusing seed 17017 is required to isolate observation cost and is not a reroll.

**Sources**: EXP-017 through EXP-019 and EXP-024 reports; high-importance attention learning.

**Estimated Effort**: medium

**Risk Assessment**: The 94.16 anchor is outcome-selected and more steps have not generally improved top-1. This is a noise-scale closure with a narrow ceiling. Require >=137 projected passes and never change seed, ratio, placement, or rerun.

## Review

The blind reviewer selected diagnostic-free full SE at 4/5 evidence but only 2/5 impact. I adopt the narrow framing: seed 17017, gate state/function, placement, accepted common state/RNG, and optimizer grouping must exactly reproduce EXP-017 without observation; >=137 projected passes must convert the exact +0.09 trajectory. Any score below 94.17 closes exposure as the missing ingredient. Neighbor mixing mismatches global interaction, and alpha 0.1 has an adverse prior. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Full diagnostic-free SE is the only candidate that preserves the function class distinguishing the positive attention signal. Its low ceiling is accepted because the result cleanly closes the last unresolved variable in that chain.

## Chosen Idea
**Selected**: Diagnostic-Free Full Two-Gate SE Closure

**Why this idea**:
This removes only read-only training observation from the 94.16 treatment, fixing seed 17017 and every model/training semantic. It is not a reroll or parameter rescue. The run proceeds only if matched timing shows a material exposure recovery to at least 137 projected passes.

**Hypothesis**:
Exact diagnostic-free ratio-16 SE on both accepted stage-3 residual branches will preserve EXP-017's initial state/function, project at least 137 passes, and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% if diagnostic overhead was the final constraint on its +0.09 trajectory.

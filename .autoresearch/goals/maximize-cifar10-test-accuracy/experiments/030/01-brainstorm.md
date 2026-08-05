# Brainstorm EXP-030
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): Beta-distributed convex image/label interpolation remains the accepted low-cost generalizer; alpha controls how often mixtures lie near endpoints.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild soft targets can reduce overconfidence, but local evidence warns against consuming the validated hard-label tail.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early-only regularization is compatible with late clean refinement, supporting changes confined to the accepted first 65%.

No network source was consulted; this offline quick pass uses the local knowledge base and complete experiment history.

## Experimental History Review

- EXP-027 remains the 94.32% baseline. Its exact `(2,2,3)` depth, early `N=1,M=5` RandAugment, alpha-0.2 mixup through 65%, and full-stage gradients are protected by the successful interaction and later failures.
- EXP-028 and EXP-029 close the strongest compute/update reallocations: late prefix freezing gained exposure but lost accuracy, while batch 128 could not retain the preregistered full-model image/update regime.
- Mixup duration is bracketed by failed 50% and 75% cutoffs; alpha 0.4 lost 0.50 points; per-example alpha-0.2 coefficients lost 0.28. The weaker alpha-0.1 side is the sole unmeasured one-constant gap, but every neighboring perturbation is negative.
- Broad early residual dropout failed by 0.55 points. Targeted third-block drop-path remains materially narrower but still risks disrupting the accepted depth/RandAugment synergy.
- Label smoothing has no scored local support and extending soft targets after 65% conflicts with the useful hard-label tail. It is retained only as a comparison showing why alpha 0.1 is the cleaner remaining target-regularization test.

## Collected Ideas

## Combinations

## Candidate Ideas

### Label-Smoothed Bridge Before the Final Clean Tail
**Summary**: Keep accepted mixup/RandAugment through 65%, use fixed epsilon-0.02 label smoothing from 65% to 85%, then restore exact hard labels for the final 15%. Preserve every other model, optimizer, seed, and data behavior.

**What it targets**: Remaining near-zero-training / nonzero-test-loss overconfidence using a lower-distortion target regularizer after image mixing ends.

**Reasoning**: Label smoothing can improve calibration and class clustering, while a final clean interval limits under-convergence risk. The temporal separation avoids stacking smoothing directly with mixup.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/004/005/013/020/027.

**Estimated Effort**: low

**Risk Assessment**: It shortens the locally validated 35% hard-label tail, has no scored positive evidence, and lower loss has repeatedly failed to move top-1. Epsilon and bridge endpoints are weakly identified.

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2 ->0.1`. Retain batch-shared sampling, the 65% cutoff, accepted deeper-plus-RandAugment model, batch 256, full FP32 optimizer/schedule, seed, worker RNG isolation, and evaluator. Beta(0.1,0.1) preserves coefficient symmetry while making mixtures more endpoint-heavy.

**What it targets**: Possible excess interpolation strength in the now-deeper accepted model, without changing exposure, duration, coefficient sharing, or the clean tail.

**Reasoning**: Alpha 0.4 establishes the stronger side is harmful, while alpha 0.1 is the remaining clean strength bracket. A single constant gives the most interpretable remaining experiment and has negligible timing risk.

**Sources**: `knowledge/papers/mixup.md`; EXP-002/004/005/015/020/027; `03-experiment-learnings.md`.

**Estimated Effort**: low

**Risk Assessment**: The accepted interaction shows no direct excess-regularization symptom, and every prior mixup change regressed. More endpoint-heavy draws may under-regularize and the likely signal may be below +0.10.

### Early Drop-Path on the Added Block
**Summary**: Apply private-RNG per-example inverted stochastic depth at fixed `p=0.05` only to `layer3[2]` through 65%, then restore exact accepted computation for the hard tail and evaluations. Keep all branch compute and global RNG streams unchanged. Exact developed specification: `experiments/028/proposals/idea-03.md`.

**What it targets**: Robustness of the specific extra block that became useful only under early RandAugment.

**Reasoning**: It is materially narrower than EXP-006: one block, half the probability, and whole-branch per-example masking. It is the only targeted feature regularizer with an explicit accepted-component rationale.

**Sources**: `experiments/028/proposals/idea-03.md`; EXP-006/011/026/027; `knowledge/papers/time-matters-regularization.md`.

**Estimated Effort**: medium

**Risk Assessment**: It stacks a third early regularizer after broad dropout failed, buys no exposure, introduces a private stochastic trajectory, and may be either too weak for +0.10 or harmful to the accepted synergy.

## Review

The reviewer narrowly selected targeted early drop-path at 2.5/5 evidence and 3/5 impact. I adopt the corrections: EXP-011 does not establish that `layer3[2]` learned a brittle shortcut, so this is an exploratory test of sparse early reliance on the newly useful transformation. Planning must preserve the exact `p=0.05`, private seed 28028, one-block placement, eval/hard-tail identity, global-RNG isolation, >=130-pass gate, no adjacent-probability retry, and preregister `final_test_acc >=94.32%` plus loss reporting against best-only overinterpretation. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Drop-path is the only finalist with an interaction-level mechanism and plausible margin-sized upside. Alpha 0.1 is cleaner but tunes a uniformly negative neighborhood with expected effect below +0.10; the label-smoothed bridge directly consumes the validated hard-label tail.

## Chosen Idea
**Selected**: Early Drop-Path on the Added Block

**Why this idea**:
The accepted gain depends on an added third stage-3 block interacting with early RandAugment. A small, isolated per-example whole-branch mask tests whether reduced early reliance improves robustness while preserving the block's full hard-tail and inference capacity. The treatment is materially distinct from broad elementwise dropout but retains its strong negative prior, so the exact one-shot closure rule is essential.

**Hypothesis**:
If sparse early reliance on the added stage-3 transformation improves the depth/RandAugment interaction, then exact private-RNG `p=0.05` drop-path on `layer3[2]` through 65% will retain at least 130 passes, raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%, and retain `final_test_acc >=94.32%`. A valid normal-exposure miss closes this targeted masking family without probability, block, seed, or cutoff tuning.

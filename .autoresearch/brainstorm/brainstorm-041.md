# Brainstorm EXP-041
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`.autoresearch/knowledge/README.md`): CIFAR recipe notes already cover cheap augmentation, schedule, width, throughput, EMA, and initialization levers. No new external source was needed for this local scalar bracket; the decision is driven by the project experiment history.

## Experimental History Review

- Current baseline is `best_test_acc=93.97%` from EXP-038 / commit `755be2c`; the active goal requires at least +0.10 percentage points, so EXP-041 must reach `94.07%` to count.
- The current anchor is `STAGE_WIDTHS=(28,56,112)`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection crop padding, `label_smoothing=0.05`, FP32 channels-last compile.
- EXP-038 showed that increasing weight decay from the label-smoothed reflection anchor to `2e-4` produced the strongest current result, reaching 93.97%.
- EXP-039 showed `3e-4` over-regularizes the current anchor and falls to 93.55%, so larger isolated decay values should be avoided.
- EXP-040 showed increasing initial LR to `0.12` weakens the `2e-4` anchor, peaking at 93.70%; keep `LR=0.1` unless a later experiment has a stronger coupled rationale.
- Failed approach memory says schedule-only second drops, smaller batches, smoothing deviations from 0.05, wider-than-anchor capacity, higher LR, and stronger-than-2e-4 decay are poor next priorities.

## Candidate Ideas

### 1. Weight Decay 1.5e-4 Local Bracket
**Summary**: Change only `WEIGHT_DECAY` from `2e-4` to `1.5e-4`, preserving the rest of the current anchor exactly.

**Reasoning**: EXP-038 validated stronger regularization relative to the prior label-smoothed reflection anchor, while EXP-039 bounded the high side by showing `3e-4` is too strong. A midpoint below `2e-4` tests whether the current best is near the top of a broad regularization basin or whether `2e-4` is slightly over-shrinking weights. The change has no throughput cost and keeps all high-confidence successful components intact.

**Sources**: `.autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-038 and EXP-039; `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md` Patterns and Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: The most likely failure is regression toward the weaker-decay side, especially because EXP-023 showed lower decay hurt an earlier anchor. Worst case is a clean no-improvement with unchanged runtime and no code complexity.

### 2. Initial LR 0.08 on the 2e-4 Anchor
**Summary**: Change only `LR` from `0.1` to `0.08`, leaving milestones unchanged at `[21000, 64000]`.

**Reasoning**: EXP-040 ruled out higher initial LR on the current anchor. Testing the lower side could reduce high-LR noise and make the first post-drop plateau better conditioned. The mechanism is distinct from weight-decay bracketing but risks under-training because the fixed budget is already tight and earlier lower-exploration schedule changes have often underperformed.

**Sources**: EXP-040 row in `.autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md` low-importance LR failure.

**Estimated Effort**: low

**Risk Assessment**: Lower LR may reduce useful exploration before the 21k drop and end below baseline. It is a clean scalar test but has weaker direct evidence than bracketing the successful weight-decay change.

### 3. Short-Window Post-Drop Averaging
**Summary**: Add a lightweight, bounded late-training weight averaging mechanism that only keeps a short window after the first LR drop, avoiding long equal averaging.

**Reasoning**: Late plateau noise could be limiting peak accuracy, and averaging can sometimes stabilize SGD solutions. However, prior EXP-004 per-step EMA had overhead, and EXP-021 naive equal averaging collapsed as snapshots accumulated. A short bounded window is a different approach, but it is more code-heavy and more failure-prone than scalar retuning.

**Sources**: `.autoresearch/knowledge/README.md` EMA reference; EXP-004 and EXP-021 failed approach entries in `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: Implementation can add overhead, disturb evaluation semantics, or underperform without BN recalibration. It should wait until simpler no-overhead brackets are exhausted.

## Idea Evaluation

The strongest local evidence is around weight decay. EXP-038 is the most recent true improvement, and EXP-039 already bounded the high side. Testing `1.5e-4` is not a repeat of a failed exact setting: it preserves the successful label smoothing and reflection padding anchor while probing the remaining untested interval between the prior `1e-4` anchor and the new `2e-4` best. It has the clearest mechanism, no runtime risk, and a clean interpretation.

Lower LR is also a reasonable optimizer-dynamics bracket after EXP-040, but its mechanism is less directly supported by prior wins and may reduce useful high-LR exploration under a fixed 300-second budget. Short-window averaging has more potential novelty but carries higher implementation and measurement risk, especially given the existing EMA and averaging failures.

## Chosen Idea
**Selected**: Weight Decay 1.5e-4 Local Bracket

**Why this idea**:
It is the lowest-risk, highest-clarity follow-up to the current successful EXP-038 result. It directly tests whether the active `2e-4` regularization anchor can be improved by a slightly softer decay while preserving every other validated component.

**Hypothesis**:
If `2e-4` is slightly above the optimum for the label-smoothed reflection anchor, then `WEIGHT_DECAY = 1.5e-4` will retain the generalization gain while reducing over-shrinkage, producing `best_test_acc >= 94.07%`. If the EXP-038 gain requires the full stronger decay, the run should cleanly classify as `no-improvement`.

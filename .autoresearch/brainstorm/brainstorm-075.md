# Brainstorm EXP-075
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix is the validated regional-mixing mechanism in this repo; the current best EXP-064 keeps `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **ResNet initialization context** (`reports/exp-report-072.md`, `references/resnet-zero-init-residual.md`)
  Initialization-only changes can affect CIFAR ResNet optimization without changing parameter count, runtime architecture, or evaluation.
- **Recent CutMix target/timing probes** (`reports/exp-report-073.md`, `reports/exp-report-074.md`)
  CutMix warmup and endpoint hard labels each gave positive but sub-threshold signals, so isolated CutMix-internal refinements are likely too small.

## Experimental History Review

- Current best remains EXP-064: probabilistic CutMix reached `best_test_acc=94.11%`; improvement now requires `best_test_acc >= 94.21%`.
- Static CutMix strength is bracketed: probability 0.25/0.75 and alpha 0.5/2.0 all failed to clear the threshold. Keep `CUTMIX_PROB=0.5` and `CUTMIX_ALPHA=1.0`.
- CutMix timing and target-softness probes are weakly positive but sub-threshold: EXP-073 warmup reached 94.14%, and EXP-074 hard CutMix endpoints reached 94.17%.
- EXP-072 fan-out Conv2d initialization reached 94.16% with unchanged throughput and anchor settings. It is the strongest non-CutMix near-miss and mechanically distinct from target smoothing.
- Goal learnings now mark label-smoothing deviations as a high-importance recurring failure. That discourages further isolated smoothing variants, but does not fully rule out a coupled test with an independent initialization mechanism.
- Recurring failures still discourage isolated schedule-only second drops, weight averaging, batch-size deviations, global label-smoothing deviations, direct mixup, policy augmentation, SE gates, no-decay parameter groups, and scalar LR or weight-decay retuning.

## Candidate Ideas

### 1. Fan-Out Conv Init Plus CutMix Endpoint Hard Labels
**Summary**: Combine EXP-072's Conv2d fan-out ReLU Kaiming initialization with EXP-074's CutMix endpoint hard-label setting. Keep clean batches at label smoothing 0.05, CutMix `alpha=1.0`, `prob=0.5`, the 21k first LR drop, and every data/architecture/optimizer anchor unchanged.

**Reasoning**: EXP-072 and EXP-074 are the two best recent near-misses: 94.16% and 94.17%. Their mechanisms are distinct enough to justify one coupled test: fan-out initialization changes residual signal scaling at startup, while hard CutMix endpoints change the sharpness of mixed-batch supervision. Neither changes throughput, parameter count, validation cadence, or the successful regional mixing exposure. If their small effects are complementary rather than pure noise, the combined run is one of the few remaining low-overhead paths that could plausibly cross 94.21%.

**Sources**: `reports/exp-report-072.md`, `reports/exp-report-074.md`, `goal-learnings/maximize-cifar10-best-test-accuracy.md`, `knowledge/papers/cutmix-regularization.md`

**Estimated Effort**: low

**Risk Assessment**: This risks additive-noise chasing, and the label-smoothing-deviation family is now a recurring failure. A no-improvement result would not be surprising; it would close the most obvious coupled near-miss path.

### 2. Fan-Out Conv Init Plus Shorter CutMix Warmup
**Summary**: Combine Conv2d fan-out initialization with a shorter early clean warmup, such as 1000 steps before enabling the static CutMix branch. Keep endpoint smoothing at 0.05 and all anchor settings unchanged.

**Reasoning**: EXP-073's 2000-step clean warmup produced a small positive result, but likely removed too much early CutMix exposure. A shorter warmup could avoid the most unstable first updates while restoring regional regularization earlier. Fan-out initialization may also stabilize early residual learning, making the warmup less harmful than a pure CutMix timing change.

**Sources**: `reports/exp-report-072.md`, `reports/exp-report-073.md`, `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: The temporal CutMix family is already weak and likely sub-threshold. Combining two small effects could still miss the threshold, and if it succeeds attribution would be less clear between early timing and initialization.

### 3. Classifier Initialization Calibration
**Summary**: Keep the CutMix anchor unchanged and adjust only the final classifier initialization, for example using a smaller normal scale or zeroing the classifier bias while leaving Conv2d initialization unchanged.

**Reasoning**: The current `_weights_init` applies Kaiming normal initialization to both Conv2d and Linear weights even though the final Linear layer is not followed by ReLU. A classifier-specific calibration might improve late separability or reduce early logit scale noise without affecting throughput or architecture.

**Sources**: `train.py` `_weights_init`, `reports/exp-report-072.md`

**Estimated Effort**: low

**Risk Assessment**: Expected impact is likely smaller than the Conv2d fan-out result and may be pure noise. It is safer scientifically than another target-smoothing tweak, but probably less likely to clear the +0.10pp threshold alone.

## Idea Evaluation

The coupled fan-out plus hard CutMix endpoint test has the highest expected impact because it joins the two strongest recent sub-threshold signals with independent mechanisms and no throughput cost. It is not a clean single-mechanism probe, but the goal now needs a meaningful jump over 94.11%, and isolated low-overhead changes have repeatedly landed in the 94.14-94.17% range.

Fan-out plus shorter warmup is also defensible, but EXP-073 already suggests early CutMix timing has a smaller effect size than endpoint hard labels. It also weakens the successful static CutMix exposure, whereas candidate 1 preserves CutMix timing exactly.

Classifier initialization is the cleanest untested initialization probe, but its expected effect is probably too small to beat the current threshold on its own. It remains a good fallback if the coupled near-miss path fails and future brainstorming wants a narrow non-augmentation test.

## Chosen Idea
**Selected**: Fan-Out Conv Init Plus CutMix Endpoint Hard Labels

**Why this idea**:
It is the best available low-overhead coupled test: EXP-072 and EXP-074 each moved above baseline without changing runtime cost, and their mechanisms target different parts of training. The experiment also preserves the validated static CutMix exposure and every hard constraint. The result will either clear the threshold or usefully close the most obvious near-miss combination.

**Hypothesis**:
If Conv2d fan-out initialization improves residual signal scaling and hard CutMix endpoints sharpen mixed-batch supervision in complementary ways, then combining them while preserving the CutMix anchor will improve `best_test_acc` from 94.11% to at least 94.21%.

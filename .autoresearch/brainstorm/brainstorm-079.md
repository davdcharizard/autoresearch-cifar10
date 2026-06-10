# Brainstorm EXP-079
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix is already the best validated augmentation family in this loop: regional patch mixing keeps local pixels realistic while using area-weighted label mixing. The local anchor uses `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint label smoothing 0.05.
- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  CIFAR residual architectures can benefit from topology/width changes, but fixed-budget local history now shows isolated residual-topology changes often trade away useful steps or underperform the CutMix anchor.
- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  The current FP32 compile/channels-last path remains the best local execution mode; new ideas should preserve it unless the experiment explicitly tests throughput.

No new external searches were added. The existing knowledge base plus 79 local experiments provides stronger guidance than broad generic sources for this narrow CIFAR-10 benchmark.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%` from commit `1119ff8`; the active +0.10pp noise guard requires `best_test_acc >= 94.21%`.
- EXP-065 through EXP-068 bracketed static CutMix probability and alpha. `p=0.25`, `p=0.75`, `alpha=0.5`, and `alpha=2.0` all missed the 94.21% threshold, so the full-run static CutMix anchor remains `p=0.5`, `alpha=1.0`.
- EXP-069 showed post-drop CutMix weakening is harmful, peaking at 93.73%; therefore any CutMix schedule should preserve the validated post-drop `p=0.5` behavior.
- EXP-073's 2000-step clean warmup reached 94.14%, a sub-threshold positive signal. This suggests early CutMix exposure may matter slightly, but fully disabling CutMix for 2000 steps is not enough.
- EXP-072 fan-out initialization and EXP-074 hard CutMix endpoints produced near-misses, but EXP-075 showed those signals were not additive. EXP-076 and EXP-078 further weakened classifier-head and block-topology paths.
- High/medium failed families now include isolated label-smoothing deviations, weight averaging, second-drop/cosine schedule probes, batch-size deviations, classifier-head tweaks, policy augmentation, SE gates, transition smoothing, and static CutMix strength brackets.

## Candidate Ideas

### 1. Short CutMix Probability Ramp
**Summary**: Preserve the current CutMix anchor long-run, but linearly ramp the active CutMix probability from 0.25 to 0.5 over the first 1000 optimizer steps. Keep `CUTMIX_ALPHA=1.0`, endpoint label smoothing 0.05, full `p=0.5` after the ramp, stage widths, optimizer, LR milestones, transforms, compile/channels-last, batch size, seed, and validation cadence unchanged.

**Reasoning**: EXP-073's clean 2000-step warmup reached 94.14%, indicating early mixed-label regional noise may be slightly too strong. Static `p=0.25` failed because it weakens CutMix for the entire run, while post-drop tapering failed because late full-strength CutMix appears important. A short probability ramp targets only the unstable first updates while still exposing the model to CutMix from step 0 and restoring the validated anchor before the first LR drop.

**Sources**: `reports/exp-report-073.md`; `reports/exp-report-065.md`; `reports/exp-report-069.md`; `knowledge/papers/cutmix-regularization.md`; `train.py` CutMix loop

**Estimated Effort**: low

**Risk Assessment**: Expected effect is likely small and may land in the same near-threshold band as EXP-073. Code risk is low: one helper or inline scalar controls the existing CutMix Bernoulli probability, with no evaluation or data-pipeline changes.

### 2. Very Short Linear LR Warmup
**Summary**: Preserve the existing step schedule and milestones but linearly ramp LR from 0.02 to 0.1 over the first 500 optimizer steps, then keep the current `LR=0.1` and milestones `[21000, 64000]`.

**Reasoning**: Several recent near-misses point to early training stability as a plausible remaining bottleneck. LR warmup is distinct from moving the first LR drop or adding a second drop because it changes only the first few hundred high-variance updates and preserves the post-drop refinement phase. It could stabilize early CutMix batches without changing CutMix semantics.

**Sources**: `reports/exp-report-073.md`; `reports/exp-report-078.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` scheduler setup

**Estimated Effort**: low-to-medium

**Risk Assessment**: Schedule-only work has a long failure history, so this should not be selected unless the CutMix-specific option is exhausted. A warmup can also undertrain the earliest phase and may not recover within the fixed budget.

### 3. Narrow Layer3 Width Rebalance
**Summary**: Keep ResNet-20 depth and the first two stage widths unchanged, but reduce only the final stage from 112 to 104 channels: `STAGE_WIDTHS=(28, 56, 104)`. Preserve the CutMix anchor, optimizer, LR schedule, transforms, compile/channels-last, batch size, and validation cadence.

**Reasoning**: EXP-078 completed fewer steps than the anchor despite only a tiny parameter change, reinforcing that fixed-budget step coverage matters. A small final-stage reduction could trade a little representation capacity for more post-drop optimization while staying closer to the successful 28/56/112 anchor than prior broad capacity changes.

**Sources**: `reports/exp-report-078.md`; `reports/exp-report-062.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/papers/wide-residual-networks.md`

**Estimated Effort**: low

**Risk Assessment**: Capacity changes have a mixed-to-weak history, and reducing final-stage channels may simply lower representational ceiling. This is safer than broad topology rewrites but less directly supported than the CutMix ramp.

## Idea Evaluation

The short CutMix probability ramp has the best evidence-to-risk balance. It is not a static CutMix bracket, because it restores `p=0.5` after the first 1000 updates; it is not the failed post-drop taper, because it preserves full late CutMix; and it is narrower than EXP-073's clean warmup, because CutMix remains possible from the start. Its mechanism is specific: reduce early regional label noise just enough to help representation formation while keeping the validated regional regularizer for nearly the entire run.

The LR warmup has a plausible early-stability mechanism, but it belongs close to the schedule family that repeatedly underperformed. It may become useful later, especially if the CutMix ramp also suggests early instability, but it should not be the immediate next experiment while a more CutMix-specific temporal test remains.

The layer3 width rebalance is simple and addresses fixed-budget step count, but local capacity evidence is weak. The current best appears to rely on the 28/56/112 capacity plus CutMix recipe, and reducing the final stage risks another valid no-improvement with a lower ceiling.

## Chosen Idea
**Selected**: Short CutMix Probability Ramp

**Why this idea**:
It targets the strongest remaining local signal: EXP-073's early-CutMix near-miss, while avoiding the exact failure modes of static low-probability CutMix and post-drop CutMix tapering. It is a small `train.py`-only change with low implementation risk and preserves the validated anchor for almost the full fixed-budget run.

**Hypothesis**:
If the current anchor is slightly hurt by too much regional mixed-label noise during the first few hundred unstable updates, then ramping `CUTMIX_PROB` from 0.25 to 0.5 over the first 1000 steps will improve post-drop convergence enough to raise `best_test_acc` from 94.11% to at least 94.21%.

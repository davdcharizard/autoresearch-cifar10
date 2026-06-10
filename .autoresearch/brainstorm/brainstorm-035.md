# Brainstorm EXP-035
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-032 report** (`reports/exp-report-032.md`)
  `label_smoothing=0.05` preserved throughput and improved the reflection anchor to 93.70%, establishing mild smoothing as the current anchor.
- **EXP-033 report** (`reports/exp-report-033.md`)
  Lowering smoothing to 0.03 reached 93.79%, only 0.01 below the required 93.80% threshold, suggesting milder smoothing is directionally useful but insufficient alone.
- **EXP-034 report** (`reports/exp-report-034.md`)
  Moving the first LR drop to 22k also reached 93.79%, again 0.01 below threshold, suggesting the label-smoothed recipe may sit on a narrow local plateau.
- **Existing schedule knowledge** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Schedule shape and timing can matter under fixed step budgets, but local evidence warns against broad schedule-only changes and second-drop retuning.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the goal's +0.10 percentage-point rule, EXP-035 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-033 and EXP-034 each reached 93.79% through different one-scalar edits: lower smoothing and later first drop. Both are valid no-improvements under the noise rule, but both are the closest post-baseline signals.
- High-importance failed families still apply: avoid widening beyond 28/56/112 and avoid isolated second LR-drop retuning.
- Cutout, aggressive smaller batch size 96, higher momentum, lower weight decay, no-decay BN/bias, projection shortcuts, zero-gamma initialization, and padding-mode siblings have all underperformed.
- The remaining plausible space is either a coupled use of the two near-miss scalar edits or a distinct but riskier stochasticity/regularization probe.

## Candidate Ideas

### 1. Combine Lower Smoothing with 22k First Drop
**Summary**: Preserve the reflection-padding 28/56/112 anchor but apply both recent near-miss scalar edits: change `label_smoothing=0.05` to `0.03` and `LR_MILESTONES = [21000, 64000]` to `[22000, 64000]`.

**Reasoning**: EXP-033 and EXP-034 independently reached 93.79%, only one hundredth below the threshold, through different mechanisms. Lower smoothing may improve peak class separation, while the 22k drop may give the smoothed model more high-LR fitting before refinement. Combining them is a small coupled probe, not an exact retry of either failed single-axis experiment, and it preserves throughput, model size, augmentation, optimizer, and validation cadence.

**Sources**: `reports/exp-report-033.md`; `reports/exp-report-034.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The two near-misses may not be additive and could simply reproduce the same 93.79 plateau. Because the change is still narrow and fully reversible, the expected failure mode is a clean no-improvement result.

### 2. Mild Batch Size 112 on the Current Anchor
**Summary**: Preserve `label_smoothing=0.05`, reflection padding, architecture, optimizer, and 21k schedule, but reduce `BATCH_SIZE` from 128 to 112.

**Reasoning**: A small increase in gradient noise might help the regularized anchor reach a better late peak. EXP-025 showed batch size 96 was too aggressive and lost useful budget, but 112 is a milder version that may still hit the 21k first drop and preserve enough post-drop refinement.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` rows EXP-025 and EXP-032; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: This neighbors a known failed approach and changes both stochasticity and throughput. If step budget drops materially, the run may miss the threshold despite any regularization benefit.

### 3. Raise Label Smoothing to 0.08
**Summary**: Preserve the current anchor and increase only the smoothing value from `0.05` to `0.08`.

**Reasoning**: EXP-032 validated mild label smoothing, and stronger smoothing could further stabilize late post-drop evaluations. This tests whether the useful direction is more confidence regularization rather than milder regularization.

**Sources**: `reports/exp-report-032.md`; `reports/exp-report-033.md`; `brainstorm/brainstorm-033.md`.

**Estimated Effort**: low

**Risk Assessment**: Stronger smoothing risks suppressing top-1 peak accuracy and moves toward the stronger-regularization family that failed in EXP-000. It is simple but less supported than composing the two nearest local signals.

## Idea Evaluation

The combined lower-smoothing plus 22k first-drop probe has the strongest local evidence: two independent, low-overhead changes both landed exactly at 93.79%, just below the tightened improvement threshold. The mechanisms are compatible rather than redundant: lower smoothing can sharpen class separation, while a later first drop gives the smoothed model more high-LR fitting time before LR 0.01 refinement. This is more plausible than another isolated micro-tweak and less throughput-risky than batch size 112.

Batch size 112 remains a reasonable future experiment, but it changes step throughput and optimization noise together and sits near a failed batch-size-96 result. Stronger smoothing to 0.08 is easy to test, but it moves toward stronger regularization after lower smoothing already produced the closest result.

EXP-035 should therefore combine the two 93.79 near-misses while preserving all other anchor choices. If the combination still fails, future brainstorms should move to a genuinely distinct lever such as batch size 112 or a carefully bounded late-averaging mechanism.

## Chosen Idea
**Selected**: Combine Lower Smoothing with 22k First Drop

**Why this idea**:
It composes the two closest valid no-improvement signals without adding runtime overhead, model capacity, dependencies, or evaluation changes. The experiment is narrow enough to attribute while testing a plausible interaction between regularization strength and first-drop timing.

**Hypothesis**:
Using `label_smoothing=0.03` with `LR_MILESTONES = [22000, 64000]` will combine sharper class separation with slightly longer high-LR fitting and raise `best_test_acc` from 93.70% to at least 93.80%.

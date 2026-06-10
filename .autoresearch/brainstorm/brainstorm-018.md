# Brainstorm EXP-018
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-016 28/56/112 21k schedule report** (`reports/exp-report-016.md`)
  The current best recipe uses 28/56/112 channels with a 21k first LR drop, reaching 93.23% with 34,208 steps.

- **EXP-017 30/60/120 width report** (`reports/exp-report-017.md`)
  Broad width scaling to 30/60/120 reduced the step budget to 27,400 and peaked at 93.16%, so the next idea should add representational power with much less compute growth.

- **Current ResNet implementation** (`train.py`)
  Downsample shortcuts currently use zero-padding after strided slicing. This is cheap, but the stage-transition shortcut has no learned mapping when channel count changes.

No new external search was needed. The next decision is driven by local evidence that broad widening is now throughput-limited, plus direct inspection of the current residual shortcut implementation.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the tightened rule, EXP-018 must reach at least `93.33%`.
- Width scaling was the strongest positive axis through 28/56/112, but EXP-017 showed the next broad width step is too expensive under the fixed 300s budget.
- The 28/56/112 recipe with 21k first drop is the current schedule anchor. It completed 34,208 steps, while 30/60/120 completed only 27,400 steps.
- Failed approaches now include regularization, Nesterov, reduced precision, per-step EMA, late 23k schedule on 28/56/112, and 30/60/120 broad widening.
- The immediate gap is to add accuracy capacity without sharply reducing step budget. The current code has only two downsample transition shortcuts, so learned projections may be a small, targeted architecture change.

## Candidate Ideas

### 1. Learned Projection Shortcuts at Downsample Transitions
**Summary**: Keep `STAGE_WIDTHS = (28, 56, 112)` and `LR_MILESTONES = [21000, 64000]`, but replace the current zero-pad shortcut for stride/channel-change blocks with a learned 1x1 convolution plus BatchNorm shortcut. Only the two transition blocks should use the projection; same-shape residual blocks remain identity shortcuts.

**Reasoning**: EXP-017 showed that adding capacity everywhere is too slow. Projection shortcuts add a small amount of capacity only at the two stage transitions, where the current shortcut is non-learned and channel-padded. This may improve feature transfer between stages while preserving nearly all of the current step budget.

**Sources**: `train.py` shortcut implementation; `reports/exp-report-016.md`; `reports/exp-report-017.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The effect size may be below the +0.10 point threshold, and BatchNorm in the shortcut may slightly affect compile/runtime. Worst case is a valid no-improvement with limited throughput loss.

### 2. Smaller Width Step 29/58/116 with 19k First Drop
**Summary**: Increase stage widths from `(28, 56, 112)` to `(29, 58, 116)` and move the first LR drop to 19000. This tests whether a very small width increment can preserve more throughput than 30/60/120 while still adding capacity.

**Reasoning**: Width scaling worked until EXP-017. A smaller increment may avoid the 27.4k-step collapse, and a 19k first drop would leave more LR 0.01 refinement time. This keeps the validated width/schedule axis alive with lower risk than another broad width jump.

**Sources**: `reports/exp-report-016.md`; `reports/exp-report-017.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: This still broadens every stage, so it may repeat the EXP-017 throughput failure. It also changes both width and schedule, making attribution less clean.

### 3. Sparse Late Averaging on 28/56/112 with 21k Schedule
**Summary**: Keep the current architecture and schedule, but maintain an averaged model after the first LR drop with low-frequency updates, such as once per epoch, then evaluate the averaged weights at epoch boundaries.

**Reasoning**: EXP-016 peaked above its final accuracy, and EXP-017 also had late low-LR fluctuations. Sparse averaging could smooth the selected weights without the per-step overhead that hurt EXP-004.

**Sources**: `reports/exp-report-004.md`; `reports/exp-report-016.md`; `reports/exp-report-017.md`; `knowledge/references/pytorch-ema-averaging.md`.

**Estimated Effort**: medium

**Risk Assessment**: Implementation risk is higher because weight swapping and BatchNorm state can compromise result validity. It may also add evaluation complexity under the once-per-epoch validation constraint.

## Idea Evaluation

Projection shortcuts are the best fit after EXP-017. The main lesson from EXP-017 is not that all capacity is bad, but that broad capacity increases are now too expensive under the fixed budget. Learned transition shortcuts target exactly two blocks and should add far less compute than widening every convolution. The mechanism is clear: learn the residual mapping across spatial/downchannel transitions instead of relying on strided slicing plus zero padding.

The 29/58/116 width step remains plausible, but it risks repeating the newly observed failure mode. If projection shortcuts fail cleanly, a smaller width step may be worth trying with a more aggressive schedule, but it is less attractive immediately after a broad-width no-improvement.

Sparse late averaging targets a real best/final gap, but it has more implementation risk and a prior nearby failure mechanism in EXP-004. It should wait until the low-overhead architecture path is tested.

## Chosen Idea
**Selected**: Learned Projection Shortcuts at Downsample Transitions

**Why this idea**:
It is a targeted architecture change on the current best 28/56/112, 21k recipe that may improve stage-transition representations without the large step-budget penalty seen in EXP-017. It stays within `train.py`, preserves the evaluator and validation cadence, and directly addresses the current bottleneck: adding useful capacity cheaply.

**Hypothesis**:
Adding learned 1x1 projection shortcuts only for downsample/channel-change blocks will preserve most of the 28/56/112 step budget while improving feature transfer enough to reach at least `93.33%` `best_test_acc`.

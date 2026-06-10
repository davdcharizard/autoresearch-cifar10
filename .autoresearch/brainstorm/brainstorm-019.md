# Brainstorm EXP-019
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening residual networks is a validated CIFAR architecture axis, but this project must calibrate width to the fixed 300s budget rather than blindly increasing capacity.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Averaged weights can be maintained without changing the optimizer or harness, but prior per-step EMA overhead in this project makes low-frequency variants the only credible retry.

- **EXP-016, EXP-017, EXP-018 reports** (`reports/exp-report-016.md`, `reports/exp-report-017.md`, `reports/exp-report-018.md`)
  The current best is 28/56/112 with a 21k first LR drop. 30/60/120 was too slow, while projection shortcuts preserved throughput but hurt accuracy, so the next idea should either use a smaller width increment or a low-overhead late-training mechanism.

No new external search was needed. Existing knowledge and the last three experiment reports provide enough signal for the next local experiment.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; the goal now requires at least `93.33%` for EXP-019 to count as an improvement.
- Width scaling remains the strongest validated axis: 20/40/80, 24/48/96, and 28/56/112 each produced improvements when paired with calibrated first LR drops.
- The next broad width step, 30/60/120 with a 20k first drop, peaked at 93.16% and completed only 27,400 steps. This suggests the 30/60/120 capacity increment is too costly under the fixed budget, not that all further capacity is impossible.
- Projection shortcuts at 28/56/112 reached 38,322 steps but peaked at 92.97%, so transition shortcut capacity is not the useful targeted-capacity lever.
- EXP-016 and EXP-018 both show late best/final gaps under the 21k schedule, so low-overhead averaging remains a plausible but implementation-riskier route.
- Known failures to avoid include recurring cutout, Nesterov, reduced precision, per-step EMA overhead, 23k first drop at 28/56/112, broad 30/60/120 widening, and projection shortcuts.

## Candidate Ideas

### 1. Minimal Width Step 29/58/116 with 19k First LR Drop
**Summary**: Increase `STAGE_WIDTHS` from `(28, 56, 112)` to `(29, 58, 116)` and move the first LR milestone from 21000 to 19000, keeping the second milestone unreachable at 64000. This is a smaller capacity step than EXP-017's `(30, 60, 120)` and gives more LR 0.01 refinement time.

**Reasoning**: Width scaling has produced every major improvement so far, but EXP-017 showed that the 30/60/120 step loses too much throughput. A one-channel base-width increment is a lower-compute way to test whether any width headroom remains. Moving the first drop to 19k compensates for the expected step reduction and should leave more post-drop refinement than EXP-017's 20k drop did.

**Sources**: `goal-learnings/maximize-cifar10-best-test-accuracy.md` patterns and failed approaches; `reports/exp-report-016.md`; `reports/exp-report-017.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: It may still reduce throughput enough to miss the threshold, or 29/58/116 may simply not add useful capacity beyond 28/56/112. The worst case is a valid no-improvement with a clear step-budget reading.

### 2. Final-Stage-Only Widening to 28/56/128
**Summary**: Keep the first two stages at `(28, 56)` and widen only the final stage to 128 channels, likely with the 21k first LR drop or a modestly earlier 20k drop. This targets high-level capacity where spatial maps are smallest, avoiding a broad early-stage compute increase.

**Reasoning**: EXP-017's broad width increase was costly, while EXP-018's projection shortcuts did not help. Final-stage-only widening is a different targeted-capacity strategy: add representational power in the 8x8 stage where convolution FLOPs are cheaper than in the 32x32 and 16x16 stages. It may preserve more throughput than widening all stages.

**Sources**: `train.py` stage structure; `reports/exp-report-017.md`; `reports/exp-report-018.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: The nonuniform stage ratio may be less effective than proportional widening, and parameter count may still rise substantially in stage 3. The schedule choice would also be less directly grounded than the 29/58/116, 19k candidate.

### 3. Sparse Late Weight Averaging After the 21k Drop
**Summary**: Keep the current 28/56/112 architecture and 21k schedule, but maintain a low-frequency averaged model after the first LR drop, such as once per epoch, and evaluate the averaged weights once per epoch instead of the raw model.

**Reasoning**: Several recent runs have best/final gaps after the LR drop, suggesting late low-LR weights fluctuate around useful basins. Per-step EMA failed partly because it added overhead; a sparse post-drop averaging scheme could smooth final weights with much less update cost while preserving the validation cadence.

**Sources**: `reports/exp-report-004.md`; `reports/exp-report-016.md`; `reports/exp-report-018.md`; `knowledge/references/pytorch-ema-averaging.md`.

**Estimated Effort**: medium

**Risk Assessment**: Implementation is riskier than a constant change because averaging BatchNorm state and swapping/evaluating weights can compromise validity. If done incorrectly, it could become invalid or reduce throughput enough to repeat the per-step EMA failure.

## Idea Evaluation

The minimal width step has the strongest project-specific evidence. All major improvements have come from width scaling plus first-drop calibration, and EXP-017 missed by only 0.07 points despite a large throughput loss. A smaller width increment directly addresses that failure mechanism while staying on the validated axis. The 19k first drop is also a clear schedule response: if the new width reduces the step budget, more LR 0.01 time is needed to reach the accuracy plateau.

Final-stage-only widening is appealing because it targets cheaper spatial resolution, but it is less directly validated. It changes the stage-width ratio and may add many parameters in stage 3 without a known schedule anchor. It is a good fallback if a smaller proportional width step still underperforms because of throughput.

Sparse late averaging targets a real late-training dynamic, but the implementation risk is higher and prior EMA overhead is a known failed approach. It should be planned carefully later, not immediately after two architecture no-improvements, unless width calibration is exhausted.

## Chosen Idea
**Selected**: Minimal Width Step 29/58/116 with 19k First LR Drop

**Why this idea**:
It is the most conservative continuation of the strongest successful pattern. It tests whether width scaling has one more usable increment while explicitly addressing EXP-017's throughput failure with a smaller model and earlier first drop. The change is simple, isolated to `train.py`, and easy to interpret from steps, epochs, and peak accuracy.

**Hypothesis**:
Changing to `STAGE_WIDTHS = (29, 58, 116)` and `LR_MILESTONES = [19000, 64000]` will preserve enough of the fixed-budget step count while adding capacity, allowing `best_test_acc` to reach at least `93.33%`.

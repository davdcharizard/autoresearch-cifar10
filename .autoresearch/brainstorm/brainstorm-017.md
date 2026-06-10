# Brainstorm EXP-017
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  The saved WRN note says widening residual networks is a promising CIFAR capacity direction after recipe changes plateau, but fixed-budget runs need careful runtime checks.

- **EXP-014 ResNet-20 width 1.75x report** (`reports/exp-report-014.md`)
  A 28/56/112 ResNet-20 with a 22k first drop reached 93.09%, showing width scaling still paid off despite fewer steps.

- **EXP-016 21k schedule report** (`reports/exp-report-016.md`)
  Moving the 28/56/112 first drop from 22k to 21k improved the baseline to 93.23%, showing this width benefits from earlier LR 0.01 refinement.

No new external search was needed. The next decision is guided by the local width/schedule trajectory plus the existing WRN knowledge entry.

## Experimental History Review

- The current baseline is EXP-016 at `best_test_acc=93.23%`; with the tightened rule, EXP-017 must reach at least `93.33%`.
- The strongest positive pattern is width scaling with schedule calibration: 20/40/80 reached 92.12%, 24/48/96 reached 92.49%, and 28/56/112 reached 93.09%.
- Schedule calibration is width-specific. A 22k drop was too early for 20/40/80 in EXP-012, but 22k worked for 28/56/112 in EXP-014 and 21k improved it further in EXP-016.
- The 23k retune on 28/56/112 failed in EXP-015 despite more total steps, so delaying the first drop is a known bad direction for the current width.
- Failed regularization and optimizer tweaks have generally not moved the metric: cutout, Nesterov, TF32, BF16, and per-step EMA were no-improvement. Capacity scaling is the clearest remaining high-evidence axis.
- The main unexplored gap is whether a modest next width step can clear 93.33% if its first LR drop is moved earlier enough to preserve post-drop refinement time.

## Candidate Ideas

### 1. ResNet-20 Width 30/60/120 with 20k First LR Drop
**Summary**: Increase `STAGE_WIDTHS` from `(28, 56, 112)` to `(30, 60, 120)` and move the first LR milestone from 21000 to 20000. Keep depth, optimizer, augmentation, precision, compile/channels-last settings, batch size, seed, and once-per-epoch evaluation unchanged.

**Reasoning**: Width scaling has delivered the largest gains so far, and EXP-016 shows the current 28/56/112 model benefits from slightly earlier low-LR refinement. A small capacity step should raise the accuracy ceiling while the 20k first drop compensates for expected lower throughput. This is the most direct extension of the validated pattern without jumping to a much slower model.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-014.md`; `reports/exp-report-016.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The wider model may complete too few steps or need a different first-drop point than 20k. Worst case is a clean no-improvement if the added capacity undertrains or if 20k is too early.

### 2. Sparse Late Averaging on 28/56/112 with 21k Schedule
**Summary**: Keep the EXP-016 architecture and schedule, but maintain a low-frequency averaged model after the first LR drop, such as updating once per epoch and evaluating the averaged weights at epoch boundaries.

**Reasoning**: EXP-016 peaked at 93.23% but ended at 93.03%, so late weights fluctuate below the best checkpoint. Sparse averaging might smooth low-LR noise without the per-step overhead that made EXP-004 miss the tightened threshold.

**Sources**: `reports/exp-report-004.md`; `reports/exp-report-016.md`; `knowledge/references/pytorch-ema-averaging.md`.

**Estimated Effort**: medium

**Risk Assessment**: BatchNorm state handling and evaluation-time weight swapping are easy to get subtly wrong. Even sparse averaging can add enough overhead or validation complexity to violate the spirit of the fixed-budget comparison.

### 3. Projection Shortcuts at Downsample Transitions
**Summary**: Replace the current zero-pad shortcut for stride/channel changes with learned 1x1 projection shortcuts in the two downsample blocks, keeping the 28/56/112 width and 21k schedule.

**Reasoning**: The current implementation uses the original CIFAR option-A style shortcut. Learned projections may improve feature transfer across stage transitions with only two extra shortcut convolutions, potentially increasing accuracy without changing depth or the main convolutional path.

**Sources**: `train.py` shortcut implementation; `reports/exp-report-016.md`; ResNet architecture context in `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: medium

**Risk Assessment**: Projection shortcuts add parameters and compute at high-value transition points, but the accuracy gain is uncertain under this small fixed-time benchmark. It could also make attribution harder because it changes the residual path rather than the validated width/schedule axis.

## Idea Evaluation

The 30/60/120 width step has the strongest evidence and clearest mechanism. The last several successful improvements came from adding capacity while calibrating the first LR drop to the slower step budget. EXP-016 specifically moved the best schedule anchor earlier to 21k, which makes a 20k first drop for a modestly wider model a defensible next extrapolation. The expected impact is also large enough to plausibly clear the new 93.33% threshold.

Sparse late averaging targets a real observation, the peak/final gap, but its implementation risk is higher. EXP-004 already showed that per-step EMA overhead can turn a small apparent gain into no-improvement under the noise-margin rule. A carefully sparse version may still be worthwhile, but it is a better follow-up after the width axis plateaus.

Projection shortcuts are plausible but less directly supported by the local evidence. They change the architecture in a more targeted way than width scaling, but the likely effect size is uncertain and may be below the +0.10 point threshold.

## Chosen Idea
**Selected**: ResNet-20 Width 30/60/120 with 20k First LR Drop

**Why this idea**:
It extends the best-supported pattern: modest ResNet-20 width increases have repeatedly improved the metric when the first LR drop is calibrated earlier for the reduced step budget. The move from 28/56/112 to 30/60/120 is small enough to avoid a large undertraining risk, and the 20k drop follows the successful EXP-016 schedule direction.

**Hypothesis**:
A 30/60/120 ResNet-20 with first LR drop at step 20000 will preserve enough LR 0.01 refinement time to turn the extra capacity into a new best, reaching at least `93.33%` `best_test_acc`.

# Brainstorm EXP-006
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new sources needed. The key insight is entirely from the experimental trajectory — EXP-005 showed that FP16 is unstable at LR=0.01 (epochs 34-52 oscillated 68-82%) but the extended LR=0.001 phase (epochs 52-106) delivered +3.5pp. Optimizing the schedule for FP16's precision characteristics is the next step.

## Experimental History Review

Current baseline: **94.44%** (EXP-005, width-2x + aug + WD=5e-4 + AMP, 106 epochs, commit b934204).

Trajectory: BASE 91.72% → 001 92.29% → 002 92.92% → 003 93.33% → 004 93.28% (fail) → 005 94.44%. Total: +2.72pp.

Critical pattern from EXP-005 (High Importance): FP16 is unstable at LR=0.01 but stable at LR=0.001. The current schedule spends 25% of budget at LR=0.01 (epochs 34-52, ~18 epochs wasted oscillating) and 25% at LR=0.001 (epochs 52-106, ~54 epochs delivering +3.5pp). Shifting budget from the unstable middle phase to the productive final phase is the clear optimization.

## Candidate Ideas

### 1. Schedule optimization: shift drops to (0.35, 0.55) for AMP

**Summary**: Change the wall-clock-fractional schedule thresholds from (0.5, 0.75) to (0.35, 0.55). This shifts the first LR drop from epoch ~53 to epoch ~37 and the second drop from epoch ~80 to epoch ~58, giving the stable LR=0.001 phase ~48 epochs (45% of budget) instead of ~26 epochs (25%). The LR=0.01 phase shrinks from ~26 epochs to ~21 epochs.

**Reasoning**: EXP-005 showed the LR=0.01 phase is unstable with FP16 (oscillating 68-82% for 18 epochs) but the LR=0.001 phase is productive (+3.5pp over ~54 epochs). Moving budget from the unstable to the productive phase should improve accuracy. The (0.35, 0.55) thresholds preserve the three-plateau structure while shifting ~20% of the budget from the middle phase to the final phase.

**Sources**: reports/exp-report-005.md § Results (FP16 instability analysis), goal-learnings § Patterns (AMP unstable at LR=0.01).

**Estimated Effort**: Very low — two constants in the lambda function.

**Risk Assessment**: Low. The three-plateau LR structure is preserved. The risk is that the high-LR phase (0.35 of budget instead of 0.5) doesn't give enough exploration, but with 106 epochs that's still ~37 epochs at LR=0.1 — comparable to EXP-003's total of 69 epochs.

### 2. Three-drop schedule: (0.3, 0.5, 0.7) with LR cascade 1.0/0.3/0.1/0.01

**Summary**: Add a fourth LR plateau by introducing an intermediate drop at 0.3 of budget (LR 0.1→0.03) before the main drops at 0.5 (→0.01) and 0.7 (→0.001). This creates a smoother decay that may help FP16 stability in the middle phases.

**Reasoning**: The abrupt 10x drop from 0.1 to 0.01 at the 50% mark may be partly responsible for the FP16 instability — a smaller initial drop (3.3x) could provide a smoother transition. The cosine schedule's smooth decay was the motivation for EXP-000 (which failed for different reasons — T_max miscalibration).

**Sources**: He-2015 schedule analysis, EXP-000 post-mortem, EXP-005 instability analysis.

**Estimated Effort**: Low — modify the lambda function to have 4 return values.

**Risk Assessment**: Medium. Adds complexity to the schedule. The intermediate LR=0.03 plateau is untested and may not provide enough exploration benefit. The total time at the stable LR=0.001 is 30% of budget (vs 25% currently) — a smaller improvement than Candidate 1's 45%.

### 3. Batch size 256 + LR 0.2 with AMP

**Summary**: Double BATCH_SIZE to 256 and LR to 0.2 (linear scaling). With AMP already in place, the larger batch should further increase throughput by reducing Python overhead per image. Batches per epoch halves from 390 to 195.

**Reasoning**: With AMP's FP16, the H20's memory and compute are underutilized at batch 128 (only 266 MB peak VRAM). Doubling the batch size could increase GPU utilization and reduce per-step Python overhead, potentially adding ~20% more epochs. Combined with the LR=0.001 phase being the productive zone, more epochs in that phase compound.

**Sources**: Linear scaling rule (Goyal et al. 2017), EXP-005 VRAM data (266 MB used).

**Estimated Effort**: Very low — two constants.

**Risk Assessment**: Medium. Larger batches can reduce generalization quality. The LR scaling may be imprecise at this operating point. The FP16 instability at LR=0.01 may worsen with LR=0.02 (the scaled equivalent).

## Idea Evaluation

**Evidence**: Candidate 1 directly targets the identified FP16 instability with a minimal change. Candidate 2 is a more complex schedule change with uncertain benefit. Candidate 3 changes batch dynamics which could interact unpredictably with the FP16 instability.

**Impact**: Candidate 1 shifts ~20% of budget from the unstable to the productive phase — if the LR=0.001 phase continues its 0.065pp/epoch rate, ~22 extra epochs could add ~+1.4pp. Candidate 3's throughput gain is uncertain. Candidate 2's intermediate LR benefit is speculative.

**Risk**: Candidate 1 is the simplest and most directly targeted at the known failure mode.

## Chosen Idea

**Selected**: Candidate 1 — **Schedule optimization: shift drops to (0.35, 0.55) for AMP**

**Why this idea**: Directly targets the identified FP16 instability at LR=0.01 by minimizing time in that regime. Two constants changed, mechanism is clear, and the productive LR=0.001 phase gets 45% of the budget instead of 25%.

**Hypothesis**: Shifting the wall-clock-fractional schedule from (0.5, 0.75) to (0.35, 0.55) will raise best_test_acc from 94.44% to **94.8-95.5%** by giving the stable LR=0.001 phase nearly double the training time. The improvement bar is 94.54%.

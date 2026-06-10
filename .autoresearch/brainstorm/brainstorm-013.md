# Brainstorm EXP-013
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider CIFAR residual networks are a credible capacity direction, and prior local evidence now supports width over depth under this benchmark's fixed time budget.

- **EXP-011 ResNet-20 Width 1.25x report** (`reports/exp-report-011.md`)
  Stage widths 20/40/80 reached 92.12% while preserving a healthy 43,713-step budget, validating modest width scaling when the LR schedule is calibrated.

- **EXP-012 Earlier First LR Drop report** (`reports/exp-report-012.md`)
  Moving the first LR drop from 24k to 22k on the widened model only reached 92.16%, below the 92.22% threshold, so the current 24k drop remains the better anchor.

No new external search was needed; the most relevant evidence is now the local experiment trajectory plus the saved WRN distillation.

## Experimental History Review

- Current baseline is EXP-011 at `best_test_acc=92.12%`; the tightened rule requires EXP-013 to reach at least `92.22%`.
- The proven recipe is FP32 `torch.compile` plus channels-last, stage widths 20/40/80, classical momentum, and LR milestones `[24000, 64000]`.
- EXP-012 showed that a 22k first drop is too early for 20/40/80: it completed more steps and more LR 0.01 refinement, but peaked at only 92.16% and finished at 91.66%.
- Width is the only capacity direction with a confirmed improvement so far. Depth scaling failed in EXP-006 because the run missed the first LR drop, so any capacity increase must preserve enough steps to enter LR 0.01.
- Cutout, Nesterov, TF32, early second drops, and per-step EMA have all underperformed or missed the +0.10 threshold. Late low-overhead averaging remains possible, but it is a more complex mechanism than a direct width step.
- The main untried gap is a second, modest width increase beyond 20/40/80 while keeping the validated 24k first drop rather than moving it earlier.

## Candidate Ideas

### 1. ResNet-20 Width 1.5x with Proven 24k First Drop
**Summary**: Increase stage widths from 20/40/80 to 24/48/96 while keeping depth, optimizer, augmentation, FP32 precision, compile, channels-last layout, batch size, seed, and LR milestones `[24000, 64000]` unchanged. This tests whether another modest width step raises the accuracy ceiling enough to clear 92.22% without adding scheduler uncertainty.

**Reasoning**: EXP-011 established that width scaling is viable in this fixed-budget harness, and EXP-012 indicates that the 24k first drop is better calibrated than an earlier 22k drop for the widened model. A 24/48/96 model should add representational capacity while retaining the recipe's known good LR timing. The risk is step loss, but EXP-011's width increase did not materially reduce the step budget relative to EXP-002, so a second moderate increase is plausible.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-011.md`; `reports/exp-report-012.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The larger model may reduce throughput enough that the post-drop LR 0.01 window is too short, or the 24k drop may become late for the new width. Worst case is a valid no-improvement that calibrates the next width/schedule boundary.

### 2. Later First LR Drop on Current 20/40/80 Model
**Summary**: Keep the EXP-011 architecture and move the first LR milestone later, for example from 24000 to 26000, leaving the second milestone unreachable at 64000. This tests whether the 20/40/80 model benefits from more high-LR training before entering LR 0.01.

**Reasoning**: EXP-012 proved that moving earlier to 22k did not help, so the opposite direction may be the local schedule adjustment that matters. A later drop preserves the current architecture and can isolate schedule timing, but the likely upside is small because EXP-011 already reached its best at the final epoch and may have needed the available LR 0.01 refinement time.

**Sources**: `reports/exp-report-011.md`; `reports/exp-report-012.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: If the drop is too late, the run may not have enough LR 0.01 time to polish weights and could regress below EXP-011. This is low blast radius but likely lower upside than adding capacity.

### 3. Low-Frequency Late Averaging on Widened ResNet-20
**Summary**: Keep the EXP-011 architecture and schedule, but add a low-overhead averaging mechanism only after the first LR drop, updating at epoch boundaries or sparse step intervals rather than every step. Evaluate the averaged weights using the same once-per-epoch cadence.

**Reasoning**: EXP-004 showed that averaging can improve accuracy slightly, but per-step EMA overhead consumed too much of the step budget. A late and sparse average may smooth final weights while preserving most throughput, especially now that the 20/40/80 model reaches a stronger low-LR plateau.

**Sources**: `reports/exp-report-004.md`; `references/pytorch-ema-averaging.md`; `reports/exp-report-011.md`.

**Estimated Effort**: medium

**Risk Assessment**: Even sparse averaging adds complexity and may mishandle batch norm buffers or add enough overhead to lose steps. It also has less direct evidence than width scaling under the current best recipe.

## Idea Evaluation

The 1.5x width candidate has the strongest evidence and highest expected impact. Width scaling is the only approach that has produced a real improvement after the FP32 throughput baseline, and the saved WRN note supports width as a sensible CIFAR residual-network scaling axis. Keeping the 24k first drop uses the schedule that actually succeeded and avoids repeating EXP-012's too-early drop.

The later-drop schedule test is cleaner and cheaper, but it only explores a local schedule neighborhood after the 22k miss. It might recover a small gain, but clearing the +0.10 threshold from a 92.12% baseline likely needs more than schedule noise.

Late averaging remains a reasonable future direction, but EXP-004 already exposed the overhead trap. It should be planned carefully after exhausting simpler capacity moves, not chosen as the immediate next run.

## Chosen Idea
**Selected**: ResNet-20 Width 1.5x with Proven 24k First Drop

**Why this idea**:
It best combines local evidence and upside. EXP-011 validated width scaling, EXP-012 rejected an earlier drop as the next exploitation move, and the 24k schedule is the known-good anchor for the widened model. A second moderate width step is more likely than another scheduler-only tweak to produce the required +0.10 point gain.

**Hypothesis**:
A 24/48/96 ResNet-20 with the validated `[24000, 64000]` milestones will retain enough fixed-budget steps to reach the LR 0.01 phase and improve `best_test_acc` to at least `92.22%`.

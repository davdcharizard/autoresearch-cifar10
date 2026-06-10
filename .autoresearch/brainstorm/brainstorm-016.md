# Brainstorm EXP-016
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-014 ResNet-20 Width 1.75x report** (`reports/exp-report-014.md`)
  The 28/56/112 model with a 22k first drop reached the current best 93.09%, proving this width point can work when schedule-calibrated.

- **EXP-015 schedule report** (`reports/exp-report-015.md`)
  Moving the same model's first drop later to 23k produced more total steps but lowered the peak to 92.88%, showing 23k is too late for this width.

- **Goal learnings** (`goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  Width scaling remains the strongest positive pattern, and the newest failed approach says 23k should not be retried on 28/56/112 without another coupled change.

No new external search was needed. The next choice is a local schedule-bracketing experiment driven by direct project evidence.

## Experimental History Review

- Current baseline is EXP-014 at `best_test_acc=93.09%`; the tightened success rule requires EXP-016 to reach at least `93.19%`.
- EXP-014 succeeded with 28/56/112 and a 22k first drop, completing 34,259 steps and peaking at 93.09%.
- EXP-015 kept 28/56/112 but moved the first drop later to 23k. It completed 38,274 steps and 99 epochs but peaked at only 92.88%, so later high-LR exploration did not help.
- The local schedule bracket is now asymmetric: 23k is too late for 28/56/112, while 22k is best known. Testing 21k completes the immediate bracket and may reveal whether slightly more low-LR refinement improves the current model.
- EXP-012 warns that 22k was too early for the smaller 20/40/80 model, but EXP-014 shows that larger/slower widths can prefer earlier drops. A 21k test is therefore a width-specific calibration, not a generic retry of an earlier-drop failure.

## Candidate Ideas

### 1. Move 28/56/112 First LR Drop from 22k to 21k
**Summary**: Keep the current best 28/56/112 model and move only the first LR milestone from 22000 to 21000. This tests whether slightly more LR 0.01 refinement time can improve the current best model after EXP-015 showed 23k is too late.

**Reasoning**: EXP-014 improved strongly with a 22k drop, while EXP-015 proved delaying to 23k hurts. The remaining local schedule question is whether 22k is optimal or still slightly late. A 21k drop should be reachable with plenty of time left, and it directly tests the most plausible one-variable route to a +0.10 point gain.

**Sources**: `reports/exp-report-014.md`; `reports/exp-report-015.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The run may under-explore at LR 0.1 and repeat the earlier-drop failure mechanism seen on smaller widened models. Failure mode is clean no-improvement and completes the local schedule bracket.

### 2. ResNet-20 Width 30/60/120 with 20k First Drop
**Summary**: Increase stage widths from 28/56/112 to 30/60/120 and move the first LR drop to 20000 to compensate for the expected lower step budget. This tests whether width scaling still has capacity headroom beyond EXP-014.

**Reasoning**: Width scaling has produced the largest improvements so far. If schedule-only tuning around 28/56/112 cannot clear 93.19%, a modest capacity step may provide a larger accuracy ceiling. The first drop should move earlier because additional width will likely reduce the step budget.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-014.md`; `reports/exp-report-015.md`.

**Estimated Effort**: low

**Risk Assessment**: This changes both capacity and schedule, so attribution is less clean. The wider model may undertrain or need a different milestone than 20k.

### 3. Sparse Late Averaging on 28/56/112
**Summary**: Keep 28/56/112 and the 22k first drop, but average weights sparsely after the first LR drop, such as once per epoch, to smooth late low-LR fluctuations without per-step EMA overhead.

**Reasoning**: EXP-014 peaked at 93.09% but ended at 92.92%, and EXP-015 similarly plateaued below its best. Low-frequency averaging could improve the selected evaluation weights without the per-step overhead that hurt EXP-004.

**Sources**: `reports/exp-report-004.md`; `references/pytorch-ema-averaging.md`; `reports/exp-report-014.md`; `reports/exp-report-015.md`.

**Estimated Effort**: medium

**Risk Assessment**: Averaging introduces implementation complexity and possible BatchNorm-state pitfalls. Even sparse averaging can add overhead or produce stale statistics if not handled carefully.

## Idea Evaluation

The 21k first-drop test has the strongest immediate evidence because it completes the local schedule bracket around the best known 28/56/112 recipe. EXP-015 showed the 23k direction is worse, so the only clean schedule-only variant left is earlier than 22k. It is narrow, low effort, and directly tests whether the current model can clear the 93.19% threshold without adding confounds.

The 30/60/120 width step remains attractive because width scaling has been the highest-upside pattern. However, it should follow the local schedule bracket unless the 21k result is clearly worse, because a width step changes both capacity and throughput.

Sparse late averaging is plausible but less mature. It needs more careful implementation than a one-line schedule test, and prior EMA overhead makes it a second-tier candidate until the schedule and width axes are mapped further.

## Chosen Idea
**Selected**: Move 28/56/112 First LR Drop from 22k to 21k

**Why this idea**:
It is the cleanest remaining one-variable test around the current best model. EXP-015 ruled out moving later to 23k, so 21k is the direct way to test whether the 22k winner benefits from even more LR 0.01 refinement time.

**Hypothesis**:
A 28/56/112 ResNet-20 with first LR drop at step 21000 will retain enough high-LR exploration while adding low-LR refinement time, improving `best_test_acc` to at least `93.19%`.

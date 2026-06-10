# Brainstorm EXP-015
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  The local trajectory continues to validate compact width scaling as the strongest CIFAR-10 capacity path under the fixed time budget.

- **EXP-014 ResNet-20 Width 1.75x report** (`reports/exp-report-014.md`)
  The current 28/56/112 model reached 93.09% after a 22k first drop, peaking late at epoch 85 after the LR 0.01 phase had stabilized.

- **Goal learnings** (`goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  Width scaling is now a repeated positive pattern, but milestone placement is width-dependent: 22k was too early for 20/40/80 yet effective for 28/56/112.

No new external search was needed. The best guidance for the next experiment is the local sequence of width/schedule results, which is more specific than generic CIFAR recipe literature.

## Experimental History Review

- Current baseline is EXP-014 at `best_test_acc=93.09%`; the tightened success rule requires EXP-015 to reach at least `93.19%`.
- EXP-011 and EXP-013 showed that widening ResNet-20 from the FP32 throughput baseline to 20/40/80 and then 24/48/96 gave repeated improvements when paired with reachable first LR drops.
- EXP-014 widened again to 28/56/112, moved the first drop to 22k, completed 34,259 steps, and reached 93.09%, a large +0.60 point gain over EXP-013.
- EXP-014 peaked at epoch 85, long after the step-22000 first drop at epoch 57. This means the model used the LR 0.01 phase effectively, but it does not prove 22k is exactly optimal.
- EXP-012 warns that making the first drop too early can lose useful high-LR exploration on smaller widened models; schedule tuning should stay local and tied to a specific width.
- The next baseline requires a smaller absolute lift than EXP-014 achieved, so a clean one-variable schedule test is a reasonable next probe before adding more capacity and confounding step-budget effects.

## Candidate Ideas

### 1. Move 28/56/112 First LR Drop from 22k to 23k
**Summary**: Keep the successful EXP-014 28/56/112 model and move only the first LR milestone from 22000 to 23000, leaving the second milestone unreachable at 64000. This tests whether slightly more high-LR exploration improves the current best model while still preserving roughly 11k low-LR refinement steps.

**Reasoning**: EXP-014 crossed the threshold quickly after the drop and peaked late, but the jump from 22k to 23k costs only about 1k LR 0.01 steps while adding high-LR exploration. EXP-012 showed that 22k can be too early for smaller widths, so the optimum may be later than 22k even at 28/56/112. This is a narrow, interpretable experiment with enough expected upside to clear the new +0.10 point threshold if schedule placement was slightly early.

**Sources**: `reports/exp-report-014.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: The effect size may be under the +0.10 point noise margin, or the later drop may leave too little LR 0.01 refinement and reduce peak accuracy. Failure mode is a clean no-improvement that maps the local schedule curve.

### 2. Move 28/56/112 First LR Drop from 22k to 21k
**Summary**: Keep the current 28/56/112 model and move the first LR milestone earlier to 21000. This tests whether the EXP-014 model would benefit from even more LR 0.01 refinement time under the same fixed budget.

**Reasoning**: EXP-014’s accuracy improved sharply after the LR drop and stayed high through the end of the run, so more low-LR time might convert into a higher late peak. However, prior schedule-only failures show that dropping too early can undercut high-LR exploration, and 22k already produced a strong result.

**Sources**: `reports/exp-report-014.md`; `reports/exp-report-012.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: This may repeat the earlier-drop failure mechanism from EXP-012, just on a larger model. It is useful for schedule mapping but has a slightly less compelling mechanism than the 23k test.

### 3. ResNet-20 Width 30/60/120 with 20k First Drop
**Summary**: Increase stage widths from 28/56/112 to 30/60/120 and move the first LR drop to 20000 to compensate for the expected slower step budget. This tests whether width scaling still has headroom past EXP-014.

**Reasoning**: Width scaling has generated the largest gains so far, and EXP-014 suggests that larger models can work if milestones are adjusted earlier. A 30/60/120 model is a cautious capacity step rather than a jump to a much larger network, and a 20k first drop should be reachable even if throughput falls.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-013.md`; `reports/exp-report-014.md`.

**Estimated Effort**: low

**Risk Assessment**: Added capacity may reduce steps enough that the model undertrains or has too little high-LR exploration. This has higher upside than schedule-only tests but also confounds width and schedule, making the result less diagnostic.

## Idea Evaluation

The 23k first-drop test is the best next experiment because it isolates the schedule question exposed by EXP-014. The current model is already strong enough to sit near the new threshold, and a one-step local retune can reveal whether 22k was slightly early without changing capacity, parameter count, or throughput. Its failure mode is informative and low risk.

The 21k first-drop test is also diagnostic, but the existing evidence is weaker. EXP-014 already had substantial LR 0.01 time and peaked late, while EXP-012 directly warns against overly early drops on widened models. If 23k fails by reducing low-LR time too much, 21k becomes a reasonable follow-up.

The 30/60/120 width step has the highest upside if schedule-only tuning stalls, but it changes two coupled factors at once and may drop the step budget sharply. Because EXP-014 just made a large gain, mapping the local schedule around the new best model is more prudent before spending an experiment on another capacity jump.

## Chosen Idea
**Selected**: Move 28/56/112 First LR Drop from 22k to 23k

**Why this idea**:
It is the cleanest one-variable test around the new best model and directly probes whether the 22k drop in EXP-014 was slightly too early. The new baseline requires 93.19%, so even a modest schedule improvement could count while preserving all validated width and throughput settings.

**Hypothesis**:
A 28/56/112 ResNet-20 with first LR drop at step 23000 will preserve enough LR 0.01 refinement time while adding useful high-LR exploration, improving `best_test_acc` to at least `93.19%`.

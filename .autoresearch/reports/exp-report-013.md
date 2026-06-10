# Report EXP-013: ResNet-20 Width 1.5x with Proven 24k First Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Log**: logs/exp-log-013.md

## Goal
EXP-013 tried to improve CIFAR-10 `best_test_acc` under the fixed evaluation harness. The baseline before this run was EXP-011 at `92.12%`, and the tightened goal rule required at least `92.22%` for a valid improvement.

## Idea & Hypothesis
The chosen idea was a second moderate width increase: use a 24/48/96 ResNet-20 while preserving the successful `[24000, 64000]` milestone schedule. The hypothesis was that added capacity would raise the accuracy ceiling while retaining enough fixed-budget steps to reach the 24k LR drop and clear the +0.10 point threshold.

## Approach
The implementation changed one constant in `train.py`: `STAGE_WIDTHS = (20, 40, 80)` became `(24, 48, 96)`. The run preserved depth, optimizer, batch size, data augmentation, seed, FP32 precision, `torch.compile`, channels-last layout, and once-per-epoch evaluation. The first LR drop stayed at step 24000 to isolate width from scheduler retuning.

## Execution
One local single-GPU run was launched on physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`. The run completed cleanly, used a 605,026-parameter model, reached the first LR drop at step 24000 during epoch 62, and finished 41,825 steps over 108 epochs. No retries or implementation adjustments were needed.

## Results
- **Primary metric**: 92.49% (baseline: 92.12%, delta: +0.37 points, +0.40%)
- **Observations**: Accuracy jumped from 88.81% pre-drop at epoch 60 to 91.39% at epoch 62, crossed the 92.22% threshold by epoch 71, and peaked at 92.49% later in the LR 0.01 phase.
- **Analysis**: The hypothesis was supported. The 24/48/96 model lost some step budget versus EXP-011 but still retained enough LR 0.01 refinement time, and the extra capacity produced the largest improvement so far.
- **Key Learning**: A 24/48/96 ResNet-20 with the 24k first drop is a better capacity point than 20/40/80, reaching 92.49% within the fixed budget.

## Verification
- **Conditions**: All checked conditions passed.
- **Review Notes**: Results are trustworthy: the run used one visible NVIDIA H20, changed only `train.py`, touched only `STAGE_WIDTHS`, reached the intended first LR drop, and reported complete final metrics.
- **Verdict**: improvement
- **Verdict Basis**: `best_test_acc=92.49%` exceeds the prior 92.12% baseline by +0.37 percentage points, clearing the required +0.10-point margin.

## Unexplored Avenues
- A further width increase, such as 28/56/112 or 32/64/128, could test whether capacity scaling still has headroom, but it may need an earlier first drop if throughput falls.
- A schedule retune around the new 24/48/96 model, especially first-drop timing near 22k-24k, could exploit the observation that the model peaked after substantial LR 0.01 refinement.
- Low-overhead late averaging may now be more attractive on the stronger 24/48/96 baseline if implemented without per-step overhead.

## Next Steps
Try a schedule-calibrated wider model with medium confidence: increase width one more step and choose the first LR milestone based on expected reachable steps. Alternatively, tune the first LR drop on 24/48/96 with medium confidence, since this model clearly benefits from the LR 0.01 phase and has a stronger ceiling.

## Exit Action Results
- None: the goal has no active exit actions.

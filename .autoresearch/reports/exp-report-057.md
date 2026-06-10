# Report EXP-057: Post-Drop Label Smoothing Anneal
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Log**: logs/exp-log-057.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The current experiment-index baseline is 93.97% from commit `755be2c`; with the explicit +0.10 percentage-point noise guard, EXP-057 needed at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-057 tested whether the active anchor should keep `label_smoothing=0.05` during high-LR training but remove smoothing after the step-21000 LR drop. The hypothesis was that high-LR smoothing helps representation learning, while post-drop hard labels might improve late low-LR refinement and overcome the observed post-drop plateaus.

## Approach
`train.py` was changed to add explicit smoothing schedule constants: pre-drop `0.05`, post-drop `0.0`, and switch step `LR_MILESTONES[0]`. The loss selects the active smoothing value before each batch, startup prints the schedule, and progress logs include `ls: ...` so the switch can be verified. All architecture, optimizer, schedule, augmentation, compile, validation, and fixed-budget settings otherwise stayed at the anchor.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Preflight checks passed and no retries were needed. Startup confirmed the expected schedule, `Batches per epoch: 390`, and `num_params=822,790`. The first LR drop occurred at step 21000, and the smoothing switch was visible at step 21050 with `lr: 0.0100` and `ls: 0.000`.

## Results
- **Primary metric**: 93.42% (baseline: 93.97%, delta: -0.55 percentage points, -0.59%)
- **Observations**: Pre-drop behavior was valid and the first LR drop was reached. After smoothing was removed, training loss collapsed sharply from roughly 0.50 to below 0.10, but validation accuracy only reached 93.42% at epoch 62 and then plateaued or drifted downward, ending at 92.96%.
- **Analysis**: The hypothesis is rejected. Removing smoothing during low-LR refinement did not sharpen the classifier beneficially; it made the model fit hard labels more aggressively without improving held-out accuracy. This reinforces the anchor's need for continued mild smoothing rather than only pre-drop smoothing. It also explains why static lower smoothing was a near miss rather than an improvement: the recipe benefits from the soft target regularization through late refinement.
- **Key Learning**: Post-drop hard-label sharpening overfits the current anchor; keep `label_smoothing=0.05` active through the full fixed-budget run.

## Verification
- **Conditions**: all process conditions passed; metric threshold failed.
- **Review Notes**: Results are trustworthy. The run completed, produced numeric summary metrics, respected `train.py`-only scope, preserved batch geometry and parameter count, and verified both the LR drop and smoothing switch.
- **Verdict**: no-improvement.
- **Verdict Basis**: `best_test_acc=93.42%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Use a gentler post-drop reduction, such as `0.05 -> 0.03`, instead of switching to hard labels. The current result says full removal is too sharp, but it does not fully rule out very mild late smoothing annealing.
- Delay smoothing reduction until the post-drop plateau is established, such as several epochs after step 21000. This would be a narrower variant, but the evidence now makes smoothing schedules lower priority than distinct mechanisms.

## Next Steps
- Medium confidence: try a minimal classifier-head dropout probe only if prioritizing a cheap final regularizer; it is distinct from label smoothing but recent isolated regularizers are weak.
- Medium confidence: test mixup without label smoothing if revisiting coupled regularization balance; EXP-055's 93.85% makes it a better-supported regularization variant than hard-label annealing.
- Higher confidence: brainstorm non-regularization mechanisms that preserve label smoothing 0.05, global `2e-4` decay, LR 0.1, batch 128, and the 21k first drop.

## Exit Action Results

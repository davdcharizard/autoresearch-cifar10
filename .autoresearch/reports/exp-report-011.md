# Report EXP-011: ResNet-20 Width 1.25x
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Log**: logs/exp-log-011.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed 300s training budget and fixed evaluation harness. The pre-experiment baseline was 91.95%, and the tightened goal requires at least +0.10 percentage points over baseline, so EXP-011 needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
The chosen idea was a modest capacity increase: keep ResNet-20 depth, widen stages from 16/32/64 to 20/40/80, and move the first LR drop earlier to step 24000. The hypothesis was that width would raise the accuracy ceiling more efficiently than EXP-006's depth increase, while the earlier drop would prevent the wider model from spending the whole run at LR 0.1.

## Approach
`train.py` now defines `STAGE_WIDTHS = (20, 40, 80)` and wires those widths through `ResNet.__init__` for `conv1`, `bn1`, all three residual stages, and the classifier. It also defines `LR_MILESTONES = [24000, 64000]` and uses that constant in `MultiStepLR`. Optimizer, batch size, augmentation, seed, FP32 precision, channels-last, `torch.compile`, depth, and once-per-epoch validation were preserved.

## Execution
One local run was launched on a single NVIDIA H20. Physical GPU 0 was occupied by an unrelated run, so EXP-011 used physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`, which preserved the single-GPU constraint. Startup and early monitoring were clean; the known non-fatal Inductor TF32 warning appeared again. The run reached the planned step-24000 LR drop during epoch 62 and completed without errors.

## Results
- **Primary metric**: 92.12% (baseline: 91.95%, delta: +0.17 points, +0.18%)
- **Observations**: The wider model had 420,670 parameters and completed 43,713 steps, compared with 269,722 parameters and 43,398 steps for EXP-002. Accuracy was only 88.04% before the LR drop, then jumped to 91.05% at epoch 62 and reached 92.12% at the final epoch.
- **Analysis**: The hypothesis was supported. A 1.25x width increase preserved enough throughput to keep the step budget near the previous baseline, and the 24k first LR drop created 19,713 LR 0.01 refinement steps. This is different from EXP-006, where deeper ResNet-32 undertrained and missed its first LR drop.
- **Key Learning**: Modest width scaling plus an earlier first LR drop can raise the fixed-budget CIFAR-10 ceiling when the schedule is calibrated to the observed step budget.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy. The run completed, reported numeric final metrics, used one visible H20 GPU, modified only `train.py`, preserved once-per-epoch evaluation, and cleared the tightened 92.05% threshold.
- **Verdict**: improvement
- **Verdict Basis**: all verification conditions passed and `best_test_acc=92.12%` exceeded the 91.95% baseline by +0.17 points, above the required +0.10 point margin.

## Unexplored Avenues
- Try a slightly wider schedule such as 24/48/96 only if the first LR drop is recalibrated from measured steps; EXP-011 suggests width has room, but throughput margin remains finite.
- Tune the first LR milestone around the wider model, for example 22000 or 26000, to see whether the final plateau improves without reducing LR 0.01 refinement too much.
- Combine the successful width schedule with a low-overhead late averaging variant, because EXP-004 was close but per-step EMA lost too many steps.

## Next Steps
1. **Medium confidence**: Tune the widened model's first LR drop around 24k. EXP-011 succeeded, but the final plateau was narrow; a small schedule sweep may add another margin.
2. **Medium confidence**: Test a slightly wider ResNet-20 only after estimating whether it will still reach at least 40k steps and a timely LR drop.
3. **Low confidence**: Revisit low-overhead late averaging on top of the widened model, with update frequency constrained so step count remains close to EXP-011.

## Exit Action Results
- No exit actions defined in the goal file.

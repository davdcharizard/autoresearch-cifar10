# Report EXP-012: Earlier First LR Drop on Widened ResNet-20
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Log**: logs/exp-log-012.md

## Goal
EXP-012 tried to improve CIFAR-10 `best_test_acc` under the fixed training harness. The current baseline was EXP-011 at `92.12%`, and the tightened goal rule requires at least a +0.10 percentage-point gain, so this experiment needed `best_test_acc >= 92.22%` to count as an improvement.

## Idea & Hypothesis
The selected idea was to keep the successful widened ResNet-20 from EXP-011 and move only the first LR milestone from step 24000 to step 22000. EXP-011 reached its best accuracy at the final epoch, so the hypothesis was that about 2000 extra LR 0.01 refinement steps would let the same 20/40/80 model clear the new threshold.

## Approach
The implementation changed one scheduler constant in `train.py`: `LR_MILESTONES = [24000, 64000]` became `[22000, 64000]`. Stage widths, depth, optimizer, augmentation, seed, FP32 precision, `torch.compile`, channels-last layout, batch size, and once-per-epoch validation were preserved. The second LR milestone remained unreachable to avoid the harmful LR 0.001 phase seen in EXP-003.

## Execution
One local single-GPU run was launched on physical GPU 1 with `CUDA_VISIBLE_DEVICES=1` because physical GPU 0 was occupied by an unrelated run. The run completed cleanly in 405.5 seconds total, reached the intended first LR drop at step 22000 during epoch 57, and produced parseable final metrics. No retries or implementation adjustments were needed.

## Results
- **Primary metric**: 92.16% (baseline: 92.12%, delta: +0.04 points, +0.04%)
- **Observations**: The earlier drop increased the LR 0.01 window and completed 45,478 steps over 117 epochs, but the best accuracy topped out at 92.16% and final accuracy declined to 91.66%.
- **Analysis**: The hypothesis was not supported. The 22k first drop did create more low-LR refinement time, but it appears to shorten high-LR training too much for the widened model; EXP-011's 24k drop remains better calibrated.
- **Key Learning**: A 22k first drop is too early for the widened ResNet-20 under the fixed 300s budget; the 24k EXP-011 schedule remains the better local optimum.

## Verification
- **Conditions**: Primary metric condition failed after baseline, single-GPU execution, and completion checks passed.
- **Review Notes**: Results are trustworthy: the run used one visible NVIDIA H20, changed only the planned scheduler constant, reached the intended first LR drop, and reported complete final metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=92.16%` is a valid result but below the tightened `92.22%` improvement threshold.

## Unexplored Avenues
- A slightly later first drop, such as 26000, could test whether EXP-011 benefited from even more high-LR training before refinement.
- A wider 24/48/96 model with a calibrated milestone may have higher upside than further tuning the already near-optimal 20/40/80 schedule.
- Low-overhead late averaging remains possible if it avoids the per-step EMA overhead that hurt EXP-004.

## Next Steps
Try a higher-upside capacity experiment next with medium confidence: increase width beyond 20/40/80 while calibrating the first LR drop to the expected step budget. A later first-drop schedule on 20/40/80 is lower effort but likely lower upside after the 22k miss.

## Exit Action Results
- None: the goal has no active exit actions.

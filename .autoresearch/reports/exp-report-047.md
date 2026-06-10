# Report EXP-047: Mild ColorJitter After Crop/Flip
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Log**: logs/exp-log-047.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness and wall-clock budget. The active baseline is 93.97% from EXP-038 at commit `755be2c`; because the goal requires at least +0.10 percentage points, EXP-047 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
EXP-047 tested a narrow photometric augmentation: mild `ColorJitter` after the existing reflection crop and horizontal flip. The hypothesis was that conservative brightness/contrast/saturation/hue perturbation could improve color and illumination invariance without the overhead or semantic distortion seen in prior RandAugment and cutout-style failures.

## Approach
Only `train.py` changed. The training transform gained `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)` after `RandomHorizontalFlip()` and before `ToTensor()`. All non-augmentation anchors were preserved: `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Both available GPUs were already under external 100% utilization at launch, but memory headroom was sufficient and the run completed without crashing. The run did not satisfy the planned schedule-behavior expectation: it ended at 20,321 steps and never reached the first LR drop at step 21,000.

## Results
- **Primary metric**: 88.89% (baseline: 93.97%, delta: -5.08 pp, -5.41%)
- **Observations**: The process completed cleanly and produced all final metrics, but every logged LR line stayed at `lr: 0.1000`. The final step count was 20,321, below the 21k first-drop anchor that prior successful experiments depended on.
- **Analysis**: The hypothesis is not cleanly answered. The measured result is far below threshold, so it is not an improvement, but the missed first LR drop makes this a weak attribution test of ColorJitter itself. The dominant lesson is operational: under heavy GPU contention, even a small augmentation can lose enough throughput to skip the critical step-schedule transition.
- **Key Learning**: Mild ColorJitter did not improve in this run, but missed-step-schedule contention prevents treating it as a clean photometric-augmentation failure.

## Verification
- **Conditions**: improvement condition failed; LR milestone behavior failed/caveat.
- **Review Notes**: Results are trustworthy as a completed benchmark run: it modified only `train.py`, used one selected GPU, preserved the fixed harness, stayed below the 10-minute cap, and produced numeric metrics. They are weak for scientific attribution because the run missed the planned step-21000 LR drop.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid numeric result, but `best_test_acc=88.89%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- **Retry targeted ColorJitter only when a clean GPU can reach step 21000**: This would separate photometric regularization from the throughput/schedule failure observed here.
- **Lower-overhead photometric normalization variants**: Static per-channel augmentation or lighter brightness/contrast-only jitter may preserve more step budget if a clean retry is still not feasible.
- **Non-augmentation lever next**: Because the current run is confounded, move to a different low-overhead lever rather than immediately promoting ColorJitter to failed-approach status.

## Next Steps
- **High confidence**: Avoid drawing a strong negative conclusion about ColorJitter from EXP-047; record the missed-drop contention as a protocol finding.
- **Medium confidence**: In the next loop, choose an experiment with very low runtime overhead or wait for lower GPU contention before another augmentation test.
- **Medium confidence**: Revisit the hybrid step-hold cosine idea only if it preserves the 21k high-LR window or adapts safely when the window is missed.

## Exit Action Results

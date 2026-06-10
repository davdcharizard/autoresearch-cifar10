# Report EXP-043: Initial LR 0.08 on 2e-4 Anchor
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-043.md
- **Plan**: plans/plan-043.md
- **Log**: logs/exp-log-043.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The pre-experiment baseline was 93.97% from EXP-038, and the goal's +0.10 percentage-point rule required EXP-043 to reach at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-043 tested the lower side of the initial-LR bracket around the current `WEIGHT_DECAY = 2e-4` label-smoothed reflection anchor. The hypothesis was that `LR = 0.08` might reduce high-LR noise and improve the late post-drop plateau enough to reach `best_test_acc >= 94.07%`.

## Approach
Only `train.py` changed during the run: `LR = 0.1` became `LR = 0.08`. Architecture, reflected `RandomCrop`, `BATCH_SIZE = 128`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 2e-4`, `LR_MILESTONES = [21000, 64000]`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed time budget, and once-per-epoch validation were preserved. The first LR drop correctly produced `lr: 0.0080`.

## Execution
One local single-GPU run was launched on GPU 0 in a foreground session. Startup confirmed CUDA execution, 822,790 parameters, the fixed 300s training budget, 390 batches per epoch, and initial `lr: 0.0800`. The first LR drop occurred at step 21000, no second LR drop occurred, and the run completed cleanly under the 10-minute wall-clock cap.

## Results
- **Primary metric**: 93.49% (baseline: 93.97%, delta: -0.48 points, -0.51%)
- **Observations**: Pre-drop accuracy reached 89.83% by epoch 45, then post-drop accuracy peaked at 93.49% on epoch 72. The final accuracy was 93.19% after 102 epochs and 39,520 steps.
- **Analysis**: The hypothesis was not supported. Lowering LR reduced useful pre-drop progress and left the post-drop plateau below the current anchor. Together with EXP-040's `LR = 0.12` failure, this brackets `LR = 0.1` as the local initial-LR setting for the `2e-4` anchor.
- **Key Learning**: Initial LR is locally bracketed for the current anchor; both 0.08 and 0.12 regress, so keep `LR = 0.1`.

## Verification
- **Conditions**: metric improvement condition failed; all hard constraints passed
- **Review Notes**: Results are trustworthy: the run completed cleanly, modified only `train.py`, preserved the fixed harness and single-GPU setup, reported a numeric metric, preserved the expected schedule behavior, and stayed under the 10-minute wall-clock cap.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run, but 93.49% is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Test a carefully bounded late post-drop averaging or EMA variant that avoids per-step overhead and long equal-averaging collapse.
- Test a clean no-restart cosine schedule only if willing to replace the calibrated 21k step-drop trajectory with a full-trajectory schedule probe.
- Revisit mixup only after stabilizing run control, since EXP-042 failed before final metrics and should not be immediately retried.

## Next Steps
Try bounded late EMA evaluation with low-to-medium confidence if accepting higher implementation risk; it directly targets late plateau drift without changing the optimizer path.

Try a clean cosine schedule with low confidence if scalar optimizer probes are exhausted; literature support is real, but the current step schedule is locally well calibrated.

Avoid further isolated initial-LR retuning with high confidence; `0.08`, `0.1`, and `0.12` now map the useful local bracket.

## Exit Action Results

# Report EXP-040: Raise Initial LR to 0.12 on 2e-4 Anchor
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-040.md
- **Plan**: plans/plan-040.md
- **Log**: logs/exp-log-040.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The pre-experiment baseline was 93.97% from EXP-038, and the goal's +0.10 percentage-point rule required EXP-040 to reach at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-040 tested whether the validated `WEIGHT_DECAY = 2e-4` anchor could benefit from a higher initial learning rate. The hypothesis was that `LR = 0.12` would improve high-LR exploration before the preserved 21k first drop.

## Approach
Only `train.py` changed: `LR = 0.1` became `LR = 0.12`. Architecture, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 2e-4`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed time budget, and once-per-epoch validation were preserved. The first LR drop correctly produced `lr: 0.0120`.

## Execution
One local single-GPU run was launched on GPU 1 because GPU 0 was occupied by an unrelated run. Startup confirmed CUDA execution, 822,790 parameters, the fixed 300s training budget, 390 batches per epoch, and initial `lr: 0.1200`. The run completed cleanly under the 10-minute wall-clock limit.

## Results
- **Primary metric**: 93.70% (baseline: 93.97%, delta: -0.27 points, -0.29%)
- **Observations**: Pre-drop accuracy was noisy, the post-drop plateau peaked at 93.70%, and final accuracy was 93.13% after 40,378 steps.
- **Analysis**: The hypothesis was not supported. A higher initial LR did not improve the `2e-4` anchor and appears to reduce stability relative to the EXP-038 recipe.
- **Key Learning**: The `2e-4` anchor should keep `LR = 0.1`; raising initial LR to 0.12 weakens the late plateau.

## Verification
- **Conditions**: metric improvement condition failed; hard constraints passed
- **Review Notes**: Results are trustworthy: the run completed cleanly, modified only `train.py`, preserved the fixed harness and single-GPU setup, and reported a numeric metric.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run, but 93.70% is below both the 93.97% baseline and the 94.07% threshold.

## Unexplored Avenues
- Test `LR = 0.08` on the `2e-4` anchor if lowering high-LR noise is considered worth a scalar probe.
- Test `WEIGHT_DECAY = 1.5e-4` to map whether the local optimum lies slightly below `2e-4`.
- Revisit low-frequency post-drop averaging only after simple scalar levers are exhausted.

## Next Steps
Try `WEIGHT_DECAY = 1.5e-4` with low-to-medium confidence to complete the local weight-decay bracket around the successful `2e-4` value.

Try `LR = 0.08` with low-to-medium confidence if the priority is optimizer dynamics rather than regularization mapping.

Avoid further isolated LR increases with high confidence; `0.12` made the anchor worse.

## Exit Action Results

# Report EXP-029: Reflection Padding for RandomCrop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Log**: logs/exp-log-029.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed evaluation harness. The previous baseline was 93.23% from EXP-016, and the goal requires at least a +0.10 percentage-point absolute gain to count as an improvement, so EXP-029 needed `best_test_acc >= 93.33%`.

## Idea & Hypothesis
The selected idea was to change the training `RandomCrop` padding mode from constant zero padding to reflection padding. The hypothesis was that reflected crop margins would reduce artificial zero-border artifacts while preserving the tuned 28/56/112 ResNet-20 anchor, throughput path, optimizer, schedule, and validation cadence.

## Approach
Implemented a one-line `train.py` change:
`transforms.RandomCrop(32, padding=4, padding_mode="reflect")`.
All other settings stayed unchanged: `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, SGD momentum 0.9, weight decay `1e-4`, FP32 channels-last compile path, seed, and once-per-epoch evaluation.

## Execution
One managed local run was executed on GPU 0 after an initial empty shell-background launch was discarded before training started. The managed run started cleanly with CUDA, `822,790` parameters, 390 batches per epoch, and the 300s training budget. The first LR drop fired at step 21000, and the run completed normally in 396.2 total seconds.

## Results
- **Primary metric**: 93.58% (baseline: 93.23%, delta: +0.35 points, +0.38%)
- **Observations**: The run crossed the 93.33% improvement threshold at epoch 72, peaked at 93.58% at epoch 74, and finished with final accuracy 93.35%.
- **Analysis**: The hypothesis was supported. Reflection padding changed augmentation boundary statistics without slowing the run or changing model capacity, and it produced the strongest result so far for the 28/56/112 anchor.
- **Key Learning**: Reflection padding is a high-value augmentation-boundary improvement for the current anchor, lifting best accuracy to 93.58% without overhead.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy; the only tracked source diff was the planned `RandomCrop` padding-mode change, validation cadence remained once per epoch, and runtime stayed under 10 minutes.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed, and `best_test_acc=93.58%` exceeded the 93.33% threshold.

## Unexplored Avenues
- Test `padding_mode="symmetric"` as a sibling boundary-fill variant; it may preserve edge continuity differently from reflection while keeping the same scope and overhead profile.
- Combine reflection padding with a narrow, validated schedule tweak only if future evidence suggests the new anchor's late refinement plateau differs from EXP-016.
- Revisit low-frequency late EMA on top of the new reflection-padding anchor; it remains plausible but carries overhead and BatchNorm risks from prior averaging experiments.

## Next Steps
- High confidence: continue from the new reflection-padding anchor and brainstorm another no-overhead or low-overhead perturbation that preserves 43k-step throughput.
- Medium confidence: try a sibling augmentation-boundary variant such as symmetric padding before stronger regularization.
- Low confidence: retry averaging only with a very sparse late EMA design, because prior EMA/equal-averaging attempts exposed overhead and stability issues.

## Exit Action Results

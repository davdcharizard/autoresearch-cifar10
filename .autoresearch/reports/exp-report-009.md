# Report EXP-009: Weak 8x8 Cutout
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Log**: logs/exp-log-009.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` time-budget harness while modifying only `train.py`. The current accepted baseline is EXP-002 at 91.95%, and the tightened verification rule requires at least +0.10 percentage points, so this experiment needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
The chosen idea was a weaker cutout ablation on the proven FP32 compile/channels-last ResNet-20 recipe. EXP-005 showed 16x16 cutout preserved throughput but over-regularized, so EXP-009 tested a lower-probability fixed 8x8 mask to see whether smaller masking could add useful generalization without the same convergence delay.

## Approach
`train.py` added explicit cutout constants: `USE_CUTOUT=True`, `CUTOUT_PROB=0.25`, `CUTOUT_AREA=0.0625`, and `CUTOUT_RATIO=1.0`. The training transform conditionally appends `transforms.RandomErasing(...)` immediately after normalization, matching EXP-005 placement while reducing mask strength. Architecture, optimizer, LR milestones `[32000, 48000]`, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation were preserved.

## Execution
One local run completed successfully on a single NVIDIA H20-class GPU. Physical GPU 0 was occupied by an unrelated run, so EXP-009 used physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`; this matched the plan's allowed adjustment. The run produced no traceback, CUDA OOM, transform error, or NaN/inf pattern and completed in 406.3 total seconds with 300.0 training seconds.

## Results
- **Primary metric**: 91.87% (baseline: 91.95%, delta: -0.08 points, -0.09%)
- **Observations**: Throughput stayed healthy at 46,047 steps and 119 epochs, so the transform did not create major overhead. Accuracy jumped after the step-32000 LR drop, reached 91.78% at epoch 97, and peaked at 91.87% at epoch 110 before finishing at 91.62%.
- **Analysis**: The hypothesis failed. Weak cutout avoided the severe undertraining seen in stronger regularization bundles, but it still did not beat the FP32 baseline, and it missed the tightened 92.05% threshold by 0.18 points.
- **Key Learning**: Even weak 8x8 cutout does not provide enough generalization benefit for this fixed-budget ResNet-20 recipe.

## Verification
- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed normally, produced numeric metrics, used one GPU, and the tracked diff only contained the planned `train.py` cutout changes.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.87%` is below the 91.95% baseline and below the required 92.05% threshold.

## Unexplored Avenues
- Different augmentation families may still help, but cutout-style masking now has two negative isolated tests and should be deprioritized unless paired with a strong schedule or architecture rationale.
- CutMix or MixUp-like policies might behave differently from erased-patch regularization, but implementing them safely inside the existing fixed-budget training loop would need careful scope control.

## Next Steps
- Try isolated Nesterov momentum only, confidence medium: it is a one-line optimizer ablation not isolated in EXP-000 and has minimal throughput risk.
- Explore a small width increase with measured step-budget calibration, confidence medium-low: width may offer a higher ceiling than depth, but EXP-006 shows capacity changes need careful schedule planning.
- Avoid more schedule-only retuning near `[32000, 48000]`, confidence high: EXP-003 and EXP-008 both reduced accuracy around the current FP32 baseline.

## Exit Action Results

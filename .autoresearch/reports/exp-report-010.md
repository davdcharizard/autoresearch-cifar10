# Report EXP-010: Isolated Nesterov Momentum
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Log**: logs/exp-log-010.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` time-budget harness while modifying only `train.py`. The current accepted baseline is EXP-002 at 91.95%, and the tightened verification rule requires at least +0.10 percentage points, so this experiment needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
The chosen idea was to isolate Nesterov momentum from EXP-000's confounded recipe bundle. The hypothesis was that switching the existing SGD optimizer to Nesterov momentum would improve optimization or late-stage refinement without changing throughput, model capacity, augmentation, loss, schedule, seed, or evaluator.

## Approach
`train.py` added `USE_NESTEROV = True` and passed `nesterov=USE_NESTEROV` to the existing `optim.SGD(...)` call. All other settings were preserved: ResNet-20, batch size 128, LR 0.1, momentum 0.9, weight decay 1e-4, milestones `[32000, 48000]`, seed 42, crop/flip augmentation, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation.

## Execution
One local run completed successfully on physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`. Startup was clean and there were no tracebacks, CUDA OOMs, optimizer configuration errors, or NaN/inf patterns. The run used 300.0 training seconds and 403.6 total seconds, completing 116 epochs and 45,163 optimizer steps.

## Results
- **Primary metric**: 91.33% (baseline: 91.95%, delta: -0.62 points, -0.67%)
- **Observations**: Throughput stayed healthy and step count exceeded EXP-002, so this was not an overhead failure. Accuracy jumped after the step-32000 LR drop but plateaued around 91.3%, peaking at epoch 96 and then drifting down to 90.74% final accuracy.
- **Analysis**: The hypothesis failed. Nesterov preserved the runtime envelope but worsened the accuracy trajectory relative to the EXP-002 classical momentum baseline, suggesting the current schedule and momentum dynamics are better calibrated without Nesterov.
- **Key Learning**: Isolated Nesterov momentum hurts this FP32 ResNet-20 recipe despite preserving throughput.

## Verification
- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed normally, produced numeric metrics, used one GPU, and the tracked diff only contained the planned optimizer flag and SGD argument.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.33%` is below the 91.95% baseline and below the required 92.05% threshold.

## Unexplored Avenues
- Classical momentum should remain the default for this schedule; any future Nesterov retry would need a different LR schedule or momentum value, which would no longer be a clean optimizer-only ablation.
- Mild label smoothing remains the other isolated EXP-000 component, but this result makes the remaining cheap recipe ablations look lower-confidence.

## Next Steps
- Try a compact width increase with measured schedule calibration, confidence medium: cheap recipe changes are plateauing, while WRN evidence still supports width as a higher-ceiling direction.
- Try isolated mild label smoothing only, confidence low-medium: it is still unisolated, but exact top-1 accuracy may degrade even if calibration improves.
- Avoid further optimizer-only Nesterov work under the current schedule, confidence high: this isolated run was clearly below baseline.

## Exit Action Results

# Report EXP-007: Enable TF32 Throughput
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Log**: logs/exp-log-007.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `train.py`-only, single-GPU, 300s training-budget benchmark. The current experiment-index baseline is EXP-002 at 91.95%, and the tightened goal requires a meaningful improvement of at least +0.10 percentage points, so this experiment needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
The chosen idea was to enable TF32 for the proven FP32 compile/channels-last ResNet-20 recipe. Prior successful runs emitted a PyTorch warning that TF32 tensor cores were available but not enabled, so the hypothesis was that TF32 would convert that unused hardware path into more useful optimizer steps without the BF16 accuracy loss seen in EXP-001.

## Approach
`train.py` was modified only to add `USE_TF32 = True` and to enable `torch.set_float32_matmul_precision("high")`, `torch.backends.cuda.matmul.allow_tf32 = True`, and `torch.backends.cudnn.allow_tf32 = True` during CUDA setup before model construction and `torch.compile`. Architecture, optimizer, LR milestones `[32000, 48000]`, batch size, augmentation, seed, compile, channels-last, and validation cadence were preserved.

## Execution
One local run was launched with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup was clean, the previous TF32 warning disappeared, and there were no tracebacks, CUDA OOMs, NaNs, or TF32 API errors. The run completed normally after the fixed training budget with 37,922 optimizer steps and 98 epochs.

## Results
- **Primary metric**: 91.39% (baseline: 91.95%, delta: -0.56 points, -0.61%)
- **Observations**: TF32 did not improve throughput for this workload. It reached only 37,922 steps versus EXP-002's 43,398 steps, hit the first LR drop at step 32,000 / epoch 83, and never reached the second 48,000-step drop.
- **Analysis**: The hypothesis failed. Removing the warning did not mean TF32 would help this small CIFAR CNN; in this setting it appears to degrade effective throughput enough to reduce both schedule exposure and final accuracy.
- **Key Learning**: TF32 is not a useful throughput lever for this ResNet-20 CIFAR-10 recipe; it reduces step budget and misses the tightened improvement threshold.

## Verification
- **Conditions**: Primary metric condition failed; completion, scope, and validation-cadence checks passed.
- **Review Notes**: Results are trustworthy: the run produced numeric metrics, used one GPU, modified only `train.py`, and preserved the fixed evaluator and once-per-epoch validation cadence.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.39%` is below both the 91.95% baseline and the required 92.05% improvement threshold.

## Unexplored Avenues
- Disable TF32 for this model path and treat the warning as non-actionable unless a future architecture has larger matmul/convolution kernels that plausibly benefit.
- If throughput remains the bottleneck, prefer schedule calibration or lower-overhead training-loop changes that can be validated by completed step count before changing arithmetic.

## Next Steps
- **High confidence**: Return to ResNet-20 schedule tuning with the proven FP32 compile/channels-last baseline, focusing on first-drop timing or budget-aware one-drop schedules rather than TF32.
- **Medium confidence**: Test a smaller augmentation or regularization change only if it preserves the EXP-002 step budget, since full cutout and EMA both lost too much effective progress.
- **Medium confidence**: Revisit ResNet-32 only with much earlier measured LR drops; EXP-006 and EXP-007 both show missed schedule phases dominate accuracy.

## Exit Action Results

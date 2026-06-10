# Report EXP-002: FP32 Throughput Without AMP
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Log**: logs/exp-log-002.md

## Goal

Maximize CIFAR-10 `best_test_acc` in the higher-is-better direction while modifying only `train.py` and preserving the fixed `prepare.py` evaluation harness. The baseline before this experiment was 91.52%.

## Idea & Hypothesis

The chosen idea was to isolate precision-preserving throughput after EXP-001 narrowly missed the baseline with BF16 autocast. The hypothesis was that cuDNN benchmarking, channels-last layout, and `torch.compile` would preserve FP32 accuracy while increasing useful optimizer steps enough to exceed 91.52%.

## Approach

`train.py` added `USE_CUDNN_BENCHMARK`, `USE_CHANNELS_LAST`, and `USE_COMPILE`. The model and CUDA input batches use channels-last memory format, cuDNN benchmarking is enabled for fixed-shape batches, and the model is wrapped with `torch.compile`. AMP/autocast was intentionally omitted, and the baseline architecture, augmentation, loss, optimizer, LR milestones, and evaluation cadence were preserved.

## Execution

One local single-GPU run completed normally. Inductor emitted a non-fatal TF32 performance warning during startup, but there were no tracebacks, OOMs, compiler failures, or timeouts. The run used 300.0 training seconds and 396.2 total seconds, completing 112 epochs and 43,398 optimizer steps.

## Results

- **Primary metric**: 91.95 (baseline: 91.52, delta: +0.43, +0.47%)
- **Observations**: Removing BF16 while retaining compile/channels-last increased steps beyond EXP-001 and improved accuracy. The run crossed the first LR milestone and spent substantial time at LR 0.01, peaking at 91.95% before the final epoch.
- **Analysis**: The hypothesis was validated. Throughput was useful, but preserving FP32 arithmetic mattered; EXP-001's BF16 path reached 39,558 steps and 91.48%, while EXP-002 reached 43,398 steps and 91.95%. The result suggests this small ResNet benefits from more fixed-budget optimizer steps and lower-LR refinement, but is sensitive to mixed-precision numerics.
- **Key Learning**: FP32 compile plus channels-last improved fixed-budget training to 43,398 steps and raised `best_test_acc` to 91.95%.

## Verification

- **Conditions**: all passed.
- **Review Notes**: Results are trustworthy. The run completed, reported a numeric metric, changed only `train.py`, left AMP absent, and preserved the benchmark harness and once-per-epoch evaluation cadence.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed and `best_test_acc=91.95%` exceeded the 91.52% baseline.

## Unexplored Avenues

- Enable `torch.set_float32_matmul_precision("high")` before compile, as Inductor warned TF32 tensor cores were available but not enabled; this may improve speed without leaving the FP32-throughput family.
- Retune LR milestones for the new 43k-step budget, especially moving the second drop earlier so the run reaches LR 0.001 before the time limit.
- Try compact WRN-16-2 with this FP32 throughput setup once the baseline recipe has been exploited further.

## Next Steps

- High confidence: add `torch.set_float32_matmul_precision("high")` to test whether the Inductor TF32 warning can unlock more throughput without AMP.
- Medium confidence: adjust milestones to `[32000, 40000]` or similar now that EXP-002 reaches 43k steps.
- Medium confidence: compose FP32 throughput with a compact WRN architecture after schedule tuning.

## Exit Action Results

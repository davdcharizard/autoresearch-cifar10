# Plan EXP-011: One Extra 8x8 Residual Block
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Explicit stage-depth implementation
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-011` from clean accepted commit `eb08811`.
- [x] Modify only `train.py` to replace the scalar block count with `STAGE_BLOCKS=(2,2,3)`, pass and strictly validate the three stage counts in `WideResNet`, and log the exact widths and depths.
- [x] Compile, run complete shape/topology/parameter assertions, audit the full diff, and require byte-identical `prepare.py`.

### Milestone 2: Matched production-path feasibility gate
- [x] Confirm exactly one H20 and local CIFAR files, then import final code through a fail-closed evaluator stub.
- [x] Benchmark exact accepted `[2,2,2]` and candidate `[2,2,3]` FP32 paths using the complete production timed body with pinned copies, separate preregistered mixup/hard-label windows, and a fixed 65/35 time-weighted aggregate.
- [x] Require parameter counts 691,674/987,098, finite `[256,10]` outputs/loss, every regime CV ratio <=0.05, candidate throughput retention >=85%, and calibrated projection >=120 passes before scoring.

### Milestone 3: Single scored run
- [x] Remove stale `run.log`; run exactly once with `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor errors, numerical health, exposure, the sole mixup transition, and completion without changing the run based on interim accuracy.
- [x] Require exit 0 and a complete final summary.

### Milestone 4: Result audit
- [x] Verify one H20, FP32 candidate topology/parameter log, 300.0-300.5 reported counted seconds, total <=600, one transition, continuous accepted decay/floor, and unique accepted-cadence evaluations.
- [x] Record realized passes against the 120 interpretation gate and accepted 141.9, plus best/final accuracy, loss, epochs, steps, VRAM, and best/final gap.
- [x] Accept only `best_test_acc >=94.17%`; a stable negative result with >=120 passes rejects only exact `[2,2,3]`.

## Code Changes
- **`train.py`**: replace `NUM_BLOCKS=2` with `STAGE_BLOCKS=(2,2,3)`; change `WideResNet` to accept and validate exactly three positive strict integer block counts; use each count for its corresponding stage; instantiate with the tuple; log `WRN stages widths=[32,64,128] blocks=[2,2,3]`. No block internals, widths, forward path, initialization, training, or evaluator logic changes.

## Configuration Changes
- Stage depths: `[2,2,2] -> [2,2,3]`; trainable parameters `691,674 -> 987,098`; widths remain `[32,64,128]`.
- FP32, batch 256, peak/floor LR, warmup, momentum, selective decay, alpha-0.2 mixup through 65%, seed, transforms, loader, and evaluation cadence: unchanged.
- No fused optimizer, bottleneck, width change, RandAugment, BF16, batch/LR adjustment, compile, or adaptive fallback.

## Execution Environment
- Method: local/offline evaluator-free feasibility preflight, followed only on pass by one scored `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one NVIDIA H20, existing environment and local CIFAR files; no network, dependency installation, remote service, GitHub, or `gh`.
- Estimated runtime: preflight under one minute; scored run about 340 seconds total.
- Log output: scored output in project-root `run.log`; preflight numbers recorded in `03-execute.md`; remove the log after analysis.
- Tool skill: none.

## Abort Criteria
- Do not launch scoring if semantic/topology checks, parameter counts, CV ratio <=0.05, 85% retention, or 120-pass projection fail. Do not try another depth or architecture.
- Stop and classify scored execution on timeout 124, traceback, CUDA/OOM, non-finite loss, wrong topology/count, missing H20, or wall time >=600 seconds.
- Do not stop for low interim accuracy or retry/reroll a valid result; do not rescue with fused SGD, LR, precision, width, batch, bottleneck, augmentation, or regularization changes.

## Verification Protocol

### Verification Procedure
1. Read the results index and require accepted baseline 94.07 with formal threshold 94.17.
2. Require exactly one `NVIDIA H20`, local `data/cifar-10-batches-py`, and clean accepted base `eb08811`. Compile `train.py`; require only `train.py` differs from the accepted commit and `prepare.py` is identical. Review the complete diff and reject changes outside the stage-depth constant, constructor validation/use, instantiation, and topology log.
3. Through a fail-closed `prepare.Eval` replacement installed before importing final `train.py`, assert no real evaluator/test loader is constructed. Test constructor rejection for wrong-length `(2,2)`, zero/negative counts, float counts, and Boolean counts; validation must use strict integer types. For accepted and candidate models, assert exact parameter counts 691,674/987,098, `[256,10]` output, and full topology: stem; layer lengths; first-block strides/projections; later `layer1` 32-to-32, `layer2` 64-to-64, and `layer3` 128-to-128 blocks, each stride 1 with no shortcut; final BN; and classifier. Specifically require candidate `layer3[2]` to be an identity-shortcut 128-to-128 block. Require a finite FP32 backward/SGD smoke step.
4. In the same evaluator-free process, construct exact accepted `[2,2,2]` and candidate `[2,2,3]` models under reset initialization seeds, then assign independent initially identical training RNG streams. Use exact production selective SGD/Nesterov groups, fixed pinned host inputs/targets, and one shared timed-step implementation reproducing nonblocking copies, LR/group writes, explicit progress branch, zero-grad, Beta/randperm when applicable, FP32 forward/loss, Boolean finite guard, backward, optimizer step, and final synchronize.
5. Warm each path for 25 mixup steps. For the mixup regime, fix identical progress at 50% and measure three continuing 50-step windows in order `accepted-A, candidate-A, candidate-B, accepted-B, accepted-C, candidate-C`. Then fix progress at 80% and repeat the same six-window order for the hard-label regime. Restore/update each path's private RNG state around every window. Record all 12 window means and peak memory; do not change order or weights after observing output.
6. For each model/regime, define the center as the median of its three window mean ms/step and population CV ratio as `statistics.pstdev/mean`. Define each model aggregate as `0.65 * mixup_median_ms + 0.35 * hard_median_ms`, retention as `accepted_aggregate_ms / candidate_aggregate_ms`, and projection as `141.9 * retention`. Require every regime CV ratio `<=0.05`, retention >=0.85, projection >=120, exact counts, finite FP32 loss, correct logits, and no OOM. This operational gate cannot inspect test accuracy.
7. Remove stale log and run the sole scored command `timeout 600s uv run train.py > run.log 2>&1`. Require exit 0 or classify failure from the final 50 lines without a result-conditioned alternative.
8. Require `Device: cuda`, exact widths/depths and 987,098 count in log/summary, reported counted seconds in `[300.0,300.5]` (one completed-step overshoot, bounded well above normal FP32 step time), `num_steps<64000`, finite loss, total <=600, exactly one mixup switch near 195 seconds/LR 0.0612, and unique evaluations only every fifth plus terminal epoch.
9. Compute passes as `num_steps*256/50000`; record against 120 and accepted 141.9. Extract primary/informational metrics and require `best_test_acc>=94.17%`. Any lower score is formal no-improvement with no rerun. If realized passes are below 120, the depth mechanism is inconclusive between inadequate exposure and allocation; only a stable negative at >=120 passes rejects exact `[2,2,3]`. Neither outcome rejects selective depth or low-resolution capacity generally.

### Informational Metrics (Optional)
- Final summary: VRAM, final accuracy/loss, training/total seconds, epochs, steps, parameters.
- Derived: passes, exposure retention, best epoch, best/final gap, evaluation count.
- Preflight: all windows, CVs, median times, retention, projected passes, peak memory.

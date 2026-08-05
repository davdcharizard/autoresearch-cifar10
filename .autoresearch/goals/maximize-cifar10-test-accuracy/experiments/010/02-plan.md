# Plan EXP-010: Selective 160-Channel Final Stage
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Explicit selective-width implementation
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-010` from clean accepted commit `eb08811`.
- [x] Modify only `train.py` to replace the uniform factor with `STAGE_WIDTHS=(32,64,160)`, pass/validate explicit widths in `WideResNet`, and log the exact topology.
- [x] Compile, run shape/parameter assertions, audit the complete diff, and require byte-identical `prepare.py`.

### Milestone 2: Matched production-path feasibility gate
- [x] Confirm one H20 and local CIFAR files, then import final code through a fail-closed evaluator stub.
- [x] Benchmark exactly accepted `[32,64,128]` and candidate `[32,64,160]` FP32 paths using the complete production timed body with pinned copies, separate preregistered mixup/hard-label windows, and a fixed 65/35 time-weighted aggregate.
- [x] Require parameter counts 691,674/961,562, finite `[256,10]` outputs/loss, CV <=5%, candidate throughput retention >=85%, and calibrated projection >=120 passes before scoring.

### Milestone 3: Single scored run
- [x] Remove stale `run.log`; run exactly once with `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor errors, numerical health, exposure, transition, and completion without changing the run from interim accuracy.
- [x] Require exit 0 and a complete final summary.

### Milestone 4: Result audit
- [x] Verify one H20, FP32 candidate topology/parameter log, 300.0-300.1 reported counted seconds, total <=600, one transition, continuous accepted decay/floor, and unique accepted-cadence evaluations.
- [x] Record realized passes against the 120 gate and accepted 141.9, plus best/final accuracy, loss, epochs, steps, VRAM, and best/final gap.
- [x] Accept only `best_test_acc >=94.17%`; a stable negative result with >=120 passes rejects only this allocation.

## Code Changes
- **`train.py`**: replace `WIDEN_FACTOR=2` with `STAGE_WIDTHS=(32,64,160)`; change `WideResNet` to accept and validate exactly three positive integer widths; use them directly for the three stages, final BN, and classifier; instantiate with the tuple; log `WRN-16 stages=[32,64,160]`. No block, forward, initialization, training, or evaluator logic changes.

## Configuration Changes
- Stage widths: `[32,64,128] -> [32,64,160]`; trainable parameters `691,674 -> 961,562`.
- FP32, batch 256, peak/floor LR, warmup, momentum, selective decay, alpha-0.2 mixup through 65%, seed, transforms, loader, and evaluation cadence: unchanged.
- No BF16, batch/LR change, extra block, bottleneck, augmentation, fusion, compile, or regularization change.

## Execution Environment
- Method: local/offline evaluator-free feasibility preflight, followed only on pass by one scored `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one NVIDIA H20, existing environment and local CIFAR files; no network, dependency installation, remote service, or GitHub.
- Estimated runtime: preflight under one minute; scored run about 340 seconds total.
- Log output: scored output in project-root `run.log`; preflight numbers recorded inline in `03-execute.md`; remove the log after analysis.
- Tool skill: none.

## Abort Criteria
- Do not launch scoring if preflight fails semantics, parameter counts, CV, 85% retention, or 120-pass projection. Do not try another width.
- Stop/classify scored execution on timeout 124, traceback, CUDA/OOM, non-finite loss, wrong topology/parameter count, missing H20, or wall time >=600 seconds.
- Do not stop for low interim accuracy or retry/reroll a valid result; do not rescue with LR, precision, width, batch, or regularization changes.

## Verification Protocol

### Verification Procedure
1. Query `exp-index.sh baseline`; require 94.07 and formal threshold 94.17.
2. Require exactly one `NVIDIA H20`, local `data/cifar-10-batches-py`, and a clean accepted base. Compile `train.py`; require only `train.py` differs from `eb08811` and `prepare.py` is identical. Review the entire diff and reject changes outside the explicit-width constant, constructor validation/use, instantiation, and topology log; reject any seed, training, optimizer, evaluator, cadence, or summary change.
3. Through a fail-closed `prepare.Eval` replacement installed before importing final `train.py`, assert no real evaluator/test loader is constructed. Test constructor rejection for wrong-length `(32,64)`, zero/negative widths, float widths, and Boolean widths; validation must use strict integer types. For both accepted and candidate, assert exact parameter counts, `[256,10]` output, and the full topology: stem channels; each stage's first/second block input/output channels; first-block strides; projection presence, channel shapes and strides; second-block absent shortcut; final BN width; and classifier input/output. Require a finite FP32 backward/SGD smoke step.
4. In the same evaluator-free process, construct exactly accepted `[32,64,128]` and candidate `[32,64,160]` models under reset initialization seeds, then assign independent initially identical training RNG streams. Use exact production selective SGD/Nesterov groups, fixed pinned host inputs/targets, and one shared timed-step implementation reproducing nonblocking copies, LR/group writes, explicit progress branch, zero-grad, Beta/randperm when applicable, FP32 forward/loss, Boolean finite guard, backward, optimizer step, and final synchronize.
5. Warm each path for 25 mixup steps. For the mixup regime, fix identical progress at 50% and measure three continuing 50-step windows in order `accepted-A, candidate-A, candidate-B, accepted-B, accepted-C, candidate-C`. Then fix progress at 80% and repeat the same six-window order for the hard-label regime. Restore/update each path's private RNG state around every window. Record all 12 window means and peak memory; do not change order or weights after observing output.
6. For each model/regime, define the center as median of its three window mean ms/step and population CV as `statistics.pstdev/mean`. Define each model's production aggregate as `0.65 * mixup_median_ms + 0.35 * hard_median_ms`, retention as `accepted_aggregate_ms / candidate_aggregate_ms`, and projection as `141.9 * retention`. Require logits `[256,10]`, finite FP32 loss, exact counts 691,674/961,562, every regime CV <=5%, retention >=0.85, projection >=120, and no OOM. This operational gate cannot inspect test accuracy.
7. Remove stale log and run the sole scored command `timeout 600s uv run train.py > run.log 2>&1`. Require exit 0 or classify failure from the final 50 lines without a result-conditioned alternative.
8. Require `Device: cuda`, exact `[32,64,160]` and 961,562 log/summary, reported counted seconds in `[300.0,300.1]` (one-step overshoot plus one-decimal formatting), `num_steps<64000`, finite loss, total <=600, one mixup switch near 195 seconds / LR 0.0612, and unique evaluations only every fifth plus terminal epoch.
9. Compute passes `num_steps*256/50000`; record against 120 and 141.9. Extract primary/informational metrics and require `best_test_acc>=94.17%`. Any lower score is formal no-improvement with no rerun. If realized passes are below 120, classify the capacity mechanism as inconclusive between inadequate exposure and allocation; only a stable negative at >=120 passes rejects `[32,64,160]`. Neither outcome rejects selective capacity generally.

### Informational Metrics (Optional)
- Final summary: VRAM, final accuracy/loss, training/total seconds, epochs, steps, parameters.
- Derived: passes, exposure retention, best epoch, best/final gap, evaluation count.
- Preflight: all windows, CVs, median times, retention, projected passes, peak memory.

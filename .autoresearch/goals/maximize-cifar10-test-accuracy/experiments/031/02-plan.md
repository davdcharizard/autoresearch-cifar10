# Plan EXP-031: End-to-End FP32 Channels-Last Training
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the isolated channels-last path
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-031` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Convert the directly initialized model to `device` plus `torch.channels_last` before optimizer construction, convert each pinned training input during H2D, and add an idempotent forward-entry conversion for frozen-evaluator NCHW compatibility.
- [x] Emit one first-step runtime audit derived from the actual convolution weight and post-mixup model input (`is_contiguous`, stride, dtype), and preserve accepted FP32/TF32/cuDNN flags, topology, state values, optimizer, data/RNG decisions, schedule, augmentation/cutoffs, budget, and evaluation cadence; compile and audit the exact diff.

### Milestone 2: Prove semantics, replay, and material throughput
- [x] Create ignored `experiments/031/preflight.py` with a fail-closed evaluator and independent `git show 67c8e98:train.py` oracle; prove construction-time logical state/RNG/optimizer identity and 987,098 parameters.
- [ ] Prove model/input/activation/gradient/momentum FP32 layout invariants, evaluator NCHW compatibility at full and final-partial batches, exact equal-prefix stochastic decisions, bounded per-tensor accepted/candidate logit/gradient/update/BN-state differences, and bitwise candidate self-replay from a complete post-warm snapshot.
- [ ] Run three paired balanced complete-body H20 timing replicates for early mixup and hard paths; print all measurements before assertions, require every arm CV <=2% and every paired reciprocal-rate speedup >=1.02, and require median projection >=135.667507 passes.

### Milestone 3: Run the sole fixed-seed score if timing qualifies
- [ ] Confirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, clean scope, no stale `run.log`, and passing preflights; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [ ] Monitor numerical/CUDA/worker health and accepted mixup/RandAugment transitions without reacting to interim accuracy; never rerun a valid completion or change layout placement, flags, precision, batch, or schedule.
- [ ] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, FP32 state, one ordered transition per policy, and unique accepted-cadence evaluations. Record realized exposure separately; `<135.667507` is a mechanism miss but does not invalidate or authorize rerunning the score.

### Milestone 4: Classify objective and mechanism
- [ ] Classify objective improvement solely by `best_test_acc >=94.42%`; separately report final-accuracy corroboration `final_test_acc >=94.32%` without allowing it to override the primary metric.
- [ ] Record final loss versus accepted 0.2523, best-final gap, steps/epochs/passes, projected/realized evaluation counts, transitions, VRAM, counted/wall time, and final source audit.
- [ ] Accept into the frontier only when the primary metric and hard task constraints pass. A stable timing miss closes channels-last without scoring; a valid score closes it regardless of metric, with no rescue placement, flag, precision, compilation, fusion, batch, or LR variant.

## Code Changes
- **`train.py` / `WideResNet.forward`**: make `x = x.contiguous(memory_format=torch.channels_last)` the first operation. It is allocation-free for the training path and converts the frozen evaluator's logical NCHW tensor before convolution.
- **`train.py` / model setup**: replace `.to(device)` with `.to(device=device, memory_format=torch.channels_last)` after direct seed-42 construction and before parameter-group creation. Do not alter cuDNN benchmark/deterministic or TF32 policy.
- **`train.py` / counted H2D body**: transfer images with `inputs.to(device, non_blocking=True, memory_format=torch.channels_last)`; retain target transfer and all subsequent mixup/model/loss/update code exactly except for one `step == 0` audit immediately before the first model call. That line must report runtime booleans/strides/dtypes from `model.conv1.weight` and the actual post-mixup input, rather than printing a constant label.
- **`.autoresearch/.../experiments/031/preflight.py`**: ignored verification-only harness for independent accepted comparison, logical-state/layout/FP32/RNG/evaluator/replay checks, and balanced full-body timing. Stub `prepare.Eval`; never construct evaluator/test data or write `run.log`.

## Configuration Changes
- Tensor storage layout: accepted contiguous/NCHW convolution path -> channels-last/NHWC-compatible convolution path for all four-dimensional model weights and activations.
- Dtype/precision: unchanged FP32 storage and accepted installed cuDNN TF32 behavior; no autocast, BF16/FP16, GradScaler, compilation, fusion, or custom kernel.
- Hyperparameters/model/data: unchanged `(2,2,3)`, widths `[32,64,128]`, batch 256, `0.2 ->0.002` cosine LR, momentum 0.9 Nesterov, matrix decay `5e-4`, alpha-0.2 mixup through 65%, early N1/M5 RandAugment, seed 42, 987,098 parameters, loader, and evaluator.

## Execution Environment
- Method: offline local semantic and timing preflights, then one local score only if all gates pass; no remote, network, installs, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, installed PyTorch/CUDA/cuDNN environment, eight persistent forkserver workers.
- Estimated runtime: preflights under 4 minutes; qualified score about 335-345 seconds wall, with a 600-second hard timeout.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any scope/frozen-file/syntax failure; logical initialization or construction RNG mismatch; wrong topology/parameter/optimizer grouping; wrong layout/dtype; evaluator bridge mutation/error at batch 256 or 16; stochastic decision mismatch; non-finite/missing gradients; any per-tensor cross-layout error above fixed bounds; candidate replay mismatch; or evaluator/test access.
- Abort before scoring on any non-finite/error/OOM timing arm, any arm CV >2%, any of three paired exact reciprocal-rate speedups <1.02, or median projected passes <135.667507. Print every raw window and derived value before assertions; never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing/duplicate summary, wrong topology/dtype/layout audit, invalid/repeated transition, duplicate evaluation epoch, or total >=600. Never rerun a valid score; below-target realized exposure remains a valid non-rerunnable result with a failed mechanism claim.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, full diff, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/031/preflight.py`. Require one idle H20, only tracked `train.py`, and no frozen-file drift.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/031/preflight.py --semantics`. Require the independent oracle and candidate to have exact named shapes/dtypes/logical initial values after contiguous normalization, identical post-construction CPU/CUDA RNG, exact optimizer class/groups/settings, and 987,098 parameters.
4. In the same command, require all convolution weights and hooked four-dimensional candidate activations to be channels-last; all floating parameters, buffers, activations, gradients, losses, and momentum state to remain FP32; pinned NCHW H2D to preserve values and produce channels-last; mixup coefficient/permutation/targets/logical values and global RNG to match accepted from restored state; and the hard path to preserve the same input values.
5. On fixed deterministic eval fixtures at both batch 256 and the frozen evaluator's final batch 16, require candidate contiguous-NCHW versus preconverted-channels-last logits to be bitwise equal and state mutation-free. Require accepted/candidate logits within `rtol=2e-4, atol=2e-5` and CE delta <=2e-5; argmax agreement is informational only. For one training step require each named gradient and parameter update to have relative L2 <=1e-3 (for zero-norm accepted tensors require candidate max-abs <=1e-6), every gradient finite/present, accepted optimizer-state schema/dtypes, and per-buffer BatchNorm running statistics within `rtol=2e-4, atol=2e-5` with counters exact. These bounds are fixed before observation and may not be loosened.
6. Prove bitwise candidate replay only after warming the exact candidate shape/layout with accepted backend flags unchanged, synchronizing, clearing `.grad`, and taking a complete snapshot of parameters, buffers, optimizer state, CPU/CUDA RNG, mode, and fixed logical channels-last inputs. Restore that snapshot twice without changing backend flags or autotune caches and require bitwise-equal loss, every named gradient, post-step state, and RNG.
7. Run `timeout 300s uv run python .../experiments/031/preflight.py --throughput`. Build three replicate pairs, each containing accepted/candidate early and hard windows from fresh deterministic logical fixtures; use reversed arm order on the middle replicate. Give each window >=25 warmups and >=50 complete measured pinned-H2D-through-synchronize FP32 SGD steps. Emit raw windows, medians, arm CVs, per-replicate reciprocal speedups, peak VRAM, and payload first. For each replicate compute `speedup_i=(0.65/candidate_early_i + 0.35/candidate_hard_i)/(0.65/accepted_early_i + 0.35/accepted_hard_i)`; require every arm CV <=.02 and every `speedup_i >=1.02`. Compute `projected_passes=133.00736*median(speedup_i)` and require >=135.667507.
8. Reconfirm audit and one idle H20, remove stale log, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
9. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, and one first-step runtime line proving actual conv weight/input channels-last booleans, strides, and FP32 dtype; require no traceback/OOM/non-finite/worker errors. Record `num_steps*256/50000`; `<135.667507` does not invalidate or authorize rerunning the sole completed score, but fails the exposure mechanism.
10. Require mixup disable exactly once at the first >=195-second step and one later RandAugment disable after iterator exhaustion with step lag `[0,195)` and no re-enable. Require unique every-fifth-epoch evaluations plus one final partial epoch; compare projected and realized evaluation opportunities with accepted 27.
11. Classify goal success only by `best_test_acc >=94.42%`. Independently report whether `final_test_acc >=94.32%` corroborates the mechanism and record loss relative to 0.2523; neither secondary signal can overturn the primary metric. Audit final source regardless of verdict.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameter count.
- Transitions/cadence: `run.log` - mixup/RandAugment epoch/step/time, transition lag, unique eval epochs, and evaluation count.
- Preflight: direct output - cross-layout error bounds, layouts/dtypes, timing windows/CVs, reciprocal-rate speedup, projected passes, and peak VRAM.
- Mechanism: realized versus projected passes/evaluations; final loss delta from 0.2523 and endpoint delta from 94.22.

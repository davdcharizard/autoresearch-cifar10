# Plan EXP-029: Batch 128 With a Fully Scaled LR Curve
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Apply the exact four-constant operating point
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-029` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py` frozen.
- [x] Change exactly `BATCH_SIZE 256->128`, `LR 0.2->0.1`, `MIN_LR 0.002->0.001`, and nonbinding `MAX_STEPS 64000->128000`; make no other production edit.
- [x] Preserve accepted `(2,2,3)` initialization, full-stage gradients, early RandAugment, alpha-0.2 batch-shared mixup, SGD/Nesterov/momentum/decay, time schedule, seed, loader semantics, budget, and evaluator cadence; compile and audit the full diff.

### Milestone 2: Prove semantics, exposure, and wall feasibility
- [x] Create ignored `experiments/029/preflight.py` with a fail-closed evaluator and independent `git show 67c8e98:train.py` oracle; prove the four intended constants, byte-equal model state/init RNG, unchanged parameter groups, exact half-LR curve, 390 batches/epoch, and finite batch-128 update.
- [x] Prove batch-128 worker-private RandAugment semantics, exhausted-boundary no-leak, and clean-tail replay against a paired batch-128 crop/flip-only oracle with identical sampler order, worker assignment, and base-transform RNG; never demand per-sample equality to batch 256.
- [x] Run balanced accepted-batch-256 versus candidate-batch-128 H20 timing across mixup/hard paths, including pinned-host copies and the exact production `t0`-to-synchronize body. Restore fresh model/optimizer/RNG fixtures for every arm/window; require every CV <=5%, >=0.9022 image-rate retention, >=120 projected passes, and >=46,875 projected updates.
- [ ] Run balanced real-loader active/inactive timing for both batch sizes at their measured GPU consumer pace; require correct 195/390 batches, CV <=5%, clean cutoff/workers, and conservative absolute/differential-stall wall projections <=500 seconds.

### Milestone 3: Run the sole fixed-seed score
- [ ] Confirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, clean scope, and all preflights; remove stale `run.log` and launch exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [ ] Monitor numerical/CUDA/worker health and both regularization transitions without reacting to interim accuracy; never rerun a valid completion or repair LR/momentum from the result.
- [ ] Require exit 0, 300 counted seconds, total <600, 987,098 parameters, >=120 realized passes, >=46,875 and <128,000 steps, and unique accepted-cadence evaluations.

### Milestone 4: Verify the indivisible operating point
- [ ] Require `best_test_acc >=94.42%` and predetermined endpoint corroboration `final_test_acc >=94.32%` versus the indexed baseline.
- [ ] Record final loss versus accepted 0.2523, best-final gap, steps/epochs/passes, transition times/lags, VRAM, counted/wall time, and the final four-line diff.
- [ ] Accept only if every condition passes. A valid miss closes this exact batch/LR/floor/cap operating point; never infer batch size alone or retry with a repaired floor, momentum, decay, cutoff, or intermediate LR.

## Code Changes
- **`train.py` constants only**: set `BATCH_SIZE=128`, `LR=0.1`, `MIN_LR=0.001`, and `MAX_STEPS=128000`. The step cap preserves the accepted 16,384,000-example safety ceiling and remains nonbinding under the time budget. No function, model, data, optimizer, RNG, timing, logging, or evaluator code changes.
- **`.autoresearch/.../experiments/029/preflight.py`**: ignored verification-only harness for the independent accepted oracle, LR/optimizer/shape checks, paired batch-128 worker cutoff replay, fresh-snapshot complete scored-body GPU timing including pinned-host copies, and paced real-loader/stall timing. It stubs `prepare.Eval` before importing either module and never constructs evaluator/test data.

## Configuration Changes
- Batch size: 256 ->128; batches per full epoch: 195 ->390; both process exactly 49,920 images and drop 80.
- Peak LR: 0.2 ->0.1; floor LR: 0.002 ->0.001; the entire warmup/cosine curve is exactly halved at equal counted-time progress.
- Safety step cap: 64,000 ->128,000; example cap remains 16,384,000.
- Momentum 0.9, Nesterov, matrix decay `5e-4`, vector no-decay, FP32, architecture, augmentation, mixup, schedule fractions, seed, loader workers/context, and evaluator: unchanged.

## Execution Environment
- Method: offline local semantic/GPU/loader preflights, then one local score only on pass; no remote, network, installs, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflights 2-4 minutes; scored run about 345-450 seconds wall, hard timeout 600.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any change beyond the four constants; accepted model state/init RNG/parameter-group mismatch; LR not exactly half at registered progress points; wrong batch/logit/epoch/example count; non-finite update; evaluator/test access; worker RNG/cutoff/no-leak failure; scope/syntax error.
- Abort before scoring on any non-finite/error/OOM timing path, any timing CV >5%, image-rate retention <0.9022, projected passes <120, projected updates <46,875 or >=128,000, loader CV >5%, malformed batches/workers/cutoff, or either wall projection >500. Never relax a stable gate.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing summary, wrong topology/counts, duplicate eval epoch, invalid/repeated transitions, realized passes <120, steps <46,875 or >=128,000, or total >=600. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, full diff, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/029/preflight.py`. Require one idle H20, only tracked `train.py`, and exactly the four constant edits.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/029/preflight.py --semantics`. Require independent accepted/candidate state and post-construction CPU/CUDA RNG equality, unchanged topology/987,098 parameters/groups, candidate `[128,10]` logits and finite update, exactly 390 batches / 49,920 images per epoch, the four constants, and candidate LR exactly half accepted at progress `0, .025, .05, .5, .65, 1`.
4. In the same semantic command, require the installed 14-op RandAugment policy/order/private worker RNG unchanged and a paired batch-128 replay: candidate active iterator then exhausted disable then clean iterator versus a batch-128 crop/flip-only oracle with identical sampler order, worker task assignment, and base-transform RNG. Require marker no-leak, clean outputs equal within that paired oracle, one non-reenable cutoff path, and no evaluator/test construction. No output/RNG equality to the accepted batch-256 loader is required.
5. Run `timeout 240s uv run python .../experiments/029/preflight.py --throughput`. For each mixup/hard regime, perform three paired >=50-step replicates in balanced accepted/candidate order. Every replicate restores fresh deterministic model, optimizer, pinned-host input/target, and private timing RNG fixtures, warms >=20 steps after the relevant batch-shape switch, then times the exact scored region from before nonblocking H2D copies/LR writes through final CUDA synchronization. Compute `image_rate=0.65*B/mixup_ms + 0.35*B/hard_ms`, retention versus accepted, `projected_passes=133.00736*retention`, and `projected_updates=projected_passes*50000/128`; require the fixed gates and every across-replicate CV <=.05.
6. Run `timeout 300s uv run python .../experiments/029/preflight.py --loader-timing`. Use fresh accepted/candidate loaders, explicit production multiprocessing context/workers/prefetch, active and inactive RandAugment phases, and paced consumer times from the corresponding GPU regime medians. Require 195/390 finite batches, the same 49,920 images, clean inactive epoch/workers, and CV <=.05. Let each phase's excluded stall be `max(0, epoch_wall - batches*consumer_step_s)`; form 65/35 weighted epoch wall and stall for each batch size. With `projected_epochs=projected_passes*50000/49920`, require both `345.3 + max(0,candidate_stall-base_stall)*projected_epochs <=500` and conservative absolute `45.3 + candidate_epoch_wall*projected_epochs <=500`. Raw paced epoch differences must not be counted as excluded overhead.
7. Reconfirm audit and one idle H20, remove stale log, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
8. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, at least 46,875 but fewer than 128,000 steps, `num_steps*128/50000 >=120`, and no traceback/OOM/non-finite/worker errors.
9. Require one mixup transition at/after 195 seconds and one later RandAugment disable after normal iterator exhaustion with step lag `[0,390)`; require unique every-fifth-epoch evaluations plus one final partial epoch.
10. Require `best_test_acc >=94.42%` and `final_test_acc >=94.32%`; stop on either failure. Audit the final diff and record final loss relative to 0.2523 regardless of verdict.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameter count.
- Operating point: `run.log` - batch count, transition epoch/step/time/lag, examples per epoch, and best-final gap.
- Preflights: direct output - per-regime timing windows/CVs/image rates, projected passes/updates, loader epoch times/CVs, and wall projections.
- Mechanism: final loss delta from 0.2523 and endpoint delta from 94.22; worse values support the weaker-floor/horizon risk but cannot identify one coupled cause.

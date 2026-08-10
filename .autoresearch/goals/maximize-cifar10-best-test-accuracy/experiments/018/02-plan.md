# Plan EXP-018: Late Arithmetic SWA with In-Budget BN Recalibration
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Isolated production implementation
- [x] Confirm the 94.15 frontier at `7c1e7d8`, create `autoresearch/maximize-cifar10-best-test-accuracy-018`, and require only known untracked `data/` with no stale run log.
- [x] Add fixed `SWA_START_FRACTION=0.86` and `SWA_FINALIZE_FRACTION=0.98`; leave model, optimizer, data, schedule, seed, evaluator, and accepted 80% switch unchanged.
- [x] Maintain a detached same-device FP32 arithmetic parameter mean at completed weak-tail epoch endpoints in `[86%,98%)`, charging allocation, averaging, copy, and distance-reduction work to `total_training_time`.
- [x] At 98%, stop SGD permanently, install the mean, reset every BN running buffer/counter, and spend the remaining counted time on no-grad weak-loader forwards before one terminal evaluation.
- [x] Preserve the ten-field summary and add one bounded `swa_finalization` provenance line; pass syntax, formatting, tracked-scope, static evaluator-count, and exact parameter-count checks.

### Milestone 2: Arithmetic, state, and control semantics
- [x] Create ignored controllers with explicit project-root import bootstrapping and obtain mandatory external Claude approval of the exact production diff/controllers before running them; no fallback reviewer.
- [x] Extract importable `SWAAccumulator`, timed snapshot/install, cumulative BN-reset/refresh, and state-validation helpers in `train.py`; controllers must call these exact production helpers rather than copy their logic.
- [x] Prove explicit FP64-reference arithmetic across at least seven known snapshots, stable parameter ordering, correct first/previous/mean state, median consecutive normalized distance `>=1e-6`, and first-to-last distance `>=1e-5`.
- [x] Prove snapshots never mutate online parameters, buffers, gradients, optimizer/momentum, loader/RNG state, or parameter-group membership and that every operation is charged exactly once.
- [x] Prove the accepted online trajectory is bitwise identical before the first snapshot and that each later snapshot changes only declared shadow/diagnostic state plus counted timer progress; any resulting LR/exposure displacement is charged and attributed, never described as bitwise trajectory identity.

### Milestone 3: BN recalibration and budget feasibility
- [x] Reset all BN running means/variances/counters exactly after SWA installation, temporarily set BN momentum to `None`, run at least 390 hard-target weak batches with explicit iterator recreation, restore original momenta, and require finite non-default buffers with aligned counters.
- [x] Prove recalibration changes buffers only: SWA parameters, gradients, optimizer state, RNG-independent evaluator inputs, and `num_steps` remain unchanged.
- [x] In five fresh timing processes, bound all projected snapshot bookkeeping below 0.5 counted seconds, one 390-batch weak refresh below 4.5 seconds, peak allocation below 700 MiB, and iterator wait headroom.
- [x] A joint conservative schedule projection must yield at least eight snapshots, at least 26,200 optimizer steps, at least 390 refresh batches, no more than 19 evaluations, and total wall below 540 seconds; production floors remain seven snapshots and 26,091 steps.

### Milestone 4: One fixed-seed production run
- [x] Reconfirm all conjunctive gates, official baseline, sole idle H20, tracked scope, fixed seed/evaluator, lifecycle, and absence of stale logs.
- [x] Run exactly once under the 600-second supervisor as `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`; never stream full output or rerun a valid result.
- [x] Monitor bounded tails/process state and abort only for crash, non-finite state, missing progress, resource/protocol failure, or timeout, never for online/SWA accuracy.

### Milestone 5: Integrity, metric, and attribution verification
- [x] Parse the ten finite summary fields plus the single SWA line; require 300 counted seconds, total below 600, 1,073,962 parameters, at least 26,091 optimizer steps, at least seven snapshots, at least 390 refresh batches, and `install_step == num_steps`.
- [x] Require one 80% switch, eight stopped workers, CutMix `[45%,55%]`, hard weak/refresh targets, unique evaluation epochs, at most one evaluation per epoch, and at most 19 evaluations.
- [x] A mergeable improvement requires `best_test_acc >=94.25`, final SWA accuracy `>=94.25` and `>=pre_swa_best`, median consecutive spread `>=1e-6`, and first-last spread `>=1e-5`; final SWA NLL below 0.1934 is the primary mechanistic diagnostic.
- [x] Record switch/first-weak/pre-SWA-best/final-SWA/NLL/exposure/distance/timing diagnostics, preserve `run.log` through analysis, then remove it before the next experiment.

## Code Changes

- **`train.py` importable helpers**: add a focused `SWAAccumulator` plus timed snapshot/install, cumulative BN-reset/refresh, and validation helpers. `main()` and ignored controllers call these exact helpers. Shadows contain parameters only; model buffers, optimizer state, and checkpoints are never averaged.
- **`train.py` constants/state**: add the fixed 0.86 snapshot-start and 0.98 finalization fractions plus shadow/counter/timing/distance state. Pre-register median consecutive normalized RMS `>=1e-6` and first-last RMS `>=1e-5` as nondegenerate spread floors.
- **`train.py` weak epoch endpoint**: after an ordinary online weak epoch but before that epoch's single evaluation, if post-training progress is in `[0.86,0.98)`, call the production timed snapshot helper. Snapshot one clones detached FP32 CUDA parameters; later snapshots update the uniform mean by `avg.lerp_(online, 1/count)`. Keep first and previous copies solely for charged distance diagnostics. If charged work crosses 98%, finalize before evaluation so the epoch still receives one look. No snapshot consumes RNG or changes the online model.
- **`train.py` 98% boundary**: add a weak-phase inner-loop break when progress reaches 0.98. Behind an explicit one-shot `swa_finalized` guard and before the existing evaluation branch, require at least seven snapshots and both spread floors, store `pre_swa_best_acc` and `install_step`, install the mean, reset BN buffers/counters, set each BN momentum to `None`, and charge the synchronized operation.
- **`train.py` BN refresh**: require at least 4.5 seconds of remaining counted budget, then reuse the active hard-target weak loader in `model.train()` under `torch.no_grad()`, explicitly creating a new iterator after each exhaustion while retaining persistent workers. Count synchronized H2D plus forward work until the original 300-second counter is exhausted, require at least 390 batches and aligned counters, restore each original BN momentum, never call backward/SGD, then let the existing evaluator perform one terminal SWA evaluation.
- **`train.py` provenance**: print one `swa_finalization` line containing snapshot count/range, `install_step`, pre-SWA online best, final SWA accuracy/loss, refresh batches, snapshot/install/refresh seconds, and consecutive/first-last distances. Keep the existing ten summary keys unchanged and require final `num_steps == install_step`.
- **Ignored EXP-018 controllers**: add arithmetic/state and real weak-loader/timing controllers under the experiment directory. They may import production code but cannot alter tracked runtime behavior.

## Configuration Changes

- SWA snapshot interval: none -> completed weak epoch endpoints with counted progress in `[0.86,0.98)`; preflight expects at least eight and production requires at least seven. This begins after initial weak adaptation while spanning LR approximately 0.0080 to 0.00034.
- SWA weighting: none -> uniform arithmetic parameter mean, with no EMA decay, checkpoint weighting, averaging of BN buffers, optimizer averaging, or CPU shadow.
- Final 2%: ordinary SGD -> installed SWA parameters plus weak-data BN refresh, all inside the same 300-second counter.
- Mergeable-result integrity: final SWA accuracy must itself be `>=94.25` and `>=pre_swa_best`, with median consecutive spread `>=1e-6` and first-last spread `>=1e-5`; final SWA NLL `<0.1934` remains diagnostic.
- Unchanged: width 2, 1,073,962 parameters, batch 128, FP32 eager execution, standard SGD/momentum/all-parameter decay, LR schedule through 98%, seed 42, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, hard weak loader, evaluator/checkpoints, worker lifecycle, and wall supervisor.

## Execution Environment

- Method: local commands from the project root; production command is `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB, pinned as visible device 0; no dependency or environment changes.
- Estimated runtime: 4-8 minutes for reviewed controllers and approximately 5.5 minutes for the sole production run; each command has a hard timeout.
- Log output: bounded controller JSON under EXP-018; production output only in `run.log`, never `tee` or full streaming.
- Tool skill: none; local execution.

## Abort Criteria

- Preflight no-go on any tracked-scope, arithmetic-reference, parameter-order, mutation/isolation, RNG, timer-charge, BN-reset, buffer-only refresh, hard-target, lifecycle, snapshot-count, exposure, memory, evaluation-count, or wall-projection failure. Do not change windows, weighting, reserve, BN policy, or use a fallback candidate.
- Reject implementation if the first snapshot or any later update is non-FP32/nonfinite, the mean differs from the explicit reference beyond FP32 rounding tolerance, online tensors change, endpoint distance is nonfinite/zero, or optimizer/gradient state is touched.
- Reject feasibility if all projected snapshot transactions jointly consume 0.5 seconds or more, 390 weak refresh batches consume 4.5 seconds or more, the joint conservative projection falls below 26,200 steps or eight snapshots, peak allocation reaches 700 MiB, expected evaluations exceed 19, or projected total reaches 540 seconds.
- During production, terminate only for crash, non-finite output/state, missing progress beyond measured startup, GPU/resource/lifecycle/protocol fault, fewer than seven snapshots at finalization, fewer than 390 refresh batches, or the 600-second timeout. Never stop for low online accuracy, worse NLL, or an unfavorable SWA result.
- Any exit-zero run with a complete finite summary and all fixed scope/seed/timer/evaluator/lifecycle/SWA-integrity conditions is valid and non-rerunnable regardless of accuracy. Repair is allowed only when no usable summary exists and an independent controller/implementation/environment defect is demonstrated without changing the reviewed SWA mechanism. No fallback reviewer or candidate is allowed.

## Verification Protocol

### Verification Procedure

1. **Baseline, branch, scope, and GPU (30 seconds).** Query `exp-index.sh baseline` and require 94.15 at `7c1e7d8`; inspect `git status --short --branch`, `git rev-parse --short HEAD`, `git diff --name-only`, and `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`. Require the EXP-018 branch from integration, only `train.py` tracked, known `data/`, no `run.log`, and one idle H20.

2. **Static and external implementation review (120 seconds plus reviewer latency).** Run `git diff --check`, `uv run python -m py_compile train.py`, Ruff/format checks available in the repo, and inspect the full diff. Create ignored controllers with root import bootstrapping, then give Claude the exact diff, plan, controllers, goal, brainstorm, and `prepare.py` under the plan-critic prompt. Persist the exit-zero nonempty review; on non-zero/empty output retry Claude or treat authentication as a user blocker, never substitute self/subagent review. Any source correction requires focused Claude re-approval.

3. **Arithmetic/state gate (90 seconds).** Run `CUDA_VISIBLE_DEVICES=0 timeout 90s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/018/preflight_swa.py --arithmetic`. Require `ARITHMETIC_GATE_PASS` from the exact imported production helpers, explicit FP64-reference agreement over at least seven known snapshots, unchanged online/optimizer/BN/RNG state, correct uniform count, distance floors, parameter ordering, and exact one-time charge accounting in persisted JSON.

4. **Real BN refresh/lifecycle gate (180 seconds).** Run `CUDA_VISIBLE_DEVICES=0 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/018/preflight_swa.py --refresh`. Require `REFRESH_GATE_PASS` from the production refresh helper, exact mean installation and BN reset, temporary cumulative momentum, original-momentum restoration, at least 390 hard weak batches across explicit iterator recreation, finite non-default aligned BN buffers/counters, unchanged parameters/optimizer/step count, eight-worker shutdown evidence, and no remaining process.

5. **Fresh-process timing and schedule gate (360 seconds).** Run one unscored device-conditioning process, then five fresh candidate processes via `CUDA_VISIBLE_DEVICES=0 timeout 360s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/018/timing_swa.py`. Measure the exact production helpers for the projected snapshot count and 390-batch cumulative refresh, including synchronization and H2D/forward boundaries. Under their joint conservative bounds require `TIMING_GATE_PASS`, snapshot work `<0.5s`, refresh `<4.5s`, peak `<700 MiB`, projected snapshots `>=8`, steps `>=26,200`, evaluations `<=19`, and wall `<540s`, with raw JSON.

6. **One production run (600 seconds).** Reconfirm steps 1-5 and launch exactly once with `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`. Poll the existing process every 30-60 seconds and inspect only bounded CR-normalized tails/process/GPU state. Exit 124 or any abort criterion is failure; accuracy never authorizes termination or retry.

7. **Necessary-condition verification (60 seconds).** Extract the ten summary keys plus `augmentation_switch`, `eval ep`, and `swa_finalization` using bounded `rg`. Numerically require a complete finite summary, approximately 300 counted seconds, total `<600`, 1,073,962 parameters, `best_test_acc >=94.25`, final SWA accuracy `>=94.25` and `>=pre_swa_best`, at least 26,091 steps, `install_step == num_steps`, at least seven snapshots, both spread floors, at least 390 refresh batches, cumulative BN counters/momentum restoration, one switch near 80%, eight workers stopped, CutMix `[45%,55%]`, hard weak refresh targets, unique epochs, at most one eval per epoch, and at most 19 evaluations. Query the moving baseline again before verdict.

8. **Attribution and cleanup.** Compare final SWA NLL with 0.1934 after the final SWA accuracy itself clears all metric/online-best gates. NLL non-improvement weakens mechanism evidence but cannot turn a valid accuracy pass into a failure. Record switch, first weak, pre-SWA best, final SWA, NLL, endpoint spread, snapshot/refresh cost, exposure, eval count, VRAM, startup, and total wall against EXP-010. Keep `run.log` through analysis, then remove it. On no-go/no-improvement restore only `train.py`, return to integration, and preserve `data/`.

### Informational Metrics (Optional)

- Standard final accuracy/loss, training/total/startup seconds, VRAM, epochs, steps, and parameters: ten final `run.log` summary lines.
- Switch, first weak, pre-install online best, final SWA accuracy/NLL, best-final gap, and evaluation count: bounded `augmentation_switch`, `eval ep`, and `swa_finalization` parse.
- SWA sample progress/count, normalized endpoint distances, snapshot/install/refresh seconds, refresh batches, BN state, and memory: SWA provenance plus ignored controller JSON.

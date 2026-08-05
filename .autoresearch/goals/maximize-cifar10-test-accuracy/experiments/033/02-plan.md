# Plan EXP-033: Three-Point Terminal Parameter Average
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the exact finite-average treatment
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-033` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Add fixed snapshot fractions `(0.95, 0.975)`, capture each trainable-parameter snapshot exactly once after the first update whose pre-step counted time is at least its threshold, and charge its clone work before the existing synchronization/timer closes.
- [x] At the existing budget-exhausted evaluation only, fully materialize and finite-check the exact uniform average, install it, keep terminal live buffers, evaluate once, and restore plus elementwise-verify every live parameter in `finally`.

### Milestone 2: Prove state, timing, and evaluation isolation
- [x] Create ignored `experiments/033/preflight.py` with an independent `git show 67c8e98:train.py` oracle and fail-closed fake evaluator; prove accepted construction/model/optimizer/RNG identity and 987,098 parameters.
- [x] Prove exact accepted trajectory through the first due snapshot, then bound the intended timer/LR/step divergence from counted clone cost; also prove threshold order/count, exact FP32 arithmetic and parameter coverage, detached/unregistered snapshots, terminal-buffer identity, unchanged parameter/optimizer object references, exception-safe restoration, finite guards, and exactly one terminal evaluator call.
- [x] Measure production-sequence snapshot and terminal-swap overhead on the H20 with raw balanced windows emitted before assertions; require every CV <=5%, projected retention >=99%, projected passes >=131.6772864, and conservative wall projection <500 seconds.

### Milestone 3: Run the sole fixed-seed score
- [x] Confirm baseline 94.32 at `67c8e98`, one idle H20, local CIFAR-10, frozen evaluator, exact scope, no stale `run.log`, and passing preflight; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Monitor numerical/CUDA/worker health and the two snapshot plus accepted mixup/RandAugment transitions without reacting to interim accuracy; never rerun a valid completion or adjust fractions, weights, parameter set, BN policy, or seed.
- [x] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, at least 131.6772864 passes, 987,098 parameters, two ordered snapshot markers, one averaged terminal evaluation, unique accepted-cadence evaluations, and exact elementwise live-state restoration.

### Milestone 4: Classify the boundary-representative result
- [x] Classify objective improvement solely by `best_test_acc >=94.42%`; independently require `final_test_acc >=94.42%` to support the averaging mechanism and report final loss versus 0.2523 without letting corroboration override the primary verdict.
- [x] Record best/final/loss deltas, best-final gap, steps/epochs/passes, snapshot steps/times, evaluation count, transitions, VRAM, counted/wall time, timing retention, and final source/state audits.
- [x] A valid normal-exposure averaged endpoint below 94.42 closes exact uniform `[95%,97.5%,100%]` trainable-parameter averaging with terminal live BN buffers; do not rescue with another window, checkpoint count, coefficient, buffer average, or recalibration.

## Code Changes
- **`train.py` / configuration**: define `AVERAGE_FRACTIONS = (0.95, 0.975)` as the only new hyperparameter. The implicit third checkpoint is the terminal live state and all three coefficients are exactly `1/3`.
- **`train.py` / training loop**: retain an ordered external list of detached FP32 device snapshots over all and only `model.parameters()`. At each step, compute whether the next threshold is already satisfied from the existing pre-step `total_training_time`; after `optimizer.step()`, clone the parameter list exactly once when due and log fraction/step/pre-step time. Perform cloning before the existing `torch.cuda.synchronize()` and `dt` calculation so its GPU work is counted.
- **`train.py` / terminal evaluation**: only when `budget_exhausted` is true, require exactly two finite snapshots; clone terminal live parameters; fully materialize `((snapshot_95 + snapshot_97_5) + terminal_live) / 3.0` in that fixed FP32 operation order and verify every result finite before overwriting any live value. Install the materialized list under `torch.no_grad()` and call the existing evaluator once. Retain all terminal live buffers, parameter objects, optimizer objects/state, and evaluation cadence. Restore terminal live parameters in `finally`, including when install/evaluation raises; then require `torch.equal` for every parameter against its backup and log one compact exact-restoration marker.
- **`.autoresearch/.../experiments/033/preflight.py`**: ignored verification-only harness for exact source scope, snapshot scheduling/arithmetic/coverage, optimizer and buffer isolation, injected-failure restoration, fake-evaluator call count, and H20 overhead timing. It must replace `prepare.Eval` before importing accepted/candidate modules, forbid `CIFAR10(train=False)`, and never create `run.log`.

## Configuration Changes
- Snapshot fractions: none -> `(0.95, 0.975)` of the 300-second counted budget, corresponding to pre-step thresholds 285.0 and 292.5 seconds.
- Average: none -> uniform arithmetic mean of snapshot-95, snapshot-97.5, and terminal live trainable parameters.
- Evaluation state: ordinary live parameters for all nonterminal evaluations; averaged trainable parameters plus terminal live BN buffers for the sole budget-exhausted evaluation.
- Accepted model/optimization/data configuration: unchanged `(2,2,3)`, batch 256, FP32, LR `0.2 -> 0.002`, Nesterov momentum 0.9, matrix decay `5e-4`, batch-shared alpha-0.2 mixup through 65%, early worker-safe N1/M5 RandAugment through the first exhausted epoch at/after 65%, seed 42, worker setup, 300-second budget, and evaluator.

## Execution Environment
- Method: offline local semantic/timing preflight, then one local score only on pass; no network, remote, installs, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflight under 3 minutes; score about 345-360 seconds wall with a 600-second hard timeout.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis and then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any scope/frozen-file/syntax failure; accepted construction/model/optimizer/RNG mismatch before the first snapshot; wrong topology/parameter count; threshold ambiguity or duplicate/missing/out-of-order snapshots; buffer inclusion; missing parameter; snapshot registration/aliasing; object/reference/optimizer mutation; non-finite arithmetic; restoration failure; evaluator/test access; duplicate terminal evaluation; or any unaccounted divergence beyond the intended counted snapshot timer/LR/terminal-step effect.
- Abort before scoring if any overhead window is non-finite, any CV exceeds 5%, projected retention is below 99%, projected passes are below 131.6772864, wall projection is at least 500 seconds, or H20 memory is unsafe. Emit raw timing and projections before assertions and never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss/state, no output for 60 seconds, malformed/missing/duplicate summary, missing/duplicate/wrong-order snapshot markers, a snapshot outside its first eligible post-update convention, wrong topology, invalid/repeated mixup or RandAugment transition, duplicate evaluation epoch, more than one terminal evaluator call, restoration failure, or total >=600 seconds. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, `git diff --unified=0 67c8e98 -- train.py`, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/033/preflight.py`. Require one idle H20, only tracked `train.py`, and the exact finite-average implementation without evaluator changes.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/033/preflight.py semantics`. Guard `prepare.Eval` and test-data construction before importing either source. Require byte-equal accepted/candidate initial model state, construction CPU/CUDA RNG, optimizer group membership/state/settings, unchanged data/augmentation/mixup/LR code, and 987,098 FP32 trainable parameters. Require exact accepted live computation and RNG through the first due snapshot; after it, recognize counted clone latency as the sole intended cause of time-derived LR/progress and possible terminal-step divergence rather than claiming impossible trajectory identity.
4. Drive a deterministic fake loop through pre-step times immediately below, at, and above 285.0 and 292.5 seconds. Require each due decision to be based on pre-step counted time, then capture the first post-SGD state after the threshold is already met, exactly once and in order. Record fraction, step, and pre-step time; reject crossing-step retroactive snapshots.
5. On toy and real models with independently constructed tensor patterns, require snapshot coverage to equal `named_parameters()` exactly; all snapshots detached, nonaliased, unregistered, FP32, device-local, and finite; no buffer or optimizer tensor included. Fully materialize all averaged tensors, fail before install if an input or result is non-finite, and require exact equality to the fixed arithmetic oracle `((s95 + s97_5) + terminal) / 3.0`, while every BN running mean/variance/counter and other buffer remains bitwise terminal.
6. Record parameter object IDs, optimizer references/state, live values, buffers, gradients, and RNG around averaging. Require unchanged IDs/references/state, one fake terminal evaluator call whether or not epoch is divisible by five, no averaging on nonterminal cadence evaluations, and elementwise `torch.equal` restoration of every live parameter after both success and an injected exception following a partial install. Prove backup/materialize/install/restore arithmetic alone consumes no CPU or CUDA RNG; separately use a fake evaluator with controlled CPU/CUDA draws to require the candidate terminal path has exactly the same RNG delta as one accepted evaluator call. Never restore evaluator-consumed RNG.
7. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/033/preflight.py timing`. On the idle H20, use at least 20 warmups and three balanced windows of >=50 repetitions. Define one candidate repetition as exactly one production `detach().clone()` sweep over all parameters followed by synchronization; define its paired control as the same parameter iteration and synchronization without allocation/copy. Alternate order, release temporary clones between repetitions, and emit candidate/control windows and their paired differences. Separately time exactly one terminal backup plus fully materialized fixed-order average, install, elementwise restore, and restoration check. Print all windows, medians, population CVs, peak allocation, projected timer/LR offset, retention/passes, possible lost-step bound, and wall projection before assertions.
8. Define `per_snapshot_increment_s=max(0, median(candidate_window_s-control_window_s)/repetitions_per_window)` and charge exactly two increments against 300 counted seconds: `retention=(300 - 2*per_snapshot_increment_s)/300`, `projected_passes=133.00736*retention`; require every candidate/control/terminal window family CV <=5%, retention >=0.99, and passes >=131.6772864. Bound subsequent time-derived LR offset from the cumulative two-snapshot cost and conservatively allow at most the corresponding rounded-up loss in terminal steps. Add the separately measured terminal sequence to accepted wall 345.3 and require <500 seconds. Treat high CV as a stable qualification miss, not a retry condition.
9. Reconfirm audit and one idle H20, remove stale `run.log`, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
10. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, at least `131.6772864` passes from `num_steps*256/50000`, exactly two snapshot markers with pre-step times at/above 285.0/292.5 and below 300, and exactly one terminal-average marker/evaluation. Require `terminal_restore_exact=true`, produced only after elementwise `torch.equal` comparison of every restored parameter with its terminal backup; log the first mismatched parameter and fail if false.
11. Require mixup disable exactly once at the first >=195-second step and one later RandAugment disable after iterator exhaustion with step lag `[0,195)`, no re-enable, unique every-fifth-epoch evaluations plus one final partial epoch, and no live/average double evaluation at terminal cadence.
12. Classify goal success only by `best_test_acc >=94.42%`. Independently classify mechanism support only if the averaged `final_test_acc >=94.42%`; report final loss versus 0.2523, but neither endpoint nor loss can overturn the goal verdict. Audit final production source and restored live state regardless of result.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameters.
- Snapshot semantics: `run.log` - 95%/97.5% fraction, step, pre-step counted time, terminal-average marker, and exact elementwise restoration result.
- Transitions/cadence: `run.log` - mixup/RandAugment epoch/step/time, transition lag, unique live evaluation epochs, terminal averaged epoch, and total evaluator calls.
- Preflight: direct output - parameter/buffer coverage, restoration/exception checks, raw overhead windows/CVs, retention/pass projection, wall projection, and memory.
- Mechanism: best/final/loss deltas from accepted 94.32/94.22/0.2523 and best-final gap.

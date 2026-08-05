# Plan EXP-028: Freeze the High-Resolution Prefix for the Hard Tail
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement one-way exhausted-boundary freezing
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-028` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py` frozen.
- [x] Add one strict freeze helper that clears outstanding gradients and sets `requires_grad_(False)` exactly on `model.conv1` and `model.layer1` parameters, leaving module training mode, BN buffers, optimizer groups, optimizer state, and every parameter value intact.
- [x] Extract a testable one-way early-phase controller and invoke it after every iterator with explicit `iterator_exhausted` and `budget_exhausted` flags. It may disable RandAugment and freeze once only after the accepted first exhausted >=65% iterator with budget remaining; log epoch, step, counted time, frozen/remaining counts, and `iterator_exhausted=true`.
- [x] Preserve accepted `(2,2,3)` topology, early RandAugment, alpha-0.2 mixup, batch-256 FP32 SGD, time-LR curve, weight decay before freezing, seed, loader, budget, and evaluator cadence; compile and audit the full diff.

### Milestone 2: Prove semantics and material tail acceleration
- [x] Create ignored `experiments/028/preflight.py` with a fail-closed evaluator; load accepted `train.py` independently from `git show 67c8e98:train.py`, prove candidate identity through the boundary, and prove exact controller/freeze semantics on 33,424 prefix / 953,674 remaining parameters.
- [x] Prove frozen parameter values and momentum buffers remain bitwise fixed over multiple tail steps, all eligible upper parameters receive finite updates, prefix outputs require no gradient, and every prefix BN running buffer remains live and matches an unfrozen forward reference on the first tail batch.
- [x] Run balanced complete hard-tail H20 timing with identical boundary state and full production timed bodies; require CV <=5% and calibrated projected passes `(16770 + 9208 * accepted_ms/frozen_ms) * 256 / 50000 >=145.0`, using EXP-027's observed exhausted-boundary and final steps.

### Milestone 3: Run the sole fixed-seed score
- [x] Confirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, clean scope, and passing preflights; remove stale `run.log` and launch exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Monitor numerical/CUDA/worker health and the mixup, exhausted RandAugment, and prefix-freeze transitions without reacting to interim accuracy; never rerun a valid completion.
- [x] Require exit 0, 300 counted seconds, total <600, 987,098 original parameters, exactly 33,424 frozen at the transition, >=145 realized passes, and unique accepted-cadence evaluations.

### Milestone 4: Verify accuracy and mechanism
- [x] Require both `best_test_acc >=94.42%` and `final_test_acc >=94.42%`, exactly 0.10 points above the indexed baseline, eliminating acceptance through extra best-of-evaluation opportunities.
- [x] Record final loss versus accepted 0.2523, best-final gap, steps/epochs/passes, transition times/lags, VRAM, counted/wall time, and final source diff.
- [x] Accept only if every goal and preregistered condition passes. A valid miss closes exact whole-prefix freezing at the exhausted 65% boundary; never tune cutoff, frozen subset, optimizer state, BN semantics, or exposure gates from the result.

## Code Changes
- **`train.py` / freeze helper**: add `freeze_training_prefix(model, optimizer)` that first clears gradients with `optimizer.zero_grad(set_to_none=True)`, then freezes every parameter reachable from `model.conv1` and `model.layer1`. It validates exactly 33,424 unique prefix parameters, 953,674 remaining trainable parameters, and unchanged optimizer membership/order, then returns counts for logging. It does not detach activations, rebuild groups, delete momentum, alter module modes, freeze buffers, or consume RNG; value/state equality is verified outside production.
- **`train.py` / boundary controller**: add `maybe_finish_early_phase(...)` taking explicit iterator/budget state. It returns unchanged active state for live, early, budget-exhausted, or already-finished calls; only a normal exhausted >=65% call flips the shared byte, invokes the freeze helper, logs both same-boundary transitions, and returns inactive. `main()` tracks whether the `for` iterator exhausted normally and has no unfreeze path.
- **`.autoresearch/.../experiments/028/preflight.py`**: ignored verification-only harness for an independent `git show 67c8e98:train.py` accepted oracle, candidate identity, real controller/freeze semantics, multi-step state/gradient/BN behavior, source guards, and balanced H20 hard-tail timing. It stubs `prepare.Eval` before importing either module and never constructs evaluator/test data.

## Configuration Changes
- Trainable parameters during first 65%: 987,098 -> unchanged.
- Trainable parameters after exhausted boundary: 987,098 -> 953,674; frozen prefix: 33,424.
- Prefix forward and BN-buffer behavior: unchanged train-mode computation; only parameter gradients, momentum updates, and weight decay stop after freezing.
- Architecture, initialization, optimizer construction/groups, LR/momentum/decay constants, batch, mixup, RandAugment, seed, loader, evaluation, and time budget: unchanged.

## Execution Environment
- Method: offline local semantic/timing preflights, then one local score only on pass; no remote, network, installs, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflights under 3 minutes; scored run about 345 seconds wall, hard timeout 600.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any accepted pre-boundary state/logit/loss/gradient/optimizer/RNG mismatch; wrong frozen set/count; parameter value change during freeze; optimizer membership/state deletion; frozen tail gradient/update; upper gradient failure; prefix BN buffer freeze/mismatch; duplicate/reversible/wrong-boundary transition; evaluator/test access; scope/syntax error.
- Abort before scoring if timing is non-finite, any path errors/OOMs, any window CV >5%, or boundary-calibrated projected passes <145.0. Treat analytic stage-share speed estimates as non-evidence and never relax a stable timing gate.
- During scoring stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed or missing summary, wrong counts/topology, duplicate eval epoch, invalid transition, realized passes <145, or total >=600. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.32 at commit `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, current branch/status, `git diff --check`, full diff, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/028/preflight.py`. Require one idle H20, only tracked production change `train.py`, and no frozen-file drift.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/028/preflight.py --semantics`. Require candidate construction/state/RNG/optimizer and matched pre-boundary synthetic updates to equal the independently loaded `git show 67c8e98:train.py` oracle. Exercise the real controller for live, early, budget-exhausted, eligible, and repeated calls. After freeze require 33,424/953,674 counts, unchanged values/groups/state, prefix `grad is None`, upper finite gradients/updates, fixed frozen values/momentum over multiple steps, live BN buffers, and exactly one exhausted one-way transition.
4. Run `timeout 240s uv run python .../experiments/028/preflight.py --throughput`. Use identical saved boundary model/optimizer states, pinned host batches, complete hard-label production steps, at least 25 warmups, and balanced three-window >=50-step accepted/frozen arms. Require finite state, CV <=0.05, and projected passes `(16770 + (25978 - 16770) * accepted_ms/frozen_ms) * 256 / 50000 >=145.0`, calibrated to EXP-027's observed exhausted-boundary/final steps rather than a uniform-throughput assumption.
5. Reconfirm audit and one idle H20, remove stale log, and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Record start/PID and do not launch a second valid score.
6. Parse with `rg`; require one finite summary, `training_seconds` 300.0-300.1, `total_seconds <600`, 987,098 original parameters, `num_steps*256/50000 >=145`, and no traceback/OOM/non-finite/worker errors.
7. Require one mixup transition at/after 195 seconds; one later same-boundary RandAugment disable and prefix freeze after normal iterator exhaustion with identical epoch/step/time and 33,424/953,674 counts; no duplicate/re-enable/unfreeze path.
8. Require unique evaluation epochs at the unchanged every-fifth-epoch cadence plus one final partial epoch. Require both `best_test_acc >=94.42%` and `final_test_acc >=94.42%`; stop on either failure. Audit the final diff and record final loss relative to 0.2523 regardless of verdict.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameter count.
- Freeze behavior: `run.log` transition - epoch/step/time, mixup-to-boundary lag, and frozen/remaining parameter counts.
- Preflight timing: direct output - accepted/frozen hard-step windows, CVs, speed ratio, and projected passes.
- Mechanism: final loss delta from 0.2523 and best-final gap; lower loss/small gap supports useful temporal allocation but cannot override the accuracy gates.

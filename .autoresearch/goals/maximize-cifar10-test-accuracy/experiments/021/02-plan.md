# Plan EXP-021: Final-Ten-Percent SAM
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement bounded late SAM
- [x] Modify only `train.py`: keep exact accepted SGD below 90% counted progress; at/above 90%, perform rho-0.05 non-adaptive SAM using the same hard-label batch for both passes.
- [x] After the first backward, compute the global L2 gradient norm; under `no_grad`, clone originals and perturb each trainable parameter by `rho * grad / norm`; clear gradients, run the second loss/backward, then restore originals with `copy_` before the existing optimizer step.
- [x] Snapshot BatchNorm running mean/variance/counter after the first forward and byte-restore them after the perturbation forward, yielding exactly one persistent BN update per optimizer step; print one SAM transition line.

### Milestone 2: Pass semantic and timing preflight
- [x] Create ignored `experiments/021/preflight.py` with dummy evaluator; verified accepted behavior below 90%, exact parameter restoration, finite perturbation norm, one persistent BN update, second-pass gradients, and unchanged optimizer groups.
- [x] Measure accepted normal steps and candidate SAM steps after warmup; projected passes were 127.24 but retention 89.67% failed the 90% floor, so scoring stopped.
- [x] Compile and audit scope/diff; only planned `train.py` changes were present and evaluator/data/schedule remained unchanged.

### Milestone 3: Execute once and verify
- [x] Confirm one H20; scored execution was not launched because the throughput abort criterion failed.
- [x] Scored-run integrity conditions were not reached.
- [x] `best_test_acc` is unavailable; recorded the preflight failure without tuning rho/window or retrying.

## Code Changes
- **`train.py` constants**: add `SAM_RHO=0.05` and `SAM_START_FRACTION=0.90`.
- **`train.py` training step**: retain the accepted first forward/backward. During the final hard-label window only, derive normalized perturbations without changing optimizer state, clone original parameters and save BN buffers after the first pass, perturb under `no_grad`, run a second forward/backward, restore BN buffers and parameters with `copy_` under `no_grad`, then call accepted Nesterov SGD once using second-pass gradients.
- **`train.py` logging**: emit exactly one transition line when SAM first activates; no diagnostics or evaluator changes.

## Configuration Changes
- Optimizer geometry: accepted SGD -> rho-0.05 SAM only for final 10% counted time, then the same SGD optimizer step.
- Model, parameters, initialization, seed 42, data, batch-shared alpha-0.2 mixup through 65%, LR/floor, weight decay, momentum, batch, workers, and evaluation cadence unchanged.

## Execution Environment
- Method: offline local execution; no remote/network/package install/W&B/GitHub/`gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv`, eight persistent workers.
- Estimated runtime: preflight under 4 minutes; scored run about 342 seconds wall, hard 600-second timeout.
- Log output: project-root `run.log`, retained through analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring for non-exact `copy_` restoration, BN persistent state differing from a one-forward oracle, missing/duplicate transition, non-finite/zero grad norm, wrong optimizer groups, throughput retention <90%, projected passes <127, syntax/scope error, or OOM. The 127-pass floor is below the ideal `141.9/1.1=129.0` estimate for a 10% double-pass window.
- During scoring abort/classify on nonzero exit, timeout, OOM, error/non-finite output, missing summary, wrong parameter count, duplicate eval, or missing/multiple mixup/SAM transitions. Never adjust rho/window/BN handling or rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query index baseline; require 94.07 at `eb08811`, threshold 94.17. Timeout 10 seconds.
2. Run H20, compile, git scope/status/untracked/root-Python and diff audits; require one H20 and only planned `train.py` logic. Timeout 30 seconds.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/021/preflight.py --semantics`; require exact below-window accepted behavior, strict 0.90 boundary, one transition, finite perturbation, exact restoration, one-forward BN state, second-pass gradients, and unchanged groups. Timeout 120 seconds.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/021/preflight.py --throughput`; require warmed stable windows, projected whole-run retention >=0.90, projected passes >=127, finite state, explicit pass. Timeout 180 seconds.
5. Remove stale log and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
6. Parse via `rg`/`awk`; require complete finite summary, `best_test_acc >=94.17`, training >=300, total <600, 691,674 params, one transition near 195 seconds and one SAM transition in `[270,271)`, unique evals, no errors. Stop at first necessary failure.
7. Final diff audit confirms only approved SAM changes.

### Informational Metrics (Optional)
- If all necessary conditions pass, collect final accuracy/loss, timing, epochs, steps/passes, VRAM, and parameters from `run.log`.

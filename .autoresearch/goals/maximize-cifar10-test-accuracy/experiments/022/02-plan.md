# Plan EXP-022: Alternating Final-Ten-Percent SAM
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement deterministic alternating late SAM
- [x] Create experiment branch `autoresearch/maximize-cifar10-test-accuracy-022` from accepted commit `eb08811`; modify only `train.py`.
- [x] Add fixed `SAM_RHO=0.05` and `SAM_START_FRACTION=0.90`; put `progress >= SAM_START_FRACTION and step % 2 == 0` in a pure `should_use_sam` predicate, leaving every other step and all earlier training on the accepted path.
- [x] Reuse EXP-021's preflighted restoration semantics: after the accepted first backward, snapshot parameters and post-first-forward BatchNorm buffers, derive and apply the global normalized perturbation, explicitly clear first-pass gradients, compute pure second-pass hard-label gradients, restore parameters and BN buffers exactly, then call the existing optimizer once.
- [x] Emit one transition line using a one-shot flag only on the first even-parity step that actually invokes SAM, without per-step diagnostics.

### Milestone 2: Pass semantic and fixed-time preflight
- [x] Compile `train.py` and run a disposable experiment-scoped preflight that directly checks the pure predicate's strict window/parity behavior, finite gradient norm, exact parameter restoration, one persistent BatchNorm update, pure second-pass gradient equality against a zeroed-gradient oracle, unchanged optimizer groups, and one correctly labeled transition.
- [x] Warm and time an alternating normal/SAM sequence through the production predicate/helper, not dense SAM in isolation; measured 0.941746 retention and 133.633698 projected passes, passing both gates.
- [x] Audit the git diff and scope; only the planned `train.py` changes are present and evaluator, data, seed, schedule, and evaluation cadence remain unchanged.

### Milestone 3: Execute the sole scored run and verify
- [x] Confirm exactly one NVIDIA H20, remove stale `run.log`, and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Clean completion: finite summary, one mixup transition, one SAM transition, no duplicate epoch evaluation, 300.0 counted seconds, and 340.4 seconds wall.
- [x] Fixed-seed `best_test_acc` was 93.79% versus 94.07 baseline and failed the 94.17 threshold; no tuning or rerun performed.

## Code Changes
- **`train.py` constants**: add `SAM_RHO=0.05` and `SAM_START_FRACTION=0.90`.
- **`train.py` helpers**: add a pure `should_use_sam(progress, step)` predicate and a bounded `sam_second_backward` helper that computes the global L2 gradient norm, clones originals, snapshots BatchNorm buffers after the first forward, applies normalized perturbations under `no_grad`, clears first-pass gradients, runs the second hard-label forward/backward, and restores all saved state with `copy_` before returning the second loss.
- **`train.py` training loop**: after the existing first backward, invoke the helper only for hard-label steps satisfying `progress >=0.90 and step % 2 == 0`; call the existing Nesterov SGD optimizer exactly once per batch.
- **`train.py` logging**: print exactly one activation line from inside the first true SAM branch, guarded by a one-shot flag. Do not add runtime diagnostics that could consume the fixed budget.

## Configuration Changes
- Optimizer geometry: accepted SGD -> rho-0.05 SAM on alternating optimizer steps during only the final 10% counted time; all other optimizer steps remain accepted SGD.
- Model, initialization, fixed seed 42, data pipeline, batch-shared alpha-0.2 mixup through 65%, LR/floor, weight decay, momentum, batch size, persistent workers, counted-time accounting, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local execution; no remote services, network, package installation, W&B, GitHub, or `gh`.
- Resources: exactly one NVIDIA H20, local CIFAR-10, existing `uv`, and persistent DataLoader workers.
- Estimated runtime: semantic/timing preflight under 4 minutes; scored run about 345 seconds wall, with a hard 600-second timeout.
- Log output: scored stdout/stderr redirected to project-root `run.log`, retained through analysis only.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on syntax error, unexpected diff/scope, wrong GPU, non-finite or zero gradient norm, gradient accumulation rather than pure second-pass gradients, inexact parameter restoration, BatchNorm state differing from a one-forward oracle, missing/duplicate/mislabeled transition, changed optimizer groups, preflight OOM, measured alternating-pattern whole-run retention <0.90, or projected exposure <127.71 passes.
- During scoring, abort/classify on nonzero exit, timeout, OOM, traceback, non-finite loss, wrong parameter count, duplicate evaluation within an epoch, missing final summary, missing/multiple mixup transitions, or missing/multiple SAM transitions.
- Never adjust the fixed seed, rho, 90% start, alternating parity, feasibility floor, or acceptance threshold after observing results; never rerun a completed valid score.

## Verification Protocol

### Verification Procedure

1. From the project root, run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; within 10 seconds require `baseline=94.07`, `baseline_commit=eb08811`, and set the acceptance threshold to 94.17.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, `uv run python -m py_compile train.py`, `git status --short`, and `git diff -- train.py`; also read `TIME_BUDGET_S` through `uv run python -c 'from prepare import TIME_BUDGET_S; print(TIME_BUDGET_S)'`. Within 30 seconds require one H20, the expected 300-second budget, and only the planned `train.py` logic.
3. Run the ignored experiment-scoped semantic preflight with a 120-second timeout. Require direct pure-predicate checks below/at 90% and for odd/even steps, one correctly labeled activation transition, 691,674 parameters, finite nonzero perturbation norm, exact parameter restoration, BatchNorm state matching one persistent forward, helper gradients equal to a fresh zeroed-gradient perturbed-weight oracle, and unchanged optimizer groups.
4. Run the ignored experiment-scoped warm throughput preflight with a 180-second timeout. Require stable finite timing of the actual alternating pattern, measured final-window whole-run retention >=0.90, projected passes >=127.71 from the accepted 141.9-pass reference, and an explicit pass.
5. Remove stale `run.log`, then execute exactly `timeout 600s uv run train.py > run.log 2>&1`. Do not run a second valid score.
6. Parse `run.log` with `rg` and `awk`; require exit 0, a complete finite summary, `best_test_acc >= 94.17`, `training_seconds >= TIME_BUDGET_S`, `total_seconds <600`, 691,674 parameters, exactly one mixup transition on the first step crossing `0.65 * TIME_BUDGET_S`, exactly one SAM transition on the first even step at/after crossing `0.90 * TIME_BUDGET_S`, and no duplicate epoch evaluations or error text. Allow one batch-duration boundary granularity rather than hardcoded timestamps.
7. Run a final `git diff -- train.py` and `git status --short` audit; require only approved candidate logic. Stop at the first failed necessary condition and classify it without post-hoc tuning.

### Informational Metrics (Optional)
- `peak_vram_mb`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, and `num_params`: collect from the final `run.log` summary after the necessary conditions are checked.
- Effective data passes: calculate `num_steps * 256 / 50000` from the final summary.
- Robustness caveat: record how many post-SAM evaluations support the peak and note that the fixed-seed, single-run criterion does not establish multi-seed robustness.

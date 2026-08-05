# Plan EXP-005: Stronger Alpha-0.4 Mixup
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement the isolated alpha change
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-005` from the current integration branch.
- [x] Change only `MIXUP_ALPHA` in `train.py` from 0.2 to 0.4; preserve every other code path and constant.
- [x] Run `uv run ruff check train.py`, `uv run python -m py_compile train.py`, and `git diff --check` successfully.

### Milestone 2: Verify strength, scope, and hardware
- [x] Assert `MIXUP_ALPHA == 0.4` and sample device-resident fixed-seed Beta(0.2,0.2) and Beta(0.4,0.4) distributions on CUDA to confirm alpha 0.4 increases the fraction of lambdas in `[0.2, 0.8]` while retaining mean near 0.5.
- [x] Confirm the diff is exactly one constant-line change, only `train.py` is modified, parameter count remains 691,674, and one evaluator call site remains.
- [x] Confirm one NVIDIA H20 with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.

### Milestone 3: Execute and monitor
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor bounded log extracts for traceback, CUDA/OOM, non-finite loss, progress, the unchanged 65% transition, and final completion.

### Milestone 4: Verify and record
- [x] Query the 94.07% baseline and require `best_test_acc >= 94.17%` (checked; actual 93.57% failed).
- [x] Verify 300 counted seconds, at most 600 total seconds, one-H20 execution, allowed evaluation cadence, and the exact one-line scope.
- [x] Record final metrics, transition, exposure, and final test loss, then remove `run.log` after analysis.

## Code Changes

- **`train.py`**: Change `MIXUP_ALPHA = 0.2` to `MIXUP_ALPHA = 0.4`. This makes the symmetric beta distribution less endpoint-heavy so materially mixed batches occur more often. The 65% cutoff, WRN, optimizer, schedule, seed value, loader, evaluator, logging, and output schema remain unchanged. Beta rejection sampling consumes alpha-dependent CUDA RNG draws, so later permutations diverge from EXP-002 despite the fixed seed; the result measures the alpha-defined stochastic training process, not a bit-identical pairing trajectory.

## Configuration Changes

- `MIXUP_ALPHA`: `0.2` -> `0.4` (increase interpolation strength while retaining the validated duration and clean tail)
- `MIXUP_END_FRACTION`: unchanged at `0.65`; all architecture, optimizer, schedule, data, and evaluation settings remain at the EXP-002 baseline.

## Execution Environment

- Method: local single-process run with `timeout 600s uv run train.py > run.log 2>&1`
- Resources: one NVIDIA H20 with 97,871 MiB and the existing local CIFAR-10 cache; no network, dependency, remote, or GitHub operation
- Estimated runtime: about 300 seconds counted and 340-370 seconds total
- Log output: capture all stdout/stderr in `run.log`, inspect bounded extracts, and remove after analysis
- Tool skill: none; fully local

## Abort Criteria

- Abort before the run if the diff is not the single alpha constant, lint/compile/distribution checks fail, hardware is not one H20, or stale output cannot be removed.
- Abort on traceback, CUDA/OOM, non-finite loss, or no progress for two minutes.
- Require exactly one `Mixup disabled` message between 64.5% and 65.5%. A structural failure invalidates the single-shot run; weak accuracy never authorizes a retry.
- `timeout 600s` is authoritative. Do not abort for weak intermediate accuracy because the full hard-label tail is required.
- Pre-register interpretation: this is a two-sided exploratory probe, not a presumed directional win. At least 94.17% passes the user's rule, with 94.17-94.21% labeled low-confidence and noise-sensitive. A result below 94.07% with normal exposure indicates the alpha-0.4 stochastic process over-regularizes; 94.07-94.16% is near-flat. Final test loss below EXP-002's 0.2432 supports the intended generalization mechanism but is informational, not a substitute for the primary threshold. Analysis must note that alpha-dependent RNG consumption changes mixup pairings even though the seed value is fixed.

## Verification Protocol

### Verification Procedure

1. Query the index baseline and require 94.07%, making 94.17% the success threshold; confirm the frozen `TIME_BUDGET_S` is 300.
2. Confirm one H20, a clean `git diff --check`, and a `git diff -- train.py` containing only `0.2 -> 0.4` for `MIXUP_ALPHA`.
3. Remove `run.log`, execute `timeout 600s uv run train.py > run.log 2>&1`, and require exit code 0.
4. Require a complete final summary with `rg '^(best_test_acc|training_seconds|total_seconds|peak_vram_mb):' run.log`; otherwise classify as crash and inspect the last 50 lines.
5. Require exactly one 64.5-65.5% transition and confirm source inspection bypasses mixup afterward.
6. Require `300.0 <= training_seconds <= 305.0` (budget plus at most one anomalously slow final step), `total_seconds <= 600.0`, and unique every-fifth-plus-final evaluation epochs.
7. Require `best_test_acc >= 94.17%`; stop on failure without result-conditioned repetition.
8. Confirm the one-line final diff, 691,674 parameters, and no dependency, seed, or evaluator change.

### Informational Metrics (Optional)

- `peak_vram_mb`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, and `num_params`: final summary in `run.log`
- realized passes: `num_steps * 256 / 50000`
- transition: `rg 'Mixup disabled' run.log`

# Plan EXP-004: Earlier 50% Mixup Cutoff
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement the isolated cutoff change
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-004` from the current integration branch.
- [x] Change only `MIXUP_END_FRACTION` in `train.py` from 0.65 to 0.50; preserve every other code path and constant.
- [x] Run `uv run ruff check train.py`, `uv run python -m py_compile train.py`, and `git diff --check` successfully.

### Milestone 2: Verify configuration and scope
- [x] Run `uv run python -c 'import train; assert train.MIXUP_END_FRACTION == 0.5; print(train.learning_rate(150.0))'` and record the LR at the planned transition.
- [x] Confirm the diff is exactly one constant-line change, `git status --short` shows only `train.py`, parameter count remains 691,674 by source identity, and one evaluator call site remains.
- [x] Confirm one NVIDIA H20 with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.

### Milestone 3: Execute and monitor
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor bounded log extracts for traceback, CUDA/OOM, non-finite loss, progress, the 50% transition, and final completion.

### Milestone 4: Verify and record
- [x] Query the 94.07% baseline and require `best_test_acc >= 94.17%` (checked; actual 93.91% failed the condition).
- [x] Verify 300 counted training seconds, at most 600 total seconds, one-H20 execution, allowed evaluation cadence, and `train.py`-only scope.
- [x] Record final metrics and the transition, then remove `run.log` after analysis captures the evidence.

## Code Changes

- **`train.py`**: Change `MIXUP_END_FRACTION = 0.65` to `MIXUP_END_FRACTION = 0.50`. This ends alpha-0.2 mixup at 150 counted seconds and gives the unchanged hard-label cosine path 150 seconds rather than 105. No helper, RNG, architecture, optimizer, loader, seed, logging, evaluator, or output-schema changes are allowed.

## Configuration Changes

- `MIXUP_END_FRACTION`: `0.65` -> `0.50` (test whether the early critical-period benefit is retained while allocating 45 additional seconds to hard-label margin refinement)
- WRN-16-2, batch 256, alpha 0.2, LR 0.2 to 0.002, 5% warmup, Nesterov, selective `5e-4` decay, seed 42, crop/flip, persistent workers, and every-fifth-plus-final evaluation: unchanged.

## Execution Environment

- Method: local single-process run from the project root with `timeout 600s uv run train.py > run.log 2>&1`
- Resources: one NVIDIA H20 with 97,871 MiB; existing local CIFAR-10 cache; no network, dependency, remote job, or GitHub operation
- Estimated runtime: about 300 seconds counted training and 340-370 seconds total
- Log output: redirect all stdout/stderr to `run.log`, inspect bounded `rg`/`tail` extracts, and remove it after analysis
- Tool skill: none; fully local execution

## Abort Criteria

- Abort before the full run if the diff contains anything beyond the single planned constant change, lint/compile fails, the cutoff assertion fails, hardware is not one H20, or stale `run.log` cannot be removed.
- Abort the full run on traceback, CUDA/OOM, non-finite loss, or no log progress for two minutes.
- Require exactly one `Mixup disabled` record between 49.5% and 50.5% counted time. A missing, repeated, or out-of-window record makes the run invalid; this one-line experiment has no retry path.
- `timeout 600s` is authoritative; exit 124 is a failed run. Do not abort for weak intermediate accuracy because the entire 150-second hard-label tail is the intervention.
- Classify invalid if code beyond the one `train.py` constant changes, seed/evaluator behavior changes, or evaluation occurs more than once per epoch.
- Pre-register interpretation without rerunning: at least 94.17% passes the user's rule; 94.17-94.21% must be reported as noise-sensitive. A 94.07-94.16% result is near-flat `no-improvement`; below 94.07% with normal exposure suggests the earlier switch under-regularizes and routes a later loop toward the opposite 75% cutoff. Because the hard-label branch is slightly cheaper, report its realized pass delta against EXP-002's 141.9 and acknowledge exposure as a co-varying factor; exposure is not an extra user-defined pass/fail condition.

## Verification Protocol

### Verification Procedure

1. Query `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=94.07`, making 94.17% the minimum successful result. Query `uv run python -c 'from prepare import TIME_BUDGET_S; print(TIME_BUDGET_S)'` and require 300.
2. Confirm one H20 via `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`. Require `git diff --check` to pass and `git diff -- train.py` to show only `0.65 -> 0.50`.
3. Remove stale output with `rm -f run.log`, then run `timeout 600s uv run train.py > run.log 2>&1`. Require exit code 0; exit 124 is a timeout failure.
4. Require a complete summary using `rg '^(best_test_acc|training_seconds|total_seconds|peak_vram_mb):' run.log`; if absent, classify as crash and inspect `tail -n 50 run.log`.
5. Require exactly one `Mixup disabled` record between 49.5% and 50.5%, then confirm source inspection bypasses mixup afterward.
6. Require `300.0 <= training_seconds <= 301.0` and `total_seconds <= 600.0`. Parse `eval ep` records to confirm each epoch appears once at most and only every fifth epoch plus the final partial epoch is evaluated.
7. Parse `best_test_acc` and require at least 94.17%; stop verification on failure and do not reroll or repeat based on the metric.
8. Confirm the final diff remains the single cutoff constant, parameter count is 691,674, and no dependency, seed, or evaluator change occurred.

### Informational Metrics (Optional)

- `peak_vram_mb`: `rg '^peak_vram_mb:' run.log`
- `final_test_acc`: `rg '^final_test_acc:' run.log`
- `final_test_loss`: `rg '^final_test_loss:' run.log`
- `training_seconds`: `rg '^training_seconds:' run.log`
- `total_seconds`: `rg '^total_seconds:' run.log`
- `num_epochs`: `rg '^num_epochs:' run.log`
- `num_steps`: `rg '^num_steps:' run.log`; derive passes as `num_steps * 256 / 50000`
- `num_params`: `rg '^num_params:' run.log`
- transition: `rg 'Mixup disabled' run.log`

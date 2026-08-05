# Plan EXP-003: Early CutMix With a Hard-Label Tail
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement isolated area-corrected CutMix
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-003` from the current integration branch and modify only `train.py`.
- [x] Replace the early mixup helper and loss path with one shared CutMix rectangle, an area-corrected mixed loss, and a 65% time gate; preserve all other accepted settings.
- [x] Log the single CutMix-to-hard-label transition with epoch, step, counted time, LR, and cumulative mean pasted-area fraction.
- [x] Run `uv run ruff check train.py` and `git diff --check` successfully.

### Milestone 2: Verify pixel semantics and throughput
- [x] Run a deterministic synthetic pixel test that checks source/donor regions, alias safety, exact area correction, target permutation, zero-area fallback, finite loss, and backward/optimizer success.
- [x] Benchmark alternating mixup/CutMix/CutMix/mixup blocks in one CUDA process, each with 20 warmup and 100 timed steps; compare the mean of each path's two blocks and require CutMix to project at least 95% of matched mixup exposure.
- [x] Confirm `git status --short` shows only `train.py`, parameter count remains 691,674, and the source still has one evaluator call site.

### Milestone 3: Execute the full experiment
- [x] Confirm exactly one NVIDIA H20 with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor bounded log extracts for traceback, CUDA/OOM, non-finite loss, progress, the 65% transition, and final completion.

### Milestone 4: Verify and record
- [x] Query the moving 94.07% baseline and require `best_test_acc >= 94.17%` (checked; actual 93.72% failed the condition).
- [x] Verify approximately 300 counted training seconds, at most 600 total seconds, one-H20 execution, allowed evaluation cadence, and `train.py`-only scope.
- [x] Record all final metrics and CutMix diagnostics, then remove `run.log` after analysis captures the evidence.

## Code Changes

- **`train.py`**: Replace `MIXUP_ALPHA` with `CUTMIX_ALPHA = 1.0` and rename the unchanged 0.65 phase boundary to `REGULARIZATION_END_FRACTION` so the active mechanism is explicit.
- **`train.py`**: Replace `mixup_batch` with a `cutmix_batch` helper. Draw a uniform retained-area coefficient, exactly equivalent to `Beta(1,1)`, plus rectangle-center coordinates from a dedicated CPU `torch.Generator` seeded from the existing `torch.initial_seed()`. Keep the permutation device-local. The isolated CPU generator avoids per-batch CUDA scalar synchronization and avoids introducing CutMix draws into the CPU RNG stream used by the loader; CutMix necessarily changes the CUDA RNG trajectory relative to mixup and no bit-identical trajectory is claimed.
- **`train.py`**: Compute a shared 32x32 rectangle from `sqrt(1 - lambda)`, using `int(width * ratio)` and `int(height * ratio)` floor conversion before clipping around an integer center. Clone the destination tensor and paste the materialized permuted donor patch. Return original/permuted targets and `lambda_effective = 1 - pasted_area / image_area`; a zero-area rectangle returns the unchanged inputs and hard-label coefficient 1.0.
- **`train.py`**: During the first 65% of prior-completed counted training time, use one forward pass and area-corrected mixed cross entropy. At and after 65%, retain the exact EXP-002 hard-label branch. Keep the existing finite-loss guard over both paths.
- **`train.py`**: Accumulate pasted-area fraction without device synchronization and emit exactly one `CutMix disabled` transition log with the realized mean. Architecture, initialization, seed, optimizer, LR schedule, crop/flip, persistent loader, evaluation cadence, and final summary schema remain unchanged.

## Configuration Changes

- `MIXUP_ALPHA`: `0.2` -> removed (the experiment replaces, rather than stacks, the early regularizer)
- `CUTMIX_ALPHA`: absent -> `1.0` (`Beta(1,1)` is uniform and avoids near-all-or-nothing patches dominating the first test)
- `MIXUP_END_FRACTION`: `0.65` -> `REGULARIZATION_END_FRACTION = 0.65` (semantic rename only; preserves the validated 195-second transition and 105-second clean tail)
- WRN-16-2, batch 256, LR 0.2 to 0.002, 5% warmup, Nesterov, selective `5e-4` decay, seed 42, crop/flip, and every-fifth-plus-final evaluation: unchanged.

## Execution Environment

- Method: local single-process execution from the project root with `timeout 600s uv run train.py > run.log 2>&1`
- Resources: one NVIDIA H20 with 97,871 MiB; existing local CIFAR-10 cache; no network, new dependency, remote job, or GitHub operation
- Estimated runtime: about 300 seconds counted training and 340-370 seconds total; all CutMix tensor work is charged to the training timer
- Log output: redirect stdout/stderr to `run.log`; inspect only bounded `rg` and `tail` extracts during execution; remove it after analysis
- Tool skill: none; execution is fully local

## Abort Criteria

- Abort smoke testing on incorrect source/donor pixels, donor aliasing, area/label coefficient mismatch, invalid shapes, non-finite loss, failed backward, CUDA error, or OOM.
- Abort before the full run if the mean of the two alternating CutMix timing blocks projects below 95% of the mean of the two mixup blocks. This matched synthetic ratio is the sole pre-run throughput gate; the historical 141.9-pass result is diagnostic only. Do not relax the threshold after measurement.
- Abort the full run on traceback, CUDA/OOM, non-finite loss, or no log progress for two minutes.
- Require exactly one `CutMix disabled` message between 63% and 68% counted time. A missing or repeated transition permits one implementation fix/retry only if detected before inspecting any accuracy result; a weak metric never authorizes a retry.
- `timeout 600s` is authoritative; exit 124 is a failed run. Do not abort for weak intermediate accuracy because the 105-second hard-label tail is part of the hypothesis.
- Classify invalid if any experiment code outside `train.py` changes, seed/evaluator behavior changes, or evaluation occurs more than once per epoch.
- Pre-register interpretation without rerunning: 94.07-94.16% with normal exposure is a `no-improvement` and returns to the accepted mixup baseline; a regression with normal exposure implicates CutMix's spatial label proxy; below 134.9 realized passes identifies unexpected implementation overhead but does not change the metric verdict. A 94.17-94.21% pass follows the user's rule but must be reported as noise-sensitive rather than strong evidence of CutMix superiority.

## Verification Protocol

### Verification Procedure

1. Query `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=94.07`, so the minimum accepted result is 94.17%. Query the frozen budget with `uv run python -c 'from prepare import TIME_BUDGET_S; print(TIME_BUDGET_S)'` and require 300 seconds before execution.
2. Confirm one `NVIDIA H20` with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`. Confirm `git status --short` reports only `train.py` after implementation and `git diff --check` passes.
3. Remove stale output with `rm -f run.log`, then run `timeout 600s uv run train.py > run.log 2>&1`. Require exit code 0; exit 124 is a hard timeout failure.
4. Require a complete result using `rg '^(best_test_acc|training_seconds|total_seconds|peak_vram_mb):' run.log`. If the summary is absent, classify as a crash and inspect `tail -n 50 run.log`.
5. Require exactly one `CutMix disabled` record between 63% and 68% counted time and a finite mean pasted-area fraction strictly between 0 and 1. Source inspection must confirm CutMix is bypassed afterward.
6. Require `300.0 <= training_seconds <= 301.0` and `total_seconds <= 600.0`. Parse `eval ep` records and confirm each epoch appears at most once and evaluations occur only every fifth epoch plus the final budget-truncated epoch.
7. Parse `best_test_acc` and require `best_test_acc >= 94.17%`; stop verification on the first failed necessary condition. Do not reroll the seed or run a result-conditioned repeat.
8. Confirm one-H20 use, unchanged 691,674 parameter count, no dependency changes, and a final diff containing only the planned `train.py` intervention.

### Informational Metrics (Optional)

- `peak_vram_mb`: `rg '^peak_vram_mb:' run.log`
- `final_test_acc`: `rg '^final_test_acc:' run.log`
- `final_test_loss`: `rg '^final_test_loss:' run.log`
- `training_seconds`: `rg '^training_seconds:' run.log`
- `total_seconds`: `rg '^total_seconds:' run.log`
- `num_epochs`: `rg '^num_epochs:' run.log`
- `num_steps`: `rg '^num_steps:' run.log`; derive realized passes as `num_steps * 256 / 50000`
- `num_params`: `rg '^num_params:' run.log`
- CutMix switch and area: `rg 'CutMix disabled' run.log`

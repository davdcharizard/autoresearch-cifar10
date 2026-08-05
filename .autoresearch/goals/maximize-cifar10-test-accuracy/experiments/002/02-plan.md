# Plan EXP-002: Early Mixup With a Hard-Label Tail
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement isolated time-gated mixup
- [x] Modify only `train.py` to add GPU-side batchwise mixup with alpha 0.2 before 65% counted training time.
- [x] Preserve all accepted EXP-001 architecture, optimizer, schedule, seed, loader, and evaluation settings.
- [x] Log the single transition from mixup to hard-label training with epoch, step, counted seconds, and LR.
- [x] Run `uv run ruff check train.py` and correct all reported issues.

### Milestone 2: Verify correctness and throughput
- [x] Run a CUDA smoke test that verifies mixed inputs and both target permutations have correct shapes, lambda lies in `[0, 1]`, loss is finite, and backward/optimizer step succeeds.
- [x] Warm up 20 steps, time 100 mixup training steps, and require at least 95% of EXP-001's matched synthetic throughput (116.1 projected passes from a 122.2 baseline); record actual full-run exposure separately.
- [x] Confirm `git status --short` reports only `train.py` modified and `git diff --check` passes.

### Milestone 3: Execute the experiment
- [x] Confirm one NVIDIA H20 with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor bounded log extracts for errors, non-finite loss, progress, the 65% phase transition, and final completion.

### Milestone 4: Verify and record
- [x] Query the moving 93.38% baseline and require `best_test_acc >= 93.48%`.
- [x] Verify 300-second training, sub-600-second total runtime, allowed evaluation cadence, and `train.py`-only scope.
- [x] Record final metrics and remove `run.log` after analysis captures the evidence.

## Code Changes
- **`train.py`**: Add `MIXUP_ALPHA = 0.2` and `MIXUP_END_FRACTION = 0.65`; no existing EXP-001 hyperparameter changes.
- **`train.py`**: Add a small `mixup_batch` helper that samples one lambda from a device-resident beta distribution, creates one device-local permutation, and returns mixed inputs plus original/permuted targets and lambda. Sampling remains deterministic under the existing fixed CPU/CUDA seed.
- **`train.py`**: Before the forward pass, compute progress from the same prior-completed `total_training_time` used by the LR schedule. While progress is below 0.65, perform one forward pass on mixed images and compute `lambda * CE(outputs, targets) + (1-lambda) * CE(outputs, permuted_targets)`. At and after 0.65, use the existing unmodified hard-label cross entropy.
- **`train.py`**: Apply the existing `torch.isfinite(loss)` runtime guard after both the mixed and hard-label loss branches, so the full 65% mixup phase remains protected.
- **`train.py`**: Emit exactly one newline-delimited `Mixup disabled` transition message on the first hard-label batch. Retain progress logs and the final output schema unchanged.

## Configuration Changes
- `MIXUP_ALPHA`: absent -> 0.2 (mild, literature-backed convex interpolation)
- `MIXUP_END_FRACTION`: absent -> 0.65 (regularize the high/medium-LR critical period and reserve 35%, about 105 seconds, for hard-label convergence)
- Architecture, batch 256, peak/floor LR, 5% warmup, cosine schedule, selective decay, Nesterov, crop/flip, seed 42, persistent workers, and every-fifth-plus-final evaluation: unchanged from EXP-001.

## Execution Environment
- Method: local single-process run from the project root with `timeout 600s uv run train.py > run.log 2>&1`
- Resources: one NVIDIA H20 with 97,871 MiB; existing local CIFAR-10 cache and persistent 8-worker loader
- Estimated runtime: about 300 seconds counted training and 340-370 seconds total; mixup tensor work is charged to the training budget
- Log output: all stdout/stderr redirected to `run.log`; inspect only bounded `rg`/`tail` extracts
- Tool skill: none; fully local and offline

## Abort Criteria
- Abort smoke testing on invalid lambda/shape, CUDA error, non-finite loss, or failed backward/optimizer step.
- Abort before the full run if the timed mixup-only smoke test projects fewer than 116.1 dataset-equivalent passes, 95% of EXP-001's matched 122.2-pass synthetic result. The absolute synthetic projection underestimates the 146-pass real run, so relative throughput is the valid gate; realized exposure is judged from final steps.
- Abort the full run on traceback, CUDA error, non-finite loss, or no progress for 2 minutes.
- Require the `Mixup disabled` message near 65% counted time; a missing transition by 68% is a code error and grounds one fix/retry.
- `timeout 600s` enforces the hard total limit; exit 124 is failure.
- Do not abort for weak intermediate test accuracy; the 105-second hard-label tail must finish before judging the research result.
- Classify invalid if any experiment code outside `train.py` changes, seed/evaluator changes, or evaluation occurs more than once per epoch.
- Pre-register interpretation without rerunning: 93.28-93.47% with at least 140 realized passes is a `no-improvement` that routes the next loop to a 50% mixup cutoff; below 140 passes leaves the mechanism unproven due throughput. A 93.48-93.57% pass follows the user-defined rule but must be flagged as noise-sensitive in analysis; do not reroll the seed.

## Verification Protocol

### Verification Procedure
1. Query `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=93.38`, so the minimum successful result is 93.48%.
2. Confirm a single `NVIDIA H20` with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
3. Run `rm -f run.log`, then `timeout 600s uv run train.py > run.log 2>&1`. Require exit code 0; exit 124 is a hard timeout failure.
4. Require a complete summary using `rg '^(best_test_acc|training_seconds|total_seconds|peak_vram_mb):' run.log`. If absent, classify as crash and inspect `tail -n 50 run.log`.
5. Require exactly one `Mixup disabled` message between 63% and 68% counted training, demonstrating the intended early-only intervention and long hard-label tail.
6. Require `training_seconds` approximately 300 and `total_seconds <= 600.0`. Confirm evaluation log entries occur only on epochs divisible by five plus the final epoch and never more than once per epoch.
7. Parse `best_test_acc` and require `best_test_acc >= 93.48%`; stop verification immediately if the threshold fails.
8. Run `git status --short` and `git diff --check`; only `train.py` may contain experiment changes.

### Informational Metrics (Optional)
- `peak_vram_mb`: `rg '^peak_vram_mb:' run.log`
- `final_test_acc`: `rg '^final_test_acc:' run.log`
- `final_test_loss`: `rg '^final_test_loss:' run.log`
- `training_seconds`: `rg '^training_seconds:' run.log`
- `total_seconds`: `rg '^total_seconds:' run.log`
- `num_epochs`: `rg '^num_epochs:' run.log`
- `num_steps`: `rg '^num_steps:' run.log`
- `num_params`: `rg '^num_params:' run.log`
- Mixup switch point: `rg 'Mixup disabled' run.log`

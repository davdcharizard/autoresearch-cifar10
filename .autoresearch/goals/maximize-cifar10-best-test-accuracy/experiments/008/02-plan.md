# Plan EXP-008: Width-2 Weight Decay 5e-4
- **Created**: 2026-08-05

## Baseline and Hypothesis

- Moving baseline: `best_test_acc = 93.55%` at commit `8faf0f3`; formal improvement requires at least 93.65%.
- Intervention: change only coupled SGD `WEIGHT_DECAY` from `1e-4` to `5e-4` on the complete accepted width-2 recipe.
- Hypothesis: stronger plateau norm pressure will carry a better-regularized width-2 representation into the unchanged weak tail, preserve approximately EXP-007's compute exposure, and raise `best_test_acc` to at least 93.65%.
- Evidence boundary: `5e-4` is a canonical CIFAR WRN point, not a fitted optimum for this 71-epoch post-activation run. The experiment tests this exact point once. A null or regression does not rule out lower decay.

## Milestones

### Milestone 1: One-literal implementation
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-008` from integration commit `8faf0f3`; preserve the untracked `data/` cache.
- [x] Change exactly `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4` in `train.py`; make no other behavioral or logging edit.
- [x] Pass compilation, Ruff, pre-commit, diff/scope checks, width-2 shape/parameter assertions, and optimizer-value assertions.

### Milestone 2: Single fixed-seed run
- [x] Confirm baseline 93.55%, exactly one idle H20, no stale `run*.log`, and a one-line tracked diff.
- [x] Launch exactly once under the 600-second supervisor with all output redirected to `run.log`; record PID and timestamps, with no seed/decay reroll.
- [x] Monitor concise tails for finite progress, one 80% worker transition, resource/lifecycle failures, and the strong/weak trajectory without streaming the log.

### Milestone 3: Metric, integrity, and mechanism verification
- [x] Require exit zero, a complete ten-field numeric summary, 300 counted seconds, total below 600 seconds, 1,073,962 parameters, and at most one evaluation per epoch.
- [x] Require one `randaugment->base` switch at 80.0-80.2% and eight stopped workers; record step retention as an attribution diagnostic rather than a goal-validity gate.
- [x] Accept only `best_test_acc >= 93.65%`; persist the strong checkpoint/loss, first weak checkpoint, tail train/test-loss evidence, test-loss minimum/final gap, best/final trajectory, and comparison with EXP-007 regardless of verdict.

## Code Changes

- **`train.py` only**:
  ```diff
  -WEIGHT_DECAY = 1e-4
  +WEIGHT_DECAY = 5e-4
  ```

The existing single `optim.SGD` parameter group consumes this constant. Do not add parameter groups or exempt BatchNorm affine values/classifier bias; this test is the exact canonical coupled-decay point across the existing parameter set. Do not add parameter-norm logging, because extra per-step reductions would alter the fixed-time intervention.

Every other tracked line must match accepted commit `8faf0f3`: width multiplier 2, post-activation blocks, raw Option-A shortcuts, initialization, N1/M7 through 80%, weak hard-label tail, batch 128, LR schedule, momentum 0.9, transforms, workers, seed, evaluation, timing, and summary.

## Configuration Changes

- Coupled SGD weight decay: `1e-4 -> 5e-4`.
- Expected steps: approximately 26,872-27,414 (within +/-1% of EXP-007's 27,143).
- Expected epochs/VRAM/timing: approximately 70-72 epochs, 598.7 MB, 300 counted seconds, and 325-345 total seconds.

The decay force is multiplied by LR and accumulated through momentum, so most new norm pressure acts during the 80% `lr=0.1` plateau. The hypothesis is that this lower-norm state carries into the weak tail; it does not claim that decay directly targets only late weak-phase overfit.

## Execution Environment

- Method: local command `timeout --signal=TERM --kill-after=10s 600s uv run train.py > run.log 2>&1` from the project root.
- Resources: exactly one idle NVIDIA H20 near 97,871 MiB; cached CIFAR-10 in preserved `data/`; no dependency/environment changes.
- Estimated runtime: 300 counted seconds, approximately 325-345 total, hard limit 600 seconds.
- Log output: all full-run output only in `run.log`; concise tails/pattern checks during monitoring. Keep through analysis, then delete before the next experiment.
- Tool skill: local execution only.

## Preflight Procedure

1. Confirm branch/status and inspect `git diff -- train.py`; require exactly the one decay literal and no tracked path besides `train.py`.
2. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run pre-commit run --files train.py`, and `git diff --check -- train.py`, each under 120 seconds.
3. In a disposable CPU check, instantiate the width-2 model and optimizer exactly as `main` does. Require 1,073,962 parameters, output `(2,10)`, one optimizer group, `lr == 0.1`, `momentum == 0.9`, and `weight_decay == 5e-4`.
4. Do not run a GPU step benchmark: the tensor graph, optimizer kernel path, shapes, and operation count are unchanged. Post-run step retention is the stronger equivalence check.

## Abort Criteria

- Abort before launch for a baseline other than 93.55 at `8faf0f3`, wrong/busy GPU, stale log conflict, any tracked diff beyond the literal, static/API/shape/parameter failure, or off-scope modification.
- During the run, terminate on the 600-second supervisor, non-zero process failure, CUDA/OOM/DataLoader error, non-finite loss/metric, 120 seconds without progress, or worker lifecycle failure.
- Do not abort on a depressed intermediate checkpoint or high train loss; those are pre-registered underfit diagnostics and the valid run must finish.
- Once launched, do not retry for low accuracy, fewer steps, altered trajectory, or a marginal result. Do not adapt to `2e-4`/`3e-4` inside EXP-008.

## Verification Protocol

### Verification Procedure

1. Query the moving baseline using `exp-index.sh baseline`; require 93.55 at `8faf0f3` immediately before launch.
2. Query `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits`; require exactly one idle H20 near 97,871 MiB.
3. Confirm no `run*.log`, execute the supervised command once, and require exit 0 before parsing.
4. Extract all summary fields with:
   ```bash
   grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   ```
   Require all ten named fields to appear exactly once with finite values, `300.0 <= training_seconds < 310.0`, `total_seconds < 600`, and `num_params = 1073962`. Values may legitimately equal one another, such as best and final accuracy.
5. Inspect `grep -E '^augmentation_switch:|eval ep' run.log`; require exactly one switch at 80.0-80.2%, eight workers stopped, unique evaluation epochs, at most one evaluation per epoch, and terminal evaluation matching `num_epochs`.
6. Record step retention against 27,143. Because the literal does not change the operation graph, a material difference is node-timing variation rather than an intervention cost; it is informational and never vetoes an otherwise goal-valid result.
7. Compare `best_test_acc` with 93.55. `>=93.65` is improvement; lower is no-improvement. No rerun is permitted.
8. Persist before log deletion: nearest train-loss EMA and test checkpoint before the switch, first weak checkpoint, tail evaluation accuracy/loss series, nearest train-loss EMA around each tail evaluation when extractable, minimum/final test loss, best epoch, best/final gap, last-three slope, timing, steps, epochs, VRAM, and parameter count.

### Informational Metrics

- Final summary metrics from `run.log`: final accuracy/loss, training/total/startup seconds, VRAM, epochs, steps, parameters.
- Norm-control diagnostics: strong checkpoint and train-loss EMA versus EXP-007's 90.08%/0.2283; tail train-loss level; minimum/final test loss and gap; first weak accuracy; best/final trajectory.
- Compute equivalence: step retention against 27,143 and mean counted step time.

## Decision and Follow-Up Rules

- **Improvement**: `best_test_acc >= 93.65%` and all goal/integrity checks pass. Commit only the literal and promote `5e-4` on width 2. A 93.65-93.70% pass is formally valid but must be described as within the approximately 0.19-point late-tail variation seen in EXP-007 unless the broader trajectory corroborates it.
- **No improvement with underfit signature**: if strong checkpoint drops materially below 90.08% and tail train loss stays elevated while exposure is equivalent, reject `5e-4` as too strong and route only a future reviewed lower-decay point (`2e-4` or `3e-4`).
- **No improvement without underfit**: if strong fit remains healthy but test trajectory does not improve, reject this point. A null cannot distinguish an incorrect `5e-4` magnitude, weak plateau-to-tail norm coupling, and an incorrect overfit premise; do not claim the decay axis is exhausted.
- **Step variation**: report actual retention and mean step time. It affects how much optimization occurred but is not caused by the scalar and does not change the fixed-time verdict; do not rerun for a better clock state.
- **Crash/invalid**: fix only a mechanical literal/API mistake within retry limits; never alter decay, seed, schedule, or model in a retry.

## Adversarial Review Refinements

The mandatory external Claude plan review completed successfully with exit code 0 and is preserved in `02-plan-review.md`; no fallback reviewer was used. The plan adopts its execution-soundness concerns: step count is now an informational attribution diagnostic rather than a hard improvement veto, summary integrity requires ten named fields exactly once rather than ten distinct numeric values, a marginal formal pass is labeled against known intra-tail variation, and a null is explicitly three-way ambiguous between magnitude, coupling, and premise. The user-defined primary metric and +0.10 threshold remain the authoritative verdict rule.

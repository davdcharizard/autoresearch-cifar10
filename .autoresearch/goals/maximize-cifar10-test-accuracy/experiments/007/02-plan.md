# Plan EXP-007: Disable Weight Decay for the Hard-Label Tail
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement the isolated late-decay switch
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-007` from accepted commit `eb08811` on the integration branch.
- [x] Name the decayed and no-decay optimizer groups, verify their initial values, and set only the decayed group's `weight_decay` from `5e-4` to `0.0` in the existing one-shot 65% mixup transition.
- [x] Log the decayed matrix-parameter L2 norm at the transition and in the final summary without changing parameters, RNG, evaluator behavior, or the accepted training path before 65%.
- [x] Run `uv run ruff check train.py`, `uv run python -m py_compile train.py`, and `git diff --check` successfully.

### Milestone 2: Verify semantics, scope, and environment
- [x] In a stubbed-`prepare` process, construct the real model/optimizer grouping, invoke the same production switch helper used by training, and assert exactly one group starts at `5e-4`, the BN/bias group starts at zero, and the helper changes only the first live value to zero; retain 691,674 parameters.
- [x] Evaluate the norm helper on fixed CUDA tensors, require a finite correct scalar and unchanged parameter values/RNG state, and confirm the diff adds no evaluator call or stochastic branch.
- [x] Confirm the diff modifies only `train.py` and the device is one NVIDIA H20.

### Milestone 3: Execute and monitor the single scored run
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1` exactly once.
- [x] Monitor bounded extracts for traceback, CUDA/OOM, non-finite loss, progress, the one 65% mixup/decay transition, and final completion.

### Milestone 4: Verify and record
- [x] Query the current 94.07% baseline and require `best_test_acc >= 94.17%` with no result-conditioned retry (checked; actual 93.74% failed).
- [x] Verify 300 counted seconds, no more than 600 total seconds, one-H20 execution, at most one evaluation per epoch, unchanged parameter count, exact scope, and whether exposure reaches 26,329 steps / 134.8 passes.
- [x] Record final accuracy/loss, exposure, transition, norms, VRAM, and mechanism interpretation, then remove `run.log` after analysis.

## Code Changes

- **`train.py`**: Add a production `disable_weight_decay(optimizer)` helper that requires exactly two live optimizer groups, validates their current values as `[WEIGHT_DECAY, 0.0]`, mutates only the first live group to zero, and returns the actual old/new/no-decay values read from those groups. The existing one-time mixup-disable branch invokes this helper, computes the matrix-parameter L2 norm for descriptive logging, and prints the returned live values. Add a deterministic `parameter_l2_norm(parameters)` helper. Compute the final decayed-parameter norm before capturing `t_end` and peak memory, then print it in the summary so its synchronization and allocation are included in runtime/VRAM accounting.

The switch occurs before the first hard-label forward pass at `progress >= MIXUP_END_FRACTION`, matching the accepted mixup boundary. The transition norm reduction and `.item()` synchronization happen once inside the timed step and are charged to the fixed training budget; the final reduction is included in total runtime and peak-memory snapshots. Norms are descriptive telemetry only because EXP-002 has no comparable norm measurements; they cannot distinguish causal norm growth from ordinary hard-label SGD. No alternate cutoff, decay value, evaluator path, seed, loss, schedule, model, or data behavior changes.

## Configuration Changes

- Matrix-parameter `weight_decay`: `5e-4` for counted progress `[0%, 65%)`, then `0.0` for `[65%, 100%]`.
- BN/bias `weight_decay`: unchanged at `0.0` for the entire run.
- All architecture, batch, LR schedule, mixup alpha/cutoff, transforms, seed, evaluator cadence, and budget values remain at the EXP-002 baseline.

## Execution Environment

- Method: local single-process execution with `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20 with the existing local CIFAR-10 cache; no network, remote service, GitHub operation, dependency installation, or additional scored run.
- Estimated runtime: about 300 counted training seconds and 340-370 total seconds; the one-time norm reduction is expected to be negligible.
- Log output: capture stdout/stderr in `run.log`, inspect bounded `rg`/`tail` extracts, and remove the log after analysis.
- Tool skill: none; fully local.

## Abort Criteria

- Abort before the scored run if lint/compile/diff/semantic checks fail, hardware is not one H20, the diff exceeds planned `train.py` scope, the initial group values are not exactly `[5e-4, 0.0]`, the switch changes the no-decay group, parameter count changes, or the norm helper mutates parameters/RNG.
- Abort on traceback, CUDA/OOM, non-finite loss, or no log progress for two minutes. `timeout 600s` is authoritative.
- Require exactly one transition message between 64.5% and 65.5%, showing matrix weight decay `5e-4 -> 0` while the no-decay group remains zero. A missing, repeated, or mistimed switch is structural failure.
- Do not abort for weak intermediate accuracy. Any completed score below 94.17% is a no-improvement and never authorizes a retry, alternate cutoff, or seed change.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require baseline 94.07%, so success requires at least 94.17%.
2. Run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`, `git diff --check`, `git diff -- train.py`, and `git status --short`; require one H20 and only the planned tracked `train.py` change.
3. Run `uv run ruff check train.py` and `uv run python -m py_compile train.py`. In a separate process, stub `prepare.Eval` before importing the actual module, instantiate the model and exact SGD optimizer, assert its live `[weight_decay]` values are `[5e-4, 0.0]`, invoke the production `disable_weight_decay(optimizer)` helper, require its returned actual values are `(5e-4, 0.0, 0.0)`, and re-read the live groups as `[0.0, 0.0]`. Assert 691,674 parameters and that every first-group tensor has rank at least two while every second-group tensor has rank below two.
4. In the same stubbed semantic check, run `parameter_l2_norm` on fixed CUDA parameters. Require the expected finite norm, bit-identical tensor values, and an unchanged CUDA RNG state. Source inspection must show one existing evaluator call site and no new random operation.
5. Remove stale output with `rm -f run.log`, then execute `timeout 600s uv run train.py > run.log 2>&1` once. A nonzero exit or missing final summary is a crash; inspect only bounded log excerpts.
6. Require a complete summary with `rg '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|peak_vram_mb|num_epochs|num_steps|num_params|final_decayed_weight_l2):' run.log`. Require `300.0 <= training_seconds <= 305.0`, `total_seconds <= 600.0`, `num_params == 691674`, finite loss/norm, and `best_test_acc >= 94.17`.
7. Separately compare `num_steps` with 26,329. Lower exposure does not invalidate an accuracy win under the fixed-time goal; for a negative run it changes attribution from harmful late-decay removal to unexpected instrumentation/runtime overhead, remains no-improvement, and does not permit repetition.
8. Run `rg 'weight decay disabled|eval ep' run.log`; require exactly one transition at 64.5-65.5% whose returned live group values are `5e-4 -> 0.0` and `0.0`, finite transition/final norms, unique every-fifth evaluations plus the final epoch, and no duplicate evaluated epoch. Confirm source computes the final norm before runtime/VRAM snapshots and the final diff keeps `prepare.py`, dependencies, seed, evaluator, mixup, LR, and architecture unchanged.

### Informational Metrics (Optional)

- `peak_vram_mb`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, and `num_params`: final summary in `run.log`.
- Realized passes: `num_steps * 256 / 50000`.
- Transition and decayed-parameter norms: transition line plus `final_decayed_weight_l2` summary. Record descriptively only; without EXP-002 norm telemetry, their direction cannot establish the causal effect of removing decay.

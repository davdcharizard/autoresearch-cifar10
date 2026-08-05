# Plan EXP-006: Early p=0.10 WRN Block Dropout
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement the isolated residual-branch dropout path
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-006` from accepted commit `eb08811` on the integration branch.
- [x] Add p=0.10 dropout after each block's second BN/ReLU and before `conv2`, with an explicit model method that sets every block's probability to zero.
- [x] Disable block dropout exactly once in the existing 65% mixup transition; preserve every accepted architecture, optimizer, schedule, input-mixup, loader, seed, and evaluator setting.
- [x] Run `uv run ruff check train.py`, `uv run python -m py_compile train.py`, and `git diff --check` successfully.

### Milestone 2: Verify semantics, scope, and warm throughput
- [x] Confirm train-mode dropout is stochastic before the transition, eval-mode forward passes are identical, setting block dropout to zero removes the dropout RNG path, and logits retain shape `[256, 10]` with finite loss.
- [x] Run a separate-process, order-balanced matched synthetic H20 benchmark for p=0 and p=0.10 after condition-specific CUDA warmup. Stub `prepare.Eval` before importing the actual `train.py` module so the test dataset is never constructed. Require p=0.10 to retain at least 95% of p=0 image throughput and to project at least 134.8 calibrated data passes (`candidate/baseline throughput * 141.9`).
- [x] Confirm the diff modifies only `train.py`, adds no parameters or evaluator calls, retains 691,674 parameters, and runs on one NVIDIA H20.

### Milestone 3: Execute and monitor the single scored run
- [x] Remove stale `run.log` and run `timeout 600s uv run train.py > run.log 2>&1` exactly once.
- [x] Monitor bounded log extracts for traceback, CUDA/OOM, non-finite loss, progress, the single 65% regularization transition, and final completion without exposing the full log.

### Milestone 4: Verify and record
- [x] Query the current 94.07% baseline and require `best_test_acc >= 94.17%` with no result-conditioned retry (checked; actual 93.52% failed).
- [x] Verify about 300 counted seconds, no more than 600 total seconds, one-H20 execution, at most one evaluation per epoch, unchanged parameter count, the exact planned code scope, and whether realized exposure reaches 26,329 steps / 134.8 passes.
- [x] Record final accuracy/loss, transition, exposure, throughput, VRAM, and dropout interpretation, then remove `run.log` after analysis.

## Code Changes

- **`train.py`**: Add `BLOCK_DROPOUT = 0.10`. Thread the probability through `WideResNet` and `_make_layer` into each `PreActBlock`. In `PreActBlock.forward`, split the current second-convolution expression into BN/ReLU, guarded `F.dropout(..., training=self.training)`, and `conv2`; the identity shortcut remains untouched. Add `WideResNet.set_block_dropout(p)` to update all block probabilities without rebuilding the model or traversing modules on every step. In the existing one-shot `progress >= MIXUP_END_FRACTION` branch, set block dropout to zero and log that both early regularizers were disabled. The `p == 0.0` guard must bypass `F.dropout` entirely so the clean tail consumes no dropout RNG or mask overhead.

The implementation deliberately does not change the block outputs after residual addition, add a second forward pass, alter mixup, or modify evaluation. CUDA dropout changes later random draws before 65%, so this fixed-seed run measures the dropout-enabled stochastic process rather than a bit-identical pairing trajectory against EXP-002.

## Configuration Changes

- `BLOCK_DROPOUT`: new value `0.10` during the first 65% of counted time, then `0.0` (conservative internal regularization with the accepted clean tail preserved).
- `MIXUP_ALPHA = 0.2` and `MIXUP_END_FRACTION = 0.65`: unchanged; architecture, batch 256, LR schedule, optimizer, transforms, evaluation cadence, and seed remain at the EXP-002 baseline.

## Execution Environment

- Method: local single-process execution with `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20 with the existing local CIFAR-10 cache; no network, remote service, GitHub operation, dependency installation, or additional result run.
- Estimated runtime: about 300 counted training seconds and 340-370 total seconds if the warm throughput gate passes.
- Log output: capture all stdout/stderr in `run.log`, inspect only bounded `rg`/`tail` extracts while monitoring, and remove the file after analysis.
- Tool skill: none; execution is fully local.

## Abort Criteria

- Abort before the scored run if lint/compile/diff checks fail, the device is not one H20, the diff extends beyond the planned `train.py` changes, parameter count changes, eval-mode dropout remains active, dropout-zero still consumes a dropout RNG draw, or the warm matched-path gate retains less than 95% throughput / 134.8 calibrated passes. The preflight must import the real implementation through a synthetic `prepare` module whose dummy `Eval` constructs no dataset; importing the real evaluator invalidates the gate and requires correcting the harness before timing.
- Abort the scored run on traceback, CUDA/OOM, non-finite loss, or no training progress for two minutes. The `timeout 600s` process limit is authoritative.
- Require exactly one combined mixup/dropout disable message between 64.5% and 65.5% counted progress. A missing, repeated, or mistimed transition is a structural failure; do not retry without first identifying an implementation defect.
- Do not abort for weak intermediate accuracy. A completed valid run below 94.17% is a no-improvement, and weak accuracy never authorizes a retry or seed change.

## Verification Protocol

### Verification Procedure

1. From the project root, run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv` and require baseline 94.07%, making 94.17% the success threshold.
2. Run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` and require exactly one listed NVIDIA H20. Run `git diff --check`, `git diff -- train.py`, and `git status --short`; require that only the intended `train.py` experiment diff is tracked.
3. Run `uv run ruff check train.py` and `uv run python -m py_compile train.py`. For the separate semantic process, insert a synthetic `prepare` module into `sys.modules` before importing `train`; provide the required constants and a dummy `Eval` whose constructor is inert. Assert six `PreActBlock` modules have p=0.10, eval-mode repeated outputs are identical, train-mode repeated outputs differ under active dropout, `set_block_dropout(0.0)` sets all six probabilities to zero, and the zero-dropout path leaves a sentinel CUDA RNG state unchanged across the guarded dropout section.
4. In another stubbed-`prepare` process, create p=0 and p=0.10 copies from the same initialized `WideResNet` state, identical optimizers, and one fixed synthetic batch of 256 inputs/targets. Give each condition 25 condition-specific warmup steps. Then collect three equal 50-step windows per condition in the fixed alternating order `p=0, p=.10, p=.10, p=0, p=0, p=.10`, synchronizing CUDA immediately before and after every window. Retain all six raw timings and compare the per-condition medians. Require finite loss, `[256, 10]` logits, relative image throughput at least 95%, and calibrated passes `141.9 * candidate_throughput / baseline_throughput >= 134.8`. This preflight must not construct the test set, call the evaluator, or write `run.log`.
5. Remove stale output with `rm -f run.log`, then run the single scored command `timeout 600s uv run train.py > run.log 2>&1`. Treat nonzero exit as a crash unless bounded log inspection proves a structural implementation error that prevented any result.
6. Require a complete final summary with `rg '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`. Require `300.0 <= training_seconds <= 305.0`, `total_seconds <= 600.0`, `num_params == 691674`, finite loss, and `best_test_acc >= 94.17`. Separately compare `num_steps` with 26,329 (134.8 passes). Exposure below that floor does not invalidate an accuracy improvement under the user's fixed-time objective; for a negative run it changes the mechanism attribution from over-regularization to excessive dropout overhead, remains `no-improvement`, and never authorizes a retry.
7. Run `rg 'disabled at|eval ep' run.log`. Require exactly one combined transition at 64.5-65.5%, and confirm each evaluated epoch appears once at most, on every fifth completed epoch plus the final budget-exhausted epoch. Source inspection must confirm p=0 bypasses dropout after the transition.
8. Confirm the final `git diff -- train.py` matches the plan and `prepare.py`, dependencies, seed value, evaluator, evaluation cadence, mixup parameters, optimizer, and LR schedule remain unchanged. Do not repeat a completed result for any accuracy outcome.

### Informational Metrics (Optional)

- `peak_vram_mb`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, and `num_params`: collect from the final summary matched by the `rg` command above.
- Realized passes: compute `num_steps * 256 / 50000` from the final summary.
- Transition timing and LR: collect from the single `disabled at` line.
- Preflight throughput ratio and calibrated passes: retain in the execution report for mechanism attribution, not as a substitute for the primary threshold.

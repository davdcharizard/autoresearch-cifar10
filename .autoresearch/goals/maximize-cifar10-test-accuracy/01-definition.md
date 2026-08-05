# Goal: Maximize CIFAR-10 Test Accuracy
**Created**: 2026-07-24

## Goal Statement
Modernize the 2016 ResNet-20 CIFAR-10 training baseline to achieve the highest possible `best_test_acc` within the fixed wall-clock training budget. Improvements must come from genuine changes to the model or training setup in `train.py`, not from the evaluation harness or seed rerolling.

## Primary Metric
- **Metric**: `best_test_acc` (%)
- **Direction**: higher is better

## Hard Constraints
- Only `train.py` may be modified by experiments; `prepare.py` and the evaluation harness are read-only.
- Do not install packages or add dependencies; use only dependencies already declared in `pyproject.toml`.
- Run every experiment on a single NVIDIA H20 GPU.
- Use the fixed wall-clock training budget defined in `prepare.py`; validation time is excluded from that budget.
- Do not run validation more than once per epoch.
- Do not use seed rerolling as an optimization method.
- Kill any run exceeding 10 minutes and classify it as a failure.
- Redirect training output to `run.log`, and remove experiment logs before starting the next experiment.

## Verification

### Necessary Conditions
- The run completes without crashing and within 10 minutes.
- `best_test_acc` exceeds the current baseline by at least 0.1 percentage points.

### Procedure
1. Confirm the run uses a single NVIDIA H20 GPU.
2. Remove any stale `run.log`, then run `uv run train.py > run.log 2>&1`.
3. Extract `best_test_acc` and `peak_vram_mb` from the final summary in `run.log`.
4. Treat a missing final `best_test_acc` summary as a crash and inspect the last 50 log lines.
5. Confirm the implementation validates no more than once per epoch and respects the fixed training-time budget.

### Informational Metrics (Optional)
- `peak_vram_mb`: peak allocated GPU memory.
- `final_test_acc`: test accuracy at the final evaluation.
- `final_test_loss`: loss at the final evaluation.
- `training_seconds`: measured training time counted against the budget.
- `total_seconds`: total elapsed wall-clock time.
- `num_epochs`: completed training epochs.
- `num_steps`: completed optimizer steps.
- `num_params`: number of trainable model parameters.

## Exit Actions

<!-- Optional. No exit actions were requested for this local-only autopilot run. -->

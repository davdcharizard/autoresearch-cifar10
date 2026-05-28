# Plan EXP-023: Cubic-Scheduled EMA (Lookahead)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md

## Milestones

### Milestone 1: Implement Cubic-Scheduled EMA in train.py
- [x] Add a `LookaheadEMA` class that maintains a shadow copy of the full model state_dict
- [x] Integrate the EMA update into the training loop: every 5 steps, compute `decay = 0.95^5 * (step/total_steps)^3` and lerp shadow toward current weights
- [x] Before the per-epoch TTA evaluation, swap EMA weights into the model (and swap back after)
- [x] At training end (final epoch), copy EMA weights fully (decay=1.0) before final evaluation
- [x] Verify no syntax errors: `uv run python -c "import train"`

### Milestone 2: Run experiment and collect metrics
- [x] Run `uv run python train.py` with output captured to log file
- [x] Confirm training completes within 300s budget
- [x] Confirm ~95-99 epochs complete (throughput not significantly degraded)

### Milestone 3: Verify results
- [x] Extract best_test_acc from log output
- [x] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold) — **FAILED**: 96.02% < 96.56%
- [x] Confirm full summary block printed

## Code Changes
- **train.py**: Add `LookaheadEMA` class (~25 lines) after the model definition section. The class:
  - On init: deep-copies `model.state_dict()` as the shadow
  - `update(step, total_steps)` method: computes cubic decay `alpha = 0.95**5 * (step/total_steps)**3`, then for each key in state_dict: `shadow[k] = shadow[k] + alpha * (model_state[k] - shadow[k])` (i.e., lerp toward current model). Called every 5 steps.
  - `swap(model)` method: swaps shadow state_dict into model for evaluation, caches model's live state_dict to restore afterward
  - `restore(model)` method: restores model's live state_dict after evaluation
  - `copy_to(model)` method: final full copy (decay=1.0) — just `model.load_state_dict(shadow)`

- **train.py (training loop)**: After `scheduler.step()`, add: `if step % 5 == 0: ema.update(step, total_steps)`. The `total_steps` variable already exists (line 181).

- **train.py (evaluation)**: Before `tta_evaluate()` call at end of each epoch, call `ema.swap(model)`. After evaluation, call `ema.restore(model)`. On the final epoch (when training budget exhausted), call `ema.copy_to(model)` instead of swap/restore so the final summary reflects EMA weights permanently.

## Configuration Changes
- No hyperparameter changes. The EMA schedule parameters are hardcoded per airbench96:
  - `lookahead_alpha = 0.95 ** 5` (base decay factor)
  - `update_period = 5` (steps between EMA updates)
  - Cubic ramp exponent: 3

## Execution Environment
- Method: local command `uv run python train.py`
- Resources: single GPU (H20), ~865 MB VRAM
- Estimated runtime: ~310-320s total (300s training + ~15s startup + eval overhead)
- Log output: `uv run python train.py 2>&1 | tee /SPXvePFS/users/david/autoresearch-cifar10/exp-023-run.log`
- Tool skill: none (local execution)

## Abort Criteria
- Training loss becomes NaN or inf within first 5 epochs
- No output produced for >60 seconds after launch
- Fewer than 85 epochs completed in 300s budget (indicating >15% throughput regression from EMA overhead, which would negate any smoothing benefit — EXP-014 showed throughput loss directly hurts accuracy)
- OOM error or CUDA error in logs

## Verification Protocol

### Verification Procedure
After training completes:

1. **Condition 1 — Accuracy threshold**: Run `grep "^best_test_acc:" exp-023-run.log`. Extract the numeric value. Pass if > 96.56 (baseline 96.46 + 0.1pp). Fail otherwise.

2. **Condition 2 — Clean completion**: Run `grep -c "^best_test_acc:" exp-023-run.log`. Pass if the count is exactly 1 (summary block printed). Also check `grep "^num_epochs:" exp-023-run.log` to confirm the summary is complete.

3. **Condition 3 — Validation frequency**: Run `grep -c "eval ep" exp-023-run.log` and `grep "^num_epochs:" exp-023-run.log`. Pass if eval count equals num_epochs (one eval per epoch). Fail if eval count exceeds num_epochs.

Timeout: 600s for the entire training + evaluation cycle. If no output after 600s, treat as infrastructure failure.

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" exp-023-run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" exp-023-run.log`
- final_test_acc: `grep "^final_test_acc:" exp-023-run.log`
- final_test_loss: `grep "^final_test_loss:" exp-023-run.log`
- num_epochs: `grep "^num_epochs:" exp-023-run.log`
- num_steps: `grep "^num_steps:" exp-023-run.log`
- num_params: `grep "^num_params:" exp-023-run.log`

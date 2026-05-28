# EXP-014: Full State Dict EMA (β=0.999)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-014
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented full state_dict EMA in train.py with three localized additions following plan-014.md milestones exactly. Added `import copy` at top and `EMA_BETA = 0.999` constant in hyperparameters section. Initialized `ema_shadow = copy.deepcopy(model.state_dict())` after model creation (line 155). Added EMA update block after `scaler.update()` and `scheduler.step()` — iterates over all keys in ema_shadow, using `lerp_()` for floating-point tensors and `copy_()` for integer tensors (num_batches_tracked). Before evaluation, saves current model state_dict via `copy.deepcopy()`, loads ema_shadow into model, runs eval, then restores original state_dict and calls `model.train()`.

### Surprises & Discoveries

No surprises. The implementation followed the plan exactly. The state_dict includes both parameters and BN buffers (running_mean, running_var, num_batches_tracked) as expected, which is the key fix over EXP-013's parameter-only approach.

### Decisions

Used `model.state_dict()` call inside the training loop (once per step) to get current weights for the EMA update rather than caching references. This is slightly less efficient than maintaining direct tensor references but ensures correctness — state_dict() always returns the authoritative view of all model state including buffers that may not be accessible via named_parameters().

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running full state_dict EMA (β=0.999) training on H20 GPU with 300s budget. This is the direct fix for EXP-013's BN mismatch failure — using model.state_dict() instead of named_parameters() to include BN running_mean/running_var buffers in the EMA shadow. Expected ~98 epochs, best_test_acc target > 95.49% (baseline 95.39%). Command: `uv run train.py > run.log 2>&1`.

Observations:
- Training completed successfully (exit code 0). No errors, tracebacks, NaN, inf, or OOM in the log.
- EMA with full state_dict fixed the BN mismatch from EXP-013: accuracy climbed normally from epoch 1 (no early suppression). EXP-013 showed severely suppressed accuracy in early epochs due to BN buffer mismatch; EXP-014 showed normal progression.
- Achieved 92 epochs in 300s (vs ~98 epochs in baseline EXP-009). The EMA overhead — calling `model.state_dict()` every step plus `copy.deepcopy()` for eval swap — cost ~6 epochs of throughput (~17ms/step vs 16ms baseline).
- Best accuracy 95.44% at some epoch, final accuracy 95.42% — a +0.05pp improvement over baseline (95.39%), but below the 0.1pp verification threshold.
- The EMA smoothing provided marginal benefit at β=0.999 with only 92 epochs. The late-training recovery pattern seen in EXP-013 (where EMA weights converged after LR drops) was less pronounced here because the BN fix eliminated the early suppression — the EMA and non-EMA trajectories were already closer throughout training.

Key Metrics:
- best_test_acc: 95.44%
- final_test_acc: 95.42%
- final_test_loss: 0.1863
- training_seconds: 300.0
- peak_vram_mb: 899.6
- num_epochs: 92
- num_steps: 17865
- num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 95.49%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Actual result: 95.44
- Pass/Fail: **FAIL** (95.44 ≤ 95.49)
- Source: run.log summary block

**Condition 2: Training script completes and prints full summary block**
- Command: `grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^num_epochs:\|^num_steps:" run.log`
- Actual result: 5 (all five summary fields present)
- Pass/Fail: **PASS**
- Source: run.log summary block

**Condition 3: Validation runs at most once per epoch**
- Command: eval count (92) vs epoch count (92)
- Actual result: evals=92, epochs=92, 92 ≤ 92
- Pass/Fail: **PASS**
- Source: run.log eval lines and summary

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 899.6
- final_test_acc: 95.42%
- final_test_loss: 0.1863
- num_epochs: 92
- num_steps: 17865
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

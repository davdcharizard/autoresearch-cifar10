# EXP-020: CosineAnnealingLR with Correct T_max

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-020
- **Commit**: e94abe3
- **PR**: (failed — `gh pr create` returned "Resource not accessible by personal access token"; user should create manually from autoresearch/exp-020 → main)
- **Outcome**: completed

## Implementation Notes

### Summary

Replaced the wall-clock-fractional MultiStepLR schedule with a step-level LambdaLR composing 5-epoch linear warmup and cosine decay. Removed `_lr_progress` cell, `_epoch_count` cell, and `_wall_clock_fractional_step_decay` function (old lines 189-205). Removed `_epoch_count[0] = epoch - 1` update in the training loop. Removed `_lr_progress[0] = total_training_time / TIME_BUDGET_S` update in the training loop. Added new scheduler code: `ESTIMATED_EPOCHS = 100`, `steps_per_epoch = len(train_loader)`, `total_steps = ESTIMATED_EPOCHS * steps_per_epoch`, `warmup_steps = 5 * steps_per_epoch`. The LambdaLR closure `_warmup_cosine(step_idx)` returns a linear ramp from 1/warmup_steps to 1.0 during the first warmup_steps steps, then cosine decay from 1.0 to 0.0 over the remaining steps. Also removed the stale multi-line comment block describing the old schedule.

### Surprises & Discoveries

None — the implementation was straightforward as planned. All three removal targets were cleanly separable.

### Decisions

- Moved `import math` inside `main()` rather than at module level to keep the diff minimal and avoid changing module-level imports. The import is only needed by the scheduler closure.
- Removed the 14-line comment block above the old scheduler code since it described the now-removed wall-clock-fractional schedule and would be misleading.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/.autoresearch/logs/exp-020-run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the full training script with CosineAnnealingLR (via LambdaLR) replacing the wall-clock-fractional MultiStepLR. Training config is identical to EXP-019 baseline (WIDTH_MULT=4, batch 256, AMP, label_smoothing=0.2, TrivialAugmentWide+RandomErasing, TTA evaluation). The only change is the LR schedule: smooth cosine decay from LR=0.2 to ~0 over ~100 estimated epochs with 5-epoch step-level linear warmup, replacing the step decay at 50%/75% wall-clock. Expecting ~98 epochs in 300s, with TTA adding ~117s eval overhead. Target: best_test_acc > 96.01%.

Observations:
- Training completed 99 epochs in 300.0s (within 1 epoch of the ESTIMATED_EPOCHS=100 estimate)
- Throughput consistent with baseline (~19,198 steps across 99 epochs, ~194 steps/epoch)
- Best accuracy 96.46% achieved at final epoch — cosine decay to near-zero LR produced tight convergence at the end of training
- No AMP instability observed — smooth LR transitions avoided the oscillation seen with step decay at intermediate LR regimes
- Peak VRAM 864.6 MB — no change from baseline, confirming zero overhead from scheduler change

Key Metrics:
- best_test_acc: 96.46%
- final_test_acc: 96.46%
- final_test_loss: 0.2838
- training_seconds: 300.0
- total_seconds: 408.2
- startup_seconds: 1.3
- peak_vram_mb: 864.6
- num_epochs: 99
- num_steps: 19,198
- num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.01%**: PASS — best_test_acc is 96.46% (> 96.01%). Source: `.autoresearch/logs/exp-020-run.log` grep `^best_test_acc:`.
2. **Full 10-field summary block printed**: PASS — all 10 fields present (best_test_acc, final_test_acc, final_test_loss, training_seconds, total_seconds, startup_seconds, peak_vram_mb, num_epochs, num_steps, num_params). Source: grep count returned 10.
3. **Eval count ≤ num_epochs**: PASS — eval count is 99 (grep -c "eval ep"), num_epochs is 99. 99 ≤ 99. Source: `.autoresearch/logs/exp-020-run.log`.

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 864.6
- final_test_acc: 96.46%
- final_test_loss: 0.2838
- num_epochs: 99
- num_steps: 19,198
- num_params: 4,286,026

## Errors & Dead Ends

## Human Notes

> 

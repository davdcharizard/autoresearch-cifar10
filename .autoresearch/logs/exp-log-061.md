# EXP-061: Final Classifier Dropout

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-061
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned head-only regularization change in `train.py`. The patch adds `CLASSIFIER_DROPOUT_P = 0.1`, creates `self.classifier_dropout = nn.Dropout(p=CLASSIFIER_DROPOUT_P)` in `ResNet.__init__`, applies it to the flattened pooled feature vector immediately before `self.fc(out)`, and prints the dropout probability at startup for verification.

### Surprises & Discoveries

No code-structure surprises. The current model head is a simple `adaptive_avg_pool2d -> view -> fc` path, so the dropout insertion point is unambiguous and does not require changes to the residual blocks, optimizer, scheduler, data loader, loss, or evaluator.

### Decisions

- Used `nn.Dropout` as a module instead of `F.dropout` so train/eval behavior follows `model.train()` and `model.eval()` automatically, including inside the fixed evaluator.
- Kept dropout after flattening and before `fc` so it regularizes only the classifier input, matching the brainstorm hypothesis.
- Left parameter count expected unchanged because dropout has no learned parameters.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 38551; shell PID 3549581; uv PID 3549582; main python PID 3549585
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 18:45:28 UTC
- **Ended**: 2026-06-09 18:52:10 UTC

Description:
- Local foreground run of EXP-061 on one selected GPU with output captured to `run.log`. This tests whether a narrow training-only classifier-head dropout can reduce final-head overfit without disturbing the residual representation, labels, optimizer, schedule, or evaluation harness. Expected behavior is startup reporting `Classifier dropout p: 0.1`, unchanged batch geometry, unchanged parameter count, first LR drop at step 21000, and final `best_test_acc` classified against the 94.07% improvement threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline for classification: `93.97%`; improvement threshold: `94.07%`.
- 2026-06-09 18:45 UTC: GPU0 selected after `nvidia-smi` showed GPU0 at `0MiB` and `0%` utilization; GPU1 was already running an unrelated process in another checkout.
- 2026-06-09 18:45 UTC: Foreground run launched on GPU0. Process table showed shell PID 3549581, uv PID 3549582, and main Python PID 3549585; `/proc/3549585/cwd` verified this project root.
- Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `Classifier dropout p: 0.1`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 18:46 UTC: Early training is healthy through epoch 3 with best test accuracy 68.59%, mostly 7-8ms batch timings, no traceback/OOM/runtime-error patterns, and GPU0 active.
- 2026-06-09 18:49 UTC: First LR drop confirmed in `run.log` at `step 21000 ep 54` with `lr: 0.0100`. Pre-drop best reached 88.55% at epoch 53, and the immediate post-drop epoch 54 eval reached 91.61%.
- 2026-06-09 18:52 UTC: Run completed cleanly with final summary metrics present. Best accuracy peaked at 93.54% and then plateaued/softened through the final epochs, ending at 93.14%. This is below both the 93.97% baseline and the 94.07% improvement threshold, so EXP-061 is a valid no-improvement.

Key Metrics:
- `best_test_acc`: 93.54%
- `final_test_acc`: 93.14%
- `final_test_loss`: 0.2526
- `training_seconds`: 300.0
- `total_seconds`: 401.8
- `startup_seconds`: 2.6
- `peak_vram_mb`: 660.4
- `num_epochs`: 103
- `num_steps`: 39,806
- `num_params`: 822,790
- Classification: no-improvement (`93.54% < 94.07%`)

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`; improvement threshold is 94.07%.
- Diff scope: passed. `git diff --name-only` listed only `train.py`.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. Local foreground run exited 0 within 10 minutes and produced numeric final metrics.
- Dropout config: passed. `run.log` reported `Classifier dropout p: 0.1`.
- Batch geometry: passed. `run.log` reported `Batches per epoch: 390`.
- LR drop behavior: passed. `run.log` showed `step 21000 ep 54` with `lr: 0.0100`.
- Final metric extraction: passed. Summary metrics were present and `num_params` remained `822,790`.
- Improvement classification: no-improvement. `best_test_acc=93.54%` is below the 94.07% improvement threshold.

### Informational Metrics
- `final_test_acc`: 93.14%
- `final_test_loss`: 0.2526
- `training_seconds`: 300.0
- `total_seconds`: 401.8
- `startup_seconds`: 2.6
- `peak_vram_mb`: 660.4
- `num_epochs`: 103
- `num_steps`: 39,806
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.

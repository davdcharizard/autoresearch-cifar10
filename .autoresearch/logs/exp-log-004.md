# EXP-004: EMA Evaluation Weights

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-004
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented EMA evaluation weights on top of the EXP-002 FP32 throughput baseline. `train.py` now imports PyTorch's `AveragedModel` and `get_ema_multi_avg_fn`, keeps an optimizer-owned `base_model`, compiles the base model only for training forward calls, updates the EMA copy after each optimizer step, and evaluates the EMA model once per epoch.

### Surprises & Discoveries
Local PyTorch 2.11.0 exposes both `AveragedModel` and `get_ema_multi_avg_fn`, so the plan did not need a manual EMA implementation or dependency changes.

### Decisions
The optimizer is attached to `base_model.parameters()` rather than the compiled wrapper to make the EMA source module explicit. The EMA model uses `use_buffers=True` so BatchNorm running statistics are averaged along with parameters.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 4088743
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 11:22
- **Ended**: 2026-06-08 11:28

Description:
- Run the EMA-enabled ResNet-20 FP32 throughput recipe locally on one GPU with output redirected to `run.log`. The intervention keeps architecture, optimizer, LR milestones, augmentation, precision, and evaluation harness unchanged while evaluating an EMA copy of the trained weights. The expected result is a cleaner late-training model snapshot that can reach at least `92.05%` `best_test_acc`.

Observations:
- Startup is clean under `CUDA_VISIBLE_DEVICES=0`: log reports `Device: cuda`, `ResNet-20 | params: 269,722`, `EMA enabled | decay: 0.999`, and `Batches per epoch: 390`. (source: run.log startup lines)
- Early EMA evaluation is healthy: evals appear once per epoch and climb from 14.96% at epoch 1 to 81.41% at epoch 8 without traceback or CUDA errors. (source: run.log lines 7-21)
- Mid-run EMA best reached 91.31% by epoch 49 with no runtime errors; EMA update overhead reduces step throughput enough that the first LR drop will occur later than EXP-002. (source: run.log lines 95-105)
- The LR dropped to 0.01 at step 32,000 near 87.4% of training time. Best accuracy peaked just before that at 91.98% on epoch 80 and did not improve afterward. (source: run.log lines 165-195)

Key Metrics:
- best_test_acc: 91.98% (source: run.log line 195)
- final_test_acc: 91.80% (source: run.log line 196)
- final_test_loss: 0.2816 (source: run.log final summary)
- training_seconds: 300.0 (source: run.log line 198)
- total_seconds: 387.9 (source: run.log line 199)
- peak_vram_mb: 380.1 (source: run.log line 201)
- num_epochs: 94 (source: run.log line 202)
- num_steps: 36,574 (source: run.log line 203)
- num_params: 269,722 (source: run.log line 204)

## Verification Results

### Conditions Checked
- `uv run train.py` completes without crashing: passed. The run reached the final summary with `training_seconds: 300.0` and no traceback. (source: run.log lines 195-204)
- The run reports a numeric `best_test_acc`: passed. The final summary reports `best_test_acc: 91.98%`. (source: run.log line 195)
- `best_test_acc` improves over the current baseline by at least +0.10 percentage points: failed. Current baseline is 91.95%, required threshold is 92.05%, and EXP-004 reached 91.98%. This is above the old baseline but below the updated noise margin. (source: run.log line 195; experiment index baseline query before run)
- Implementation scope and hard-constraint review: skipped — verification stopped after the primary metric condition failed.

### Informational Metrics
- Not collected under the goal protocol because a necessary condition failed.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

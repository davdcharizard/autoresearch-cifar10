# EXP-027: Exclude BatchNorm and Bias from Weight Decay

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-027
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

EXP-027 applies the planned targeted optimizer regularization change: `train.py` now builds `decay_params` and `no_decay_params` from `model.named_parameters()`, applying `WEIGHT_DECAY` only to rank > 1 non-bias tensors and `weight_decay=0.0` to 1D tensors and biases. The current-best 28/56/112 architecture, batch size, learning rate, momentum, LR milestones, FP32 compile/channels-last throughput path, fixed 300s training budget, augmentation, seed, optimizer class, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. Because `torch.compile` wraps parameter names but preserves dimensions and `.bias` suffixes, rank and suffix checks are sufficient for the intended grouping.

### Decisions

The grouping uses `param.ndim <= 1 or name.endswith(".bias")` for no-decay parameters. In this model, that captures BatchNorm scale/shift and classifier bias while preserving decay on convolution and linear weight matrices.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 27741; shell PID 1812847; uv PID 1812848; main Python PID 1812851
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 19:19 UTC
- **Ended**: 2026-06-08 19:25 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with targeted weight decay: `WEIGHT_DECAY=1e-4` for conv/linear weight tensors and `weight_decay=0.0` for BatchNorm and bias parameters. This tests whether removing decay from normalization and bias terms improves calibration/generalization without changing architecture, schedule, or throughput. The run must stay on one GPU, preserve once-per-epoch validation, report `Batches per epoch: 390`, hit the step-21000 LR drop, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=28`, and `improvements=6`; EXP-027 threshold is 93.33%.
- GPU check showed both physical GPUs idle, so the run was launched with `CUDA_VISIBLE_DEVICES=0`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`, validating that batch size, architecture, and fixed budget were preserved. (source: run.log startup lines)
- Early training is healthy with epoch evaluations printing and no traceback/OOM/NaN/Inf patterns found. Pre-drop evaluation reached `best_test_acc=86.27%` by epoch 15. (source: run.log early eval lines through epoch 16)
- The first LR drop was reached at `step 21000` with `lr: 0.0100`, preserving the current anchor schedule. Post-drop accuracy climbed to `best_test_acc=92.68%` by epoch 59, still below the `93.33%` improvement threshold at the latest mid-run check. (source: run.log lines containing `step 21000` and eval lines through epoch 60)
- The run completed normally with exit code 0 and printed summary metrics. Final `best_test_acc=92.99%`, below the 93.23% baseline and the 93.33% improvement threshold, so EXP-027 is no-improvement. (source: run.log summary metrics)

Key Metrics:
- `best_test_acc`: 92.99%
- `final_test_acc`: 92.86%
- `final_test_loss`: 0.3516
- `training_seconds`: 300.0
- `total_seconds`: 392.6
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 111
- `num_steps`: 43188
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-027 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-027`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Optimizer param groups: PASS. `rg -n "decay_params|no_decay_params|weight_decay" train.py` showed both planned groups, `weight_decay=WEIGHT_DECAY`, and `weight_decay=0.0`.
- Preserved batch size and schedule: PASS. `Batches per epoch: 390` confirmed `BATCH_SIZE=128`, and `step 21000` showed `lr: 0.0100`.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=392.6`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=92.99%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned optimizer param-group diff was present during the run, `training_seconds=300.0`, `total_seconds=392.6`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 92.86%
- `final_test_loss`: 0.3516
- `training_seconds`: 300.0
- `total_seconds`: 392.6
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 111
- `num_steps`: 43188
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.

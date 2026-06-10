# EXP-070: Standard CIFAR Channel-Std Normalization

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md
- **Plan**: plans/plan-070.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-070
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-070` from `autoresearch/dev` and implemented the planned input-conditioning change in `train.py`. The training normalization now keeps the existing CIFAR channel means but uses standard CIFAR-10 channel standard deviations `(0.2470, 0.2435, 0.2616)` instead of `(1, 1, 1)`. A startup print was added so the normalization tuple is auditable in `run.log`; the CutMix anchor, architecture, optimizer, LR milestones, reflection crop padding, label smoothing, compile/channels-last path, and validation cadence were preserved.

### Surprises & Discoveries

No implementation surprises. The normalization constants are centralized in `main()` immediately before the training transform, so the behavioral change is a localized tuple update plus one logging line.

### Decisions

Kept the existing channel means unchanged and changed only the std tuple, matching the brainstorm hypothesis. Added logging rather than a new config constant to keep the code change minimal while making the run self-verifying.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 18082; shell PID 3753798, uv PID 3753799, main Python PID 3753802
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 20:55 UTC
- **Ended**: 2026-06-09 21:02 UTC

Description:
- Run the one-file input-conditioning experiment locally on a single available GPU using the fixed CIFAR-10 training harness. This preserves the EXP-064 CutMix anchor and changes only training input channel scaling from unit std to standard CIFAR-10 std. The required improvement threshold is `best_test_acc >= 94.21%`; output is captured to `run.log` for startup markers, LR milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, `Normalize mean: (0.4914, 0.4822, 0.4465), std: (0.247, 0.2435, 0.2616)`, 300s budget, and 390 batches per epoch. GPU1 was selected because GPU0 was occupied by another repo's run; `/proc/3753802/cwd` confirmed the main Python process is in this project root. (source: run.log L1-L6; `/proc/3753802/cwd`)
- The run stayed healthy from an infrastructure perspective: no traceback, CUDA OOM, non-finite, `nan`, or `inf` markers were found. The first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. Accuracy was badly miscalibrated under standard CIFAR std scaling: pre-drop best reached only 61.13% at epoch 45, post-drop best reached 75.03% at epoch 79, and the final summary remained far below the 94.21% improvement threshold. (source: run.log L96, L113, L164, L216-L224)

Key Metrics:
- `best_test_acc`: 75.03%
- `final_test_acc`: 67.83%
- `final_test_loss`: 1.0162
- `training_seconds`: 300.0
- `total_seconds`: 393.4
- `startup_seconds`: 2.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40,225
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch/diff scope: passed. The run executed on `autoresearch/exp-070`; tracked code diff was limited to `train.py`. Persistent untracked `data/` was present and ignored. (source: `git status --short --branch`; `git diff --name-only`)
- Syntax/lint: passed before launch. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed.
- Experiment completion: passed. The foreground local run exited cleanly within the 10-minute wall-clock limit and wrote final summary metrics to `run.log`. (source: run.log L216-L224)
- Startup and scope markers: passed. `run.log` confirmed CUDA, ResNet-20 with 822,790 params, unchanged CutMix anchor settings, standard CIFAR std tuple, 300s budget, and 390 batches per epoch. (source: run.log L1-L6)
- LR milestone: passed. The first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. (source: run.log L113)
- Primary metric extraction: passed. `best_test_acc` was numeric at 75.03%. (source: run.log L216)
- Improvement threshold: failed as an improvement gate, but valid as a completed no-improvement result. Baseline was 94.11% and the required threshold was 94.21%; EXP-070 reached only 75.03%.
- Hard constraints: passed. Only `train.py` was modified among tracked files; `prepare.py`, dependency files, and the evaluation harness were unchanged.

### Informational Metrics
- Final checkpoint accuracy was 67.83%, much lower than the best epoch, showing unstable evaluation after the input rescaling change.
- Runtime stayed inside the expected budget: 300.0 training seconds and 393.4 total seconds.
- Peak VRAM was 660.4 MB, matching prior local CutMix runs and indicating no resource pressure.
- The run completed 104 epochs / 40,225 steps, slightly more than recent local runs because per-step throughput was healthy.

## Errors & Dead Ends
- No infrastructure errors. The experimental result itself is a dead end: standard CIFAR channel-std normalization severely degraded the current CutMix anchor.

## Human Notes

> Autopilot execution; no human intervention during implementation.

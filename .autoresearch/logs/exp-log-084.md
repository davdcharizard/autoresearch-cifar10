# EXP-084: Horizontal Flip Probability 0.45

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-084.md
- **Plan**: plans/plan-084.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-084
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved upper-side horizontal-flip bracket around the EXP-082 spatial augmentation anchor. `train.py` changes only the training transform from `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.45)` and updates the startup marker to `RandomHorizontalFlip p: 0.45`. All other augmentation, CutMix, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target transform and startup marker were explicit and localized, so the change was a direct two-line patch.

### Decisions

Kept EXP-084 isolated to horizontal flip probability, as planned. No opportunistic cleanup or coupled configuration changes were added, preserving a clean comparison against the 94.36% EXP-082 baseline and the 94.46% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 4946; launcher PID 4061895; uv PID 4061896; train PID 4061899
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 00:17 UTC
- **Ended**: 2026-06-10 00:24 UTC

Description:
- Local single-GPU foreground run of EXP-084 using the current CutMix anchor with horizontal flip probability increased from 0.4 to 0.45. This tests whether restoring a little horizontal invariance improves beyond the EXP-082 baseline of 94.36%. The run is expected to preserve throughput, parameter count, the step-21000 LR drop, and fixed wall-clock training behavior while attempting to reach at least 94.46% `best_test_acc`.

Observations:
- GPU0 was selected because `nvidia-smi` showed GPU0 at 0 MiB / 0% utilization while GPU1 had an active run from another workspace. (source: `nvidia-smi` pre-launch output, 2026-06-10 00:17 UTC)
- Startup confirmed CUDA execution, `RandomHorizontalFlip p: 0.45`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L6)
- Pre-drop training progressed normally with no tracebacks, NaNs, or CUDA errors; the best validation accuracy before the first LR drop was 88.75%. (source: run.log L82-L112)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity; post-drop accuracy climbed to 93.66% by epoch 59. (source: run.log L113-L124)
- Late refinement peaked at 94.05% in epoch 96 and did not exceed that value through epoch 101. This is below the 94.36% EXP-082 baseline and below the 94.46% noise-guard threshold, so EXP-084 is `no-improvement`. (source: run.log L186-L219)

Key Metrics:
- `best_test_acc`: 94.05%
- `final_test_acc`: 93.68%
- `final_test_loss`: 0.2537
- `training_seconds`: 300.0
- `total_seconds`: 395.2
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,294
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code-scope constraint: passed. `git diff --name-only` listed only `train.py`.
- Syntax and style: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Implementation from code/log: passed. The diff changes `RandomHorizontalFlip` to `p=0.45`, `run.log` line 2 confirms `RandomHorizontalFlip p: 0.45`, and line 4 confirms unchanged CutMix alpha/probability/label smoothing.
- Scheduler behavior: passed. `run.log` line 113 shows step 21000 switched to `lr: 0.0100`.
- Run completion and primary metric: passed. `run.log` lines 210-219 report final metrics including numeric `best_test_acc: 94.05%`.
- Hard constraints: passed. Only `train.py` changed; parameter count stayed 822,790; validation remained once per epoch; fixed 300s training budget was used.
- Improvement threshold: failed for improvement classification. The active baseline is 94.36%, and the required threshold is 94.46%; EXP-084 reached 94.05%, so it is below the +0.10pp noise guard and is classified as `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.68%
- `final_test_loss`: 0.2537
- `training_seconds`: 300.0
- `total_seconds`: 395.2
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,294
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

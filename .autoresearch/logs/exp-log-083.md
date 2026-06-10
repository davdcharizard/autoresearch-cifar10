# EXP-083: Horizontal Flip Probability 0.35

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-083.md
- **Plan**: plans/plan-083.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-083
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned single-knob lower-side bracket around the EXP-082 spatial augmentation anchor. `train.py` now changes only `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.35)` and updates the startup marker to `RandomHorizontalFlip p: 0.35`. All other augmentation, CutMix, model, optimizer, schedule, seed, compile/channels-last, and validation behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target transform and marker were already explicit from EXP-082, so the change was a direct two-line patch.

### Decisions

Kept the experiment isolated to horizontal flip probability, as planned. No configuration coupling or opportunistic cleanup was added, preserving a clean comparison against the 94.36% EXP-082 baseline and the 94.46% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 21274; launcher PID 4036813; uv PID 4036814; train PID 4036817
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 00:00 UTC
- **Ended**: 2026-06-10 00:08 UTC

Description:
- Local single-GPU foreground run of EXP-083 using the current CutMix anchor with horizontal flip probability reduced from 0.4 to 0.35. This tests whether further spatial de-regularization can improve beyond the EXP-082 baseline of 94.36%. The run is expected to preserve throughput, parameter count, the step-21000 LR drop, and fixed wall-clock training behavior while attempting to reach at least 94.46% `best_test_acc`.

Observations:
- GPU0 was selected because `nvidia-smi` showed GPU0 at 0 MiB / 0% utilization while GPU1 had residual load before launch. (source: `nvidia-smi` pre-launch output, 2026-06-10 00:00 UTC)
- Startup confirmed CUDA execution, `RandomHorizontalFlip p: 0.35`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L6)
- Pre-drop training progressed normally with no tracebacks, NaNs, or CUDA errors; by epoch 26 the best validation accuracy was 85.21%. (source: run.log L8-L58)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity; post-drop accuracy climbed to 93.51% by epoch 59. (source: run.log L112-L124)
- Late refinement peaked at 94.17% in epoch 79 and never exceeded that value through epoch 102. This is below the 94.36% EXP-082 baseline and below the 94.46% noise-guard threshold, so EXP-083 is `no-improvement`. (source: run.log L160-L221)

Key Metrics:
- `best_test_acc`: 94.17%
- `final_test_acc`: 93.18%
- `final_test_loss`: 0.2680
- `training_seconds`: 300.0
- `total_seconds`: 393.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,691
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code-scope constraint: passed. `git diff --name-only` listed only `train.py`.
- Syntax and style: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Implementation from code/log: passed. The diff changes `RandomHorizontalFlip` to `p=0.35`, `run.log` line 2 confirms `RandomHorizontalFlip p: 0.35`, and line 4 confirms unchanged CutMix alpha/probability/label smoothing.
- Scheduler behavior: passed. `run.log` line 113 shows step 21000 switched to `lr: 0.0100`.
- Run completion and primary metric: passed. `run.log` lines 212-221 report final metrics including numeric `best_test_acc: 94.17%`.
- Hard constraints: passed. Only `train.py` changed; parameter count stayed 822,790; validation remained once per epoch; fixed 300s training budget was used.
- Improvement threshold: failed for improvement classification. The active baseline is 94.36%, and the required threshold is 94.46%; EXP-083 reached 94.17%, so it is below the +0.10pp noise guard and is classified as `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.18%
- `final_test_loss`: 0.2680
- `training_seconds`: 300.0
- `total_seconds`: 393.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,691
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

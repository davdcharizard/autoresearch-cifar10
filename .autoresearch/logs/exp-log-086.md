# EXP-086: Crop Padding 2 on Padding-3 / Flip p=0.4 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-086.md
- **Plan**: plans/plan-086.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-086
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved lower-side crop-strength bracket on the current EXP-085 spatial anchor. `train.py` changes only the training crop transform from `padding=3` to `padding=2` while preserving `RandomHorizontalFlip(p=0.4)`, and updates the startup marker to `RandomCrop padding: 2 reflect`. All other augmentation, CutMix, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target crop transform and existing crop marker were explicit and localized.

### Decisions

Kept EXP-086 isolated to crop padding on the validated padding-3 / flip p=0.4 anchor, as planned. No coupled changes to CutMix, schedule, optimizer, normalization, or architecture were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 71085; launcher PID 4103903; uv PID 4103904; train PID 4103907
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 00:45 UTC
- **Ended**: 2026-06-10 00:53 UTC

Description:
- Local single-GPU foreground run of EXP-086 using the current CutMix anchor with `RandomHorizontalFlip(p=0.4)` preserved and reflection crop padding reduced from 3 to 2. This tests whether EXP-085's successful spatial de-regularization has more lower-side crop-jitter headroom. The run is expected to preserve throughput, parameter count, the step-21000 LR drop, and fixed wall-clock training behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- GPU0 was selected because both H20 GPUs were free and `nvidia-smi` showed no running processes before launch. (source: `nvidia-smi` pre-launch output, 2026-06-10 00:45 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 2 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached on schedule at step 21000 with `lr: 0.0100`; pre-drop best was 88.85% at epoch 34 and stayed 88.85% through epoch 53. (source: run.log L69, L113-L115)
- Early post-drop convergence reached 93.43% by epoch 59, still below the 94.61% improvement threshold. (source: run.log L115-L125)
- Late refinement peaked at 94.22% in epoch 86 and did not exceed that value through epoch 102. This is below the 94.51% EXP-085 baseline and below the 94.61% noise-guard threshold, so EXP-086 is `no-improvement`. (source: run.log L179-L222)

Key Metrics:
- `best_test_acc`: 94.22%
- `final_test_acc`: 93.88%
- `final_test_loss`: 0.2741
- `training_seconds`: 300.0
- `total_seconds`: 397.4
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,505
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code-scope constraint: passed. `git diff --name-only` listed only `train.py`.
- Syntax and style: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Implementation from code/log: passed. The diff changes the reflection crop to `padding=2`, `run.log` line 2 confirms `RandomCrop padding: 2 reflect`, line 3 confirms `RandomHorizontalFlip p: 0.4`, and line 5 confirms unchanged CutMix alpha/probability/label smoothing.
- Scheduler behavior: passed. `run.log` line 114 shows step 21000 switched to `lr: 0.0100`.
- Run completion and primary metric: passed. `run.log` lines 213-222 report final metrics including numeric `best_test_acc: 94.22%`.
- Hard constraints: passed. Only `train.py` changed; parameter count stayed 822,790; validation remained once per epoch; fixed 300s training budget was used.
- Improvement threshold: failed for improvement classification. The active baseline is 94.51%, and the required threshold is 94.61%; EXP-086 reached 94.22%, so it is below the +0.10pp noise guard and is classified as `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.88%
- `final_test_loss`: 0.2741
- `training_seconds`: 300.0
- `total_seconds`: 397.4
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,505
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

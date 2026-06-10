# EXP-091: Fine Lower CutMix Alpha 0.75 on Spatial Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-091.md
- **Plan**: plans/plan-091.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-091
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved fine CutMix alpha reduction on the EXP-085 spatial anchor. `train.py` changes only `CUTMIX_ALPHA = 1.0` to `CUTMIX_ALPHA = 0.75`. CutMix probability, CutMix endpoint label smoothing, clean label smoothing, reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target value was an existing top-level hyperparameter, and the startup marker automatically reflects the new alpha through the formatted CutMix settings print.

### Decisions

Kept EXP-091 isolated to one CutMix scalar as planned. No coupled changes to crop, flip, CutMix probability, label smoothing, weight decay, schedule, optimizer, normalization, architecture, or batch size were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 66778; launcher PID 23006; uv PID 23007; train PID 23010
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 01:54 UTC
- **Ended**: 2026-06-10 02:00 UTC

Description:
- Local single-GPU foreground run of EXP-091 using the current spatial anchor with CutMix alpha reduced from 1.0 to 0.75. This tests whether the less spatially aggressive padding-3 / flip-p=0.4 recipe benefits from a slightly different CutMix patch-area distribution while preserving p=0.5 application frequency. The run is expected to preserve throughput, parameter count, first LR drop timing, and fixed wall-clock behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- Preflight confirmed only `train.py` changed, the diff was `CUTMIX_ALPHA = 1.0` -> `0.75`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`. (source: command output, 2026-06-10)
- GPU0 was selected because both H20 GPUs were idle and no active training processes were found. After startup, GPU0 showed expected training memory. (source: `nvidia-smi` and `pgrep`, 2026-06-10 01:54 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, `CutMix alpha: 0.75, prob: 0.5, label smoothing: 0.05`, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity. Pre-drop best was 88.99% through epoch 53; early post-drop convergence reached 93.68% by epoch 58, still below the 94.61% improvement threshold. (source: run.log L95-L123)
- The run completed cleanly with no error signatures. Final `best_test_acc` was 94.34%, which is 0.17pp below the 94.51% baseline and 0.27pp below the 94.61% improvement threshold; EXP-091 is therefore classified as no-improvement. (source: run.log L213-L222)

Key Metrics:
- `best_test_acc`: 94.34%
- `final_test_acc`: 93.60%
- `final_test_loss`: 0.2581
- `training_seconds`: 300.0
- `total_seconds`: 394.3
- `startup_seconds`: 2.2
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39540
- `num_params`: 822,790
- Verdict: no-improvement (`best_test_acc` < 94.61% threshold)

## Verification Results

### Conditions Checked

- Code scope: PASS. `git diff --name-only` listed only `train.py`; untracked `data/` was ignored as pre-existing dataset state.
- Syntax: PASS. `python3 -m py_compile train.py` exited 0.
- Style: PASS. `uv run ruff check train.py` reported `All checks passed!`.
- Implementation markers: PASS. The diff changes only `CUTMIX_ALPHA = 1.0` to `0.75`; `run.log` confirms `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, and `CutMix alpha: 0.75, prob: 0.5, label smoothing: 0.05`. (source: git diff; run.log L2-L5)
- Scheduler behavior: PASS. Step 21000 switched to `lr: 0.0100`. (source: run.log L114)
- Metric presence: PASS. `run.log` includes numeric `best_test_acc: 94.34%` and `peak_vram_mb: 660.4`. (source: run.log L213-L219)
- Hard constraints: PASS. Only `train.py` changed; startup log and diff preserve architecture, parameter count, normalization, optimizer/schedule anchors, fixed time budget, seed behavior, validation cadence, spatial settings, CutMix probability, and label smoothing.
- Improvement threshold: FAIL for improvement classification. Baseline remains 94.51% at commit `83d4e94`; the required threshold with +0.10pp noise guard is 94.61%; EXP-091 reached 94.34%.

### Informational Metrics

- `best_test_acc`: 94.34%
- `final_test_acc`: 93.60%
- `final_test_loss`: 0.2581
- `training_seconds`: 300.0
- `total_seconds`: 394.3
- `startup_seconds`: 2.2
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39540
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

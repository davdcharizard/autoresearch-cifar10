# EXP-090: Fine Lower CutMix Probability p=0.4 on Spatial Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-090.md
- **Plan**: plans/plan-090.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-090
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved fine CutMix probability reduction on the EXP-085 spatial anchor. `train.py` changes only `CUTMIX_PROB = 0.5` to `CUTMIX_PROB = 0.4`. Reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, CutMix alpha, CutMix endpoint label smoothing, clean label smoothing, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target value was an existing top-level hyperparameter, and the startup marker automatically reflects the new probability through the formatted CutMix settings print.

### Decisions

Kept EXP-090 isolated to one CutMix scalar as planned. No coupled changes to crop, flip, CutMix alpha, label smoothing, weight decay, schedule, optimizer, normalization, architecture, or batch size were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 36760; launcher PID 1416; uv PID 1417; train PID 1420
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 01:43 UTC
- **Ended**: 2026-06-10 01:50 UTC

Description:
- Local single-GPU foreground run of EXP-090 using the current spatial anchor with CutMix probability reduced from 0.5 to 0.4. This tests whether the less spatially aggressive padding-3 / flip-p=0.4 recipe wants slightly less regional mixed-label pressure while preserving the validated CutMix mechanism. The run is expected to preserve throughput, parameter count, first LR drop timing, and fixed wall-clock behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- Preflight confirmed only `train.py` changed, the diff was `CUTMIX_PROB = 0.5` -> `0.4`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`. (source: command output, 2026-06-10)
- GPU0 was selected because `nvidia-smi` showed GPU0 idle while GPU1 had active load from another workspace. After startup, GPU0 showed expected training memory. (source: `nvidia-smi`, 2026-06-10 01:43 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, `CutMix alpha: 1.0, prob: 0.4, label smoothing: 0.05`, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity. Pre-drop best was 88.67% through epoch 53; early post-drop convergence reached 93.35% by epoch 59, still below the 94.61% improvement threshold. (source: run.log L107-L125)
- The run completed cleanly with no error signatures. Final `best_test_acc` was 94.13%, which is 0.38pp below the 94.51% baseline and 0.48pp below the 94.61% improvement threshold; EXP-090 is therefore classified as no-improvement. (source: run.log L217-L226)

Key Metrics:
- `best_test_acc`: 94.13%
- `final_test_acc`: 93.88%
- `final_test_loss`: 0.2491
- `training_seconds`: 300.0
- `total_seconds`: 399.3
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40415
- `num_params`: 822,790
- Verdict: no-improvement (`best_test_acc` < 94.61% threshold)

## Verification Results

### Conditions Checked

- Code scope: PASS. `git diff --name-only` listed only `train.py`; untracked `data/` was ignored as pre-existing dataset state.
- Syntax: PASS. `python3 -m py_compile train.py` exited 0.
- Style: PASS. `uv run ruff check train.py` reported `All checks passed!`.
- Implementation markers: PASS. The diff changes only `CUTMIX_PROB = 0.5` to `0.4`; `run.log` confirms `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, and `CutMix alpha: 1.0, prob: 0.4, label smoothing: 0.05`. (source: git diff; run.log L2-L5)
- Scheduler behavior: PASS. Step 21000 switched to `lr: 0.0100`. (source: run.log L114)
- Metric presence: PASS. `run.log` includes numeric `best_test_acc: 94.13%` and `peak_vram_mb: 660.4`. (source: run.log L217-L223)
- Hard constraints: PASS. Only `train.py` changed; startup log and diff preserve architecture, parameter count, normalization, optimizer/schedule anchors, fixed time budget, seed behavior, validation cadence, spatial settings, and all CutMix settings except the planned probability.
- Improvement threshold: FAIL for improvement classification. Baseline remains 94.51% at commit `83d4e94`; the required threshold with +0.10pp noise guard is 94.61%; EXP-090 reached 94.13%.

### Informational Metrics

- `best_test_acc`: 94.13%
- `final_test_acc`: 93.88%
- `final_test_loss`: 0.2491
- `training_seconds`: 300.0
- `total_seconds`: 399.3
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40415
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

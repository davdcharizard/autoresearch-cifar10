# EXP-089: Fine Lower Flip Bracket p=0.375 Under Padding 3

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-089.md
- **Plan**: plans/plan-089.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-089
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved fine lower flip bracket on the EXP-085 spatial anchor. `train.py` changes only `RandomHorizontalFlip(p=0.4)` to `RandomHorizontalFlip(p=0.375)` and updates the startup marker to `RandomHorizontalFlip p: 0.375`. Reflection crop padding 3, unit-std normalization, static CutMix settings, clean label smoothing, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target transform and marker were explicit and localized.

### Decisions

Kept EXP-089 isolated to the remaining lower-side flip bracket. No coupled changes to crop padding, CutMix, weight decay, schedule, optimizer, normalization, or architecture were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 46513; launcher PID 4173620; uv PID 4173621; train PID 4173624
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 01:30 UTC
- **Ended**: 2026-06-10 01:37 UTC

Description:
- Local single-GPU foreground run of EXP-089 using the current CutMix anchor with reflection crop padding 3 preserved and horizontal flip probability reduced from 0.4 to 0.375. This tests the remaining lower-side local flip bracket after crop padding 2, p=0.425, and stronger weight decay all failed. The run is expected to preserve throughput, parameter count, first LR drop timing, and fixed wall-clock training behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- GPU0 was selected because `nvidia-smi` showed GPU0 idle while GPU1 had active load from another workspace. After startup, GPU0 showed expected training memory. (source: `nvidia-smi`, 2026-06-10 01:30 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.375`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity. Pre-drop best was 88.29% through epoch 53; early post-drop convergence reached 94.07% by epoch 72, still below the 94.61% improvement threshold. (source: run.log L107-L151)
- The run completed cleanly with no error signatures. Final `best_test_acc` was 94.29%, which is 0.22pp below the 94.51% baseline and 0.32pp below the 94.61% improvement threshold; EXP-089 is therefore classified as no-improvement. (source: run.log L211-L220)

Key Metrics:
- `best_test_acc`: 94.29%
- `final_test_acc`: 93.96%
- `final_test_loss`: 0.2749
- `training_seconds`: 300.0
- `total_seconds`: 393.7
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39286
- `num_params`: 822,790
- Verdict: no-improvement (`best_test_acc` < 94.61% threshold)

## Verification Results

### Conditions Checked

- Code scope: PASS. `git diff --name-only` listed only `train.py`; untracked `data/` was ignored as pre-existing dataset state.
- Syntax: PASS. `python3 -m py_compile train.py` exited 0.
- Style: PASS. `uv run ruff check train.py` reported `All checks passed!`.
- Implementation markers: PASS. The diff changes only `RandomHorizontalFlip(p=0.4)` to `p=0.375` plus the startup marker; `run.log` confirms `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.375`, and unchanged CutMix alpha/probability/label smoothing. (source: git diff; run.log L2-L5)
- Scheduler behavior: PASS. Step 21000 switched to `lr: 0.0100`. (source: run.log L114)
- Metric presence: PASS. `run.log` includes numeric `best_test_acc: 94.29%` and `peak_vram_mb: 660.4`. (source: run.log L211-L217)
- Hard constraints: PASS. Only `train.py` changed; startup log and diff preserve architecture, parameter count, normalization, optimizer/schedule anchors, fixed time budget, seed behavior, validation cadence, and CutMix settings.
- Improvement threshold: FAIL for improvement classification. Baseline remains 94.51% at commit `83d4e94`; the required threshold with +0.10pp noise guard is 94.61%; EXP-089 reached 94.29%.

### Informational Metrics

- `best_test_acc`: 94.29%
- `final_test_acc`: 93.96%
- `final_test_loss`: 0.2749
- `training_seconds`: 300.0
- `total_seconds`: 393.7
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39286
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

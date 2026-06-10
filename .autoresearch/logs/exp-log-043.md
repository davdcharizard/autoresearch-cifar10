# EXP-043: Initial LR 0.08 on the 2e-4 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-043.md
- **Plan**: plans/plan-043.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-043
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - skipped if no remote exists)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned isolated lower-initial-LR probe in `train.py` by changing only `LR = 0.1` to `LR = 0.08`. The architecture, batch size, momentum, weight decay, LR milestones, label smoothing, reflected crop padding, compile/channels-last path, seed, and once-per-epoch validation path were preserved.

### Surprises & Discoveries

No implementation surprises. The current anchor exposes `LR` as a single top-level scalar, and the preflight diff confirms there are no other tracked source changes.

### Decisions

Kept the existing `MultiStepLR` milestones unchanged at `[21000, 64000]`. This isolates the lower initial LR while still making the first post-drop LR `0.0080`, matching the planned lower-LR bracket.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 11360; shell PID 2897369, `uv run train.py` PID 2897370, Python worker PID 2897373
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 12:18 UTC
- **Ended**: 2026-06-09 12:25 UTC

Description:
- Run the current CIFAR-10 training harness locally on a single selected GPU with `LR = 0.08`. This tests whether less high-LR noise improves the late post-drop plateau of the current `2e-4` label-smoothed reflection anchor. The success threshold is `best_test_acc >= 94.07%`, because the active baseline is 93.97% and the goal requires a +0.10 percentage-point improvement.

Observations:
- Startup is clean on GPU 0: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, `Batches per epoch: 390`, and first progress lines at `lr: 0.0800`. (source: `run.log` L1-L12)
- Pre-drop training remained finite and reached a best validation accuracy of 89.83% by epoch 45. The first LR drop occurred at step 21000 with `lr: 0.0080`, as planned. (source: `run.log` L94-L111)
- Post-drop accuracy peaked at 93.49% on epoch 72 and did not recover above the current 93.97% baseline. The final evaluation was 93.19% after 102 epochs / 39,520 steps. (source: `run.log` L145-L210)

Key Metrics:
- `best_test_acc`: 93.49%
- `final_test_acc`: 93.19%
- `final_test_loss`: 0.2510
- `training_seconds`: 300.0
- `total_seconds`: 413.4
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39520
- `num_params`: 822,790
- Provisional verdict for analysis: valid `no-improvement`; `best_test_acc` is below the 94.07% threshold.

## Verification Results

### Conditions Checked
- Baseline and improvement threshold: passed. Current baseline is 93.97%, so EXP-043 must reach at least 94.07% to count as an improvement.
- Scope before launch: passed. The only tracked code diff during execution was `train.py` with `LR = 0.1` changed to `LR = 0.08`; `.autoresearch/` artifacts are local-only and `data/` remains untracked.
- Syntax and lint: passed. `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported all checks passed.
- Lower LR, preserved anchor, and validation cadence: passed. `LR = 0.08` was present; architecture, batch size, momentum, weight decay, milestones, label smoothing, reflected crop, and once-per-epoch evaluation anchors were preserved.
- Preserved schedule behavior, batch geometry, parameter count, and active LR: passed. `Batches per epoch: 390`, first progress lines showed `lr: 0.0800`, step 21000 showed `lr: 0.0080`, step 64000 was absent, and `num_params` was `822,790`.
- Experiment completion: passed. The foreground process exited 0, printed numeric final metrics, and completed in 413.4 total seconds, below the 10-minute wall-clock cap.
- Metric improvement: failed. `best_test_acc` was 93.49%, below the required 94.07% improvement threshold.
- Hard constraints: passed. Only the planned `train.py` LR scalar diff was present during the run, the fixed 300s training budget was used, and no protected files changed.

### Informational Metrics
- Final accuracy was 93.19%, final loss was 0.2510, and peak reported VRAM was 660.4 MB. The run completed 102 epochs and 39,520 optimizer steps.

## Errors & Dead Ends

## Human Notes

> No human notes yet.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

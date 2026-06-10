# EXP-088: Fine Stronger Weight Decay 2.5e-4 on Spatial Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-088.md
- **Plan**: plans/plan-088.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-088
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved fine stronger-decay retune on the EXP-085 spatial anchor. `train.py` changes `WEIGHT_DECAY` from `2e-4` to `2.5e-4` and adds a startup marker `Weight decay: 0.00025`. Reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, static CutMix settings, clean label smoothing, architecture, optimizer type, LR milestones, batch size, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises after correcting an initial local patch insertion that briefly duplicated the `WEIGHT_DECAY` assignment before preflight. The final diff contains exactly one `WEIGHT_DECAY = 2.5e-4` definition.

### Decisions

Added an explicit weight-decay startup marker instead of relying only on the git diff, matching the plan's verification requirement and making `run.log` self-identifying for EXP-088.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 41305; launcher PID 4152579; uv PID 4152580; train PID 4152583
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 01:17 UTC
- **Ended**: 2026-06-10 01:23 UTC

Description:
- Local single-GPU foreground run of EXP-088 using the current padding-3 / flip-p=0.4 CutMix anchor with only weight decay increased from `2e-4` to `2.5e-4`. This tests whether slightly stronger non-spatial shrinkage improves the newer spatially de-regularized recipe. The run is expected to preserve throughput, parameter count, first LR drop timing, and fixed wall-clock training behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- GPU0 was selected because `nvidia-smi` showed both GPU0 and GPU1 idle at 0 MiB / 0% utilization before launch. After startup, GPU0 showed expected training memory while GPU1 remained idle. (source: `nvidia-smi`, 2026-06-10 01:17 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, `Weight decay: 0.00025`, and a 300s time budget. (source: run.log L1-L8)
- The first LR drop was reached on schedule at step 21000 with `lr: 0.0100`; pre-drop best was 87.91% at epoch 45 and 53, and early post-drop convergence reached 93.46% by epoch 59. (source: run.log L107-L123)
- The run peaked at 94.07% on epoch 71, then stayed below that value through epoch 101 and finished at 93.12%. This is below both the 94.51% baseline and the 94.61% noise-guard threshold. (source: run.log L150-L221)

Key Metrics:
- `best_test_acc`: 94.07%
- `final_test_acc`: 93.12%
- `final_test_loss`: 0.3075
- `training_seconds`: 300.0
- `total_seconds`: 395.4
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,060
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- **Code-scope constraint**: passed. `git diff --name-only` listed only `train.py`; untracked `data/` was unrelated and preserved.
- **Syntax and style**: passed. `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- **Implementation intent**: passed. `git diff train.py` changed `WEIGHT_DECAY` from `2e-4` to `2.5e-4` and added the planned startup marker; no other configuration changed.
- **Startup markers**: passed. `run.log` confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, `Weight decay: 0.00025`, and `Time budget: 300s`.
- **Scheduler integrity**: passed. The first LR drop was reached at step 21000 with `lr: 0.0100`.
- **Primary metric availability**: passed. `run.log` reported numeric `best_test_acc: 94.07%`.
- **Hard constraints**: passed. Only `train.py` changed; no dependency, data, `prepare.py`, evaluation-harness, seed, validation cadence, architecture, optimizer type, LR milestone, normalization, CutMix, or fixed-budget behavior changes were made.
- **Improvement threshold**: failed for improvement classification. Baseline was 94.51%, the +0.10 percentage-point threshold required 94.61%, and EXP-088 reached 94.07%, so the verdict is `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.12%
- `final_test_loss`: 0.3075
- `training_seconds`: 300.0
- `total_seconds`: 395.4
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,060
- `num_params`: 822,790

## Errors & Dead Ends
- No crash, OOM, NaN, non-finite-loss, or infrastructure error occurred. The approach failed because the primary metric did not exceed the baseline or threshold.

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

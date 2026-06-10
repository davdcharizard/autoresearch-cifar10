# EXP-092: Fine Lower Weight Decay 1.75e-4 on Spatial Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-092.md
- **Plan**: plans/plan-092.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-092
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed; no-improvement (`best_test_acc=94.14%` < `94.61%` threshold)

## Implementation Notes

### Summary

Implemented the approved fine lower weight-decay bracket on the EXP-085 spatial anchor. `train.py` changes only `WEIGHT_DECAY = 2e-4` to `WEIGHT_DECAY = 1.75e-4`. CutMix settings, clean label smoothing, reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, architecture, optimizer type, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target value was an existing top-level hyperparameter.

### Decisions

Kept EXP-092 isolated to one optimizer regularization scalar as planned. No coupled changes to crop, flip, CutMix, label smoothing, schedule, optimizer type, normalization, architecture, or batch size were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 10006; launcher PID 44819; uv PID 44820; train PID 44823
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 02:05 UTC
- **Ended**: 2026-06-10 02:16 UTC

Description:
- Local single-GPU foreground run of EXP-092 using the current spatial/CutMix anchor with weight decay reduced from `2e-4` to `1.75e-4`. This tests whether the anchor is slightly over-regularized by coupled L2 shrinkage after stronger decay and CutMix scalar brackets failed. The run is expected to preserve throughput, parameter count, first LR drop timing, and fixed wall-clock behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- Preflight confirmed only `train.py` changed, the diff was `WEIGHT_DECAY = 2e-4` -> `1.75e-4`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`. (source: command output, 2026-06-10)
- GPU0 was selected because both H20 GPUs were idle and no active training processes were found. After startup, GPU0 showed expected training memory. (source: `nvidia-smi` and `pgrep`, 2026-06-10 02:05 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached at step 21000 with `lr: 0.0100`, preserving schedule integrity. Pre-drop best was 88.77% through epoch 53; early post-drop convergence reached 93.44% by epoch 57, still below the 94.61% improvement threshold. (source: run.log L95-L121)
- Run completed cleanly with no traceback, runtime, CUDA OOM, NaN, or non-finite markers. Late post-drop best reached 94.14% at epoch 91, but this is 0.37pp below the 94.51% baseline and 0.47pp below the 94.61% noise-guarded improvement threshold. (source: run.log L177-L220; error scan command, 2026-06-10)

Key Metrics:
- best_test_acc: 94.14%
- final_test_acc: 94.13%
- final_test_loss: 0.2488
- training_seconds: 300.0
- total_seconds: 397.0
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 101
- num_steps: 39,173
- num_params: 822,790
- Classification: no-improvement (`94.14% < 94.61%`)

## Verification Results

### Conditions Checked
- Code-scope constraint: `git diff --name-only` listed only `train.py`; pass.
- Syntax/style: `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`; pass.
- Implementation: `git diff train.py` showed only `WEIGHT_DECAY = 2e-4` -> `1.75e-4`; pass.
- Startup markers: `run.log` confirmed `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, and unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`; pass.
- Scheduler behavior: `run.log` confirmed `step 21000` with `lr: 0.0100`; pass.
- Completion/metric: `run.log` reported numeric `best_test_acc: 94.14%` and `peak_vram_mb: 660.4`; pass.
- Hard constraints: only `train.py` changed; parameter count remained 822,790; fixed 300s training budget and once-per-epoch evaluation behavior were preserved; pass.
- Improvement threshold: current baseline is 94.51% and the +0.10pp noise guard requires `best_test_acc >= 94.61%`; EXP-092 reached 94.14%, so this condition fails for improvement classification and is no-improvement.

### Informational Metrics
- final_test_acc: 94.13%
- final_test_loss: 0.2488
- training_seconds: 300.0
- total_seconds: 397.0
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 101
- num_steps: 39,173
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

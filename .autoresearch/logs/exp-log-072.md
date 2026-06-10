# EXP-072: Fan-Out Kaiming Conv Initialization

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-072.md
- **Plan**: plans/plan-072.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-072
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-072 implements the approved conv-only initialization probe. The experiment branch `autoresearch/exp-072` was created from `autoresearch/dev`, and `train.py` was the only tracked file changed. `_weights_init` now initializes `nn.Conv2d` modules with explicit fan-out ReLU Kaiming normal initialization while keeping `nn.Linear` on the existing default Kaiming normal call. A startup print marker was added so `run.log` identifies this initialization variant.

### Surprises & Discoveries

No implementation surprises. The baseline code already centralized Conv2d and Linear initialization in a single `_weights_init` helper, so the planned split required no structural changes.

### Decisions

The Linear initialization was left unchanged to isolate the Conv2d fan-out hypothesis. All EXP-064 anchor settings were preserved: `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, unit-std normalization, reflection crop padding, `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05`.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 79061 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 21:24
- **Ended**: 2026-06-09 21:31

Description:
- This run tests fan-out ReLU Kaiming initialization for convolution layers on the current EXP-064 CutMix anchor. The only intended tracked code change is in `train.py`; it changes Conv2d initialization and adds a startup marker while preserving the classifier initialization and all training hyperparameters. The hypothesis is that conv fan-out scaling will better match the residual stack and raise `best_test_acc` from 94.11% to at least the 94.21% improvement threshold.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both H20 GPUs idle; Run 1 launched on GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected anchor and variant: `Device: cuda`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `Conv init: kaiming_normal fan_out relu; Linear init: default kaiming_normal` (source: run.log L1-L4).
- Early training is producing normal progress; epoch 1 evaluation reached `test_acc: 45.77%` and GPU0 showed active utilization (source: run.log L7-L8).
- Pre-drop best reached 88.45% at epoch 46, then the first LR drop fired cleanly at step 21000 in epoch 54 (`lr: 0.0100`) (source: run.log L98-L114).
- Post-drop convergence was immediate: best rose from 91.02% at epoch 54 to 93.29% by epoch 59 (source: run.log L114-L124).
- The run completed cleanly with no traceback, CUDA OOM, non-finite, `nan`, or `inf` markers. Late post-drop accuracy peaked at 94.16% in epoch 74, then stayed below the 94.21% improvement threshold through the final epoch 102 checkpoint at 94.04%. (source: run.log L154-L221)

Key Metrics:
- `best_test_acc`: 94.16%
- `final_test_acc`: 94.04%
- `final_test_loss`: 0.2693
- `training_seconds`: 300.0
- `total_seconds`: 394.5
- `startup_seconds`: 2.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,757
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch/diff scope: passed. The run executed on `autoresearch/exp-072`; tracked code diff was limited to `train.py`. Persistent untracked `data/` was present and ignored. (source: `git status --short --branch`; `git diff --name-only`)
- Syntax/lint: passed before launch. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed.
- Experiment completion: passed. The foreground local run exited cleanly within the 10-minute wall-clock limit and wrote final summary metrics to `run.log`. (source: run.log L212-L221)
- Startup and scope markers: passed. `run.log` confirmed CUDA, ResNet-20 with 822,790 params, unchanged CutMix anchor settings, the fan-out conv initialization marker, 300s budget, and 390 batches per epoch. (source: run.log L1-L6)
- LR milestone: passed. The first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. (source: run.log L113)
- Primary metric extraction: passed. `best_test_acc` was numeric at 94.16%. (source: run.log L212)
- Improvement threshold: failed as an improvement gate, but valid as a completed no-improvement result. Baseline was 94.11% and the required threshold was 94.21%; EXP-072 reached 94.16%, a +0.05pp change that is below the noise guard.
- Hard constraints: passed. Only `train.py` was modified among tracked files; `prepare.py`, dependency files, and the evaluation harness were unchanged.

### Informational Metrics
- Final checkpoint accuracy was 94.04%, below the best checkpoint but close to the baseline plateau.
- Runtime stayed inside the hard cap: 300.0 training seconds and 394.5 total seconds.
- Peak VRAM was 660.4 MB, matching prior local CutMix runs and indicating no resource pressure.
- The run completed 102 epochs / 39,757 steps and reached the first LR drop cleanly.

## Errors & Dead Ends
- No infrastructure errors. The experimental result itself is a dead end under the goal guard: fan-out Conv2d Kaiming initialization reached 94.16%, which is higher than the 94.11 baseline but below the required 94.21% improvement threshold.

## Human Notes

> Autopilot execution; no human intervention during implementation.

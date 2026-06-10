# Plan EXP-076: Xavier Classifier Init With Zero Bias
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-076.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-076` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Split `_weights_init` so `nn.Conv2d` keeps the current `init.kaiming_normal_(m.weight)` behavior.
- [x] Change the `nn.Linear` branch to use `init.xavier_uniform_(m.weight)`.
- [x] Zero the Linear bias with `init.zeros_(m.bias)` when a bias is present.
- [x] Add a startup marker confirming `Classifier init: xavier_uniform weight, zero bias`.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged parameter count 822,790, CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`, and the classifier-init marker.
- [x] Monitor progress until the first LR drop at step 21000 is observed.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-076.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py**: Split `_weights_init` into explicit Conv2d and Linear branches. Conv2d keeps the current default Kaiming normal call so the residual/convolution stack remains identical to the EXP-064 anchor. Linear uses Xavier uniform initialization and zero bias to test a classifier-specific logit-scale calibration.
- **train.py**: Add a startup print for `Classifier init: xavier_uniform weight, zero bias` so the run log can distinguish this experiment from the anchor and from EXP-072/075.

## Configuration Changes
- Classifier Linear weight initialization: default Kaiming normal -> `init.xavier_uniform_(m.weight)`.
- Classifier Linear bias initialization: PyTorch default uniform -> `init.zeros_(m.bias)`.
- Conv2d initialization: unchanged default Kaiming normal.
- Architecture, parameter count, optimizer, schedule, CutMix, label smoothing, batch size, transforms, compile/channels-last, seed, validation cadence, and evaluation harness: unchanged.

This plan deliberately avoids the failed Conv2d fan-out family, residual-BN identity initialization, classifier dropout, label-smoothing deviations, CutMix alpha/probability brackets, and CutMix timing changes. It tests only the untried final classifier initialization mismatch identified in `train.py`.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM about 660 MB because parameter count and batch size are unchanged.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-076.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm unchanged parameter count, CutMix anchor settings, and classifier Xavier/zero-bias initialization.
- The first LR drop at step 21000 is missed due to infrastructure slowdown or premature termination.

## Verification Protocol

### Verification Procedure
1. Confirm the code-scope constraint:
   - Command: `git diff --name-only`
   - Pass condition: the only modified tracked file is `train.py`.
2. Confirm syntax and style:
   - Command: `python3 -m py_compile train.py`
   - Pass condition: command exits 0.
   - Command: `uv run ruff check train.py`
   - Pass condition: command exits 0.
3. Confirm implementation from code and log:
   - Command: inspect `git diff train.py` and `grep "Classifier init:\\|CutMix alpha:" run.log`.
   - Pass condition: Conv2d still uses default Kaiming normal; Linear uses Xavier uniform and zero bias; startup log confirms the classifier init marker; CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
4. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
5. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only classifier initialization and startup marker changes appear; `prepare.py`, evaluation, dependencies, validation frequency, seed, architecture, optimizer, LR schedule, normalization, CutMix, label smoothing, and fixed time-budget behavior remain unchanged.
6. Compare against the current baseline:
   - Baseline from experiment index: `best_test_acc=94.11%` at commit `1119ff8`.
   - Improvement threshold with +0.10 percentage-point noise guard: `best_test_acc >= 94.21%`.
   - Pass condition for improvement: final `best_test_acc` is at least `94.21%`. Any value below `94.21%`, including smaller gains over 94.11%, is `no-improvement`.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log`
- final_test_loss: `grep "^final_test_loss:" run.log`
- training_seconds: `grep "^training_seconds:" run.log`
- total_seconds: `grep "^total_seconds:" run.log`
- startup_seconds: `grep "^startup_seconds:" run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_steps: `grep "^num_steps:" run.log`
- num_params: `grep "^num_params:" run.log`

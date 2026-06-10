# Plan EXP-077: Anti-Aliased Residual Downsample
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-077.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-077` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] In `BasicBlock.__init__`, add a transition flag for residual-branch average pooling when `stride != 1`.
- [x] Set the residual branch `conv1` stride to 1 for downsample blocks and keep it unchanged for non-downsample blocks.
- [x] In `BasicBlock.forward`, average-pool the residual branch input before `conv1` only for stride-2 transition blocks.
- [x] Keep the option-A shortcut path exactly unchanged: strided slicing plus zero-channel padding.
- [x] Add a startup marker confirming `Residual downsample: avgpool before stride-2 conv`.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`, and the residual-downsample marker.
- [x] Confirm parameter count is unchanged or explain any change before interpreting the metric.
- [x] Monitor progress until the first LR drop at step 21000 is observed.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-077.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py**: Change only `BasicBlock` transition behavior. For blocks constructed with `stride != 1`, pool the residual branch input with `F.avg_pool2d(..., kernel_size=stride, stride=stride)` before the first convolution, and make that first convolution use `stride=1`. For blocks with `stride == 1`, behavior remains unchanged.
- **train.py**: Keep the option-A shortcut implementation unchanged so EXP-077 is distinct from EXP-059's failed shortcut average-pool experiment.
- **train.py**: Add a startup print for `Residual downsample: avgpool before stride-2 conv` so the run log can distinguish this architecture variant from the CutMix anchor.

## Configuration Changes
- Residual branch downsampling in transition blocks: stride-2 convolution -> average-pool by 2 followed by stride-1 convolution.
- Shortcut downsampling: unchanged strided slicing plus zero-channel padding.
- Architecture width/depth, optimizer, schedule, CutMix, label smoothing, batch size, transforms, compile/channels-last, seed, validation cadence, and evaluation harness: unchanged.

This plan deliberately avoids the failed shortcut-only average pooling, learned projection shortcuts, SE gates, full capacity changes, classifier-head tweaks, label-smoothing deviations, CutMix alpha/probability brackets, and isolated optimizer retunes. It isolates only the learned residual path's spatial downsampling.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM near the current anchor because parameter count and batch size should be unchanged.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-077.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, shape, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm unchanged CutMix anchor settings and the residual-downsample variant.
- The first LR drop at step 21000 is missed due to infrastructure slowdown or premature termination.
- Parameter count changes unexpectedly without an implementation explanation.

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
   - Command: inspect `git diff train.py` and `grep "Residual downsample:\|CutMix alpha:" run.log`.
   - Pass condition: only stride-2 residual branch downsampling changes; option-A shortcut remains strided slicing plus zero padding; startup log confirms the residual-downsample marker and CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
4. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
5. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only `train.py` changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, optimizer, LR schedule, normalization, CutMix, label smoothing, and fixed time-budget behavior remain unchanged.
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

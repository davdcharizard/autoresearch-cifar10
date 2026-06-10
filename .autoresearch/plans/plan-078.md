# Plan EXP-078: Pre-Activation BasicBlock
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-078.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-078` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Convert `BasicBlock` to a pre-activation residual block: `BN/ReLU -> conv1 -> BN/ReLU -> conv2 -> add shortcut`.
- [x] Change the first block BN to normalize `in_channels`, because it is applied before `conv1`.
- [x] Remove the final per-block ReLU after residual addition.
- [x] Keep the option-A shortcut path unchanged: strided slicing plus zero-channel padding from the original block input.
- [x] Add a final `BatchNorm2d(w3)` plus ReLU before global average pooling so the network has the standard final pre-activation output normalization.
- [x] Add a startup marker confirming `Block topology: pre-activation BasicBlock`.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`, and the pre-activation marker.
- [x] Confirm the parameter count change is the expected small pre-activation topology effect: final BN adds parameters and transition-block first BN shapes change.
- [x] Monitor progress until the first LR drop at step 21000 is observed.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-078.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py / `BasicBlock.__init__`**: replace the current post-activation BN layout with `bn1 = nn.BatchNorm2d(in_channels)` and `bn2 = nn.BatchNorm2d(out_channels)`. Keep `conv1` stride behavior and `conv2` unchanged.
- **train.py / `BasicBlock.forward`**: compute `out = F.relu(self.bn1(x))`, pass through `conv1`, then `out = self.conv2(F.relu(self.bn2(out)))`; add the original-input shortcut; return the sum without a final ReLU.
- **train.py / shortcut path**: keep the original option-A implementation exactly on the original `x`, using `shortcut[:, :, :: self.stride, :: self.stride]` and zero-channel padding.
- **train.py / `ResNet.__init__` and `forward`**: add `self.final_bn = nn.BatchNorm2d(w3)` and apply `F.relu(self.final_bn(out))` after `layer3` and before adaptive average pooling.
- **train.py / startup log**: print `Block topology: pre-activation BasicBlock`.

## Configuration Changes
- Residual block topology: post-activation BasicBlock -> pre-activation BasicBlock.
- Final output normalization: add final BN/ReLU before pooling.
- Expected parameter count: slightly different from 822,790 because transition-block first BN layers normalize `in_channels` and a final BN is added. This is an intended architecture effect, not a scope violation.
- Architecture depth/width, optimizer, LR schedule, CutMix, label smoothing, batch size, transforms, compile/channels-last, seed, validation cadence, and evaluation harness: unchanged.

This plan explicitly avoids retrying failed transition downsampling smoothing, learned projection shortcuts, SE gates, full capacity changes, classifier-head tweaks, label-smoothing deviations, CutMix alpha/probability brackets, and isolated schedule or optimizer retunes. It isolates block topology while preserving the current CutMix anchor.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM close to the current anchor because the topology adds only a small BN parameter change.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-078.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, shape, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm unchanged CutMix anchor settings and the pre-activation topology marker.
- The first LR drop at step 21000 is missed due to infrastructure slowdown or premature termination.
- Parameter count changes far beyond the expected small BN-topology effect.

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
   - Command: inspect `git diff train.py` and `grep "Block topology:\|CutMix alpha:" run.log`.
   - Pass condition: `BasicBlock` is pre-activation, option-A shortcut still uses original-input strided slicing plus zero padding, final BN/ReLU is present before pooling, startup log confirms the topology marker, and CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
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

# Plan EXP-087: Fine Upper Flip Bracket p=0.425 Under Padding 3
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-087.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-087` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Change `transforms.RandomHorizontalFlip(p=0.4)` to `transforms.RandomHorizontalFlip(p=0.425)`.
- [x] Keep `transforms.RandomCrop(32, padding=3, padding_mode="reflect")`, unit-std normalization, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR milestones, batch size, seed, compile/channels-last, fixed 300s budget, and validation cadence unchanged.
- [x] Update the startup marker to `RandomHorizontalFlip p: 0.425` so `run.log` proves EXP-087 ran the intended flip setting.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and Schedule Integrity Confirmed
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.425`, `ResNet-20 | params: 822,790`, and unchanged CutMix settings.
- [x] Confirm the first LR drop at step 21000 is reached with `lr: 0.0100`; if missed, treat the comparison as suspect.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-087.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.61%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py / transform pipeline**: Change the training transform's horizontal flip probability from 0.4 to 0.425 while keeping reflection crop padding 3.
- **train.py / startup log**: Update the flip marker to `RandomHorizontalFlip p: 0.425` so execution can verify the intended augmentation. Keep the existing `RandomCrop padding: 3 reflect` marker.

## Configuration Changes
- `RandomHorizontalFlip`: `p=0.4` -> `p=0.425`.
- All other configuration remains unchanged: reflection `RandomCrop` padding 3, unit-std normalization, `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, clean-batch label smoothing 0.05, `STAGE_WIDTHS=(28, 56, 112)`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `LR_MILESTONES=[21000, 64000]`, batch size 128, FP32 compile/channels-last, and seed 42.

This plan intentionally differs from the failed p=0.45 experiment (EXP-084): EXP-087 starts from the stronger padding-3 anchor and uses a smaller upper-side flip move. It also avoids closed crop-padding-below-3, CutMix timing/strength, LR startup, label-smoothing, batch-size, schedule-only, policy-augmentation, cutout, and broad architecture families.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM and parameter count should match the current anchor.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-087.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, shape, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.425`, and unchanged CutMix settings.
- Parameter count differs from `822,790`.
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
   - Command: inspect `git diff train.py` and `grep "RandomCrop padding:\|RandomHorizontalFlip p:\|CutMix alpha:" run.log`.
   - Pass condition: `RandomHorizontalFlip` uses `p=0.425`, startup log confirms `RandomCrop padding: 3 reflect` and `RandomHorizontalFlip p: 0.425`, and CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
4. Confirm scheduler behavior from log:
   - Command: inspect progress lines around step 21000 in `run.log`.
   - Pass condition: step 21000 switches to `lr: 0.0100`.
5. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
6. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only `train.py` changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, optimizer type, LR milestones, normalization, architecture, and fixed time-budget behavior remain unchanged.
7. Compare against the current baseline:
   - Baseline from experiment index: `best_test_acc=94.51%` at commit `83d4e94`.
   - Improvement threshold with +0.10 percentage-point noise guard: `best_test_acc >= 94.61%`.
   - Pass condition for improvement: final `best_test_acc` is at least `94.61%`. Any value below `94.61%`, including smaller gains over 94.51%, is `no-improvement`.

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

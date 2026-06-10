# Plan EXP-075: Fan-Out Conv Init Plus CutMix Endpoint Hard Labels
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-075.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-075` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Split `_weights_init` so `nn.Conv2d` uses `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`.
- [x] Keep `nn.Linear` on the current default `init.kaiming_normal_(m.weight)` behavior.
- [x] Add `CLEAN_LABEL_SMOOTHING = 0.05` and set `CUTMIX_LABEL_SMOOTHING = 0.0`.
- [x] Keep the non-CutMix branch on `CLEAN_LABEL_SMOOTHING`.
- [x] Keep both CutMix endpoint losses on `CUTMIX_LABEL_SMOOTHING`.
- [x] Add startup markers confirming `Conv init: kaiming fan_out relu for Conv2d; default kaiming for Linear`, CutMix endpoint smoothing `0.0`, and clean smoothing `0.05`.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged parameter count, the conv-init marker, CutMix `alpha=1.0`, `prob=0.5`, clean smoothing `0.05`, and CutMix endpoint smoothing `0.0`.
- [x] Monitor progress until the first LR drop at step 21000 is observed.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-075.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py**: Change `_weights_init` to use an explicit Conv2d branch with fan-out ReLU Kaiming normal initialization, while preserving the current default Kaiming normal behavior for Linear weights.
- **train.py**: Add `CLEAN_LABEL_SMOOTHING = 0.05` and set `CUTMIX_LABEL_SMOOTHING = 0.0`.
- **train.py**: Use `CUTMIX_LABEL_SMOOTHING` for both CutMix endpoint cross-entropy losses and `CLEAN_LABEL_SMOOTHING` for the clean non-CutMix loss.
- **train.py**: Print startup markers for the initialization variant and smoothing split.

## Configuration Changes
- Conv2d initialization: default Kaiming normal -> `mode="fan_out", nonlinearity="relu"` (tests EXP-072's near-miss initialization signal).
- Linear initialization: unchanged default Kaiming normal (keeps classifier behavior from the current anchor and EXP-072).
- `CUTMIX_LABEL_SMOOTHING`: `0.05` -> `0.0` for CutMix endpoint losses only (tests EXP-074's near-miss sharpening signal).
- Clean-batch label smoothing: remains `0.05`.
- `CUTMIX_ALPHA`: unchanged at `1.0`.
- `CUTMIX_PROB`: unchanged at `0.5`.
- Architecture, optimizer, schedule, batch size, transforms, compile/channels-last, seed, validation cadence, and evaluation harness: unchanged.

This plan deliberately touches a high-importance failed family, label-smoothing deviations, but it is not another isolated smoothing retry. The plan couples the narrow CutMix endpoint-only change from EXP-074 with EXP-072's independent Conv2d initialization near-miss while preserving clean-batch smoothing and the validated static CutMix exposure.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM around the existing CutMix anchor, about 660 MB, because parameter count and batch size are unchanged.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-075.md`.
- Tool skill: No remote submission skill is used. Prior infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm the conv-init variant, clean smoothing `0.05`, and CutMix endpoint smoothing `0.0`.
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
   - Command: inspect `git diff train.py` and `grep "Conv init:\\|CutMix alpha:\\|label smoothing" run.log`.
   - Pass condition: Conv2d uses fan-out ReLU Kaiming normal; Linear remains default Kaiming normal; clean batches still use smoothing 0.05; CutMix endpoint losses use smoothing 0.0; CutMix alpha/prob remain 1.0/0.5.
4. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
5. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only initialization, CutMix endpoint smoothing, clean smoothing constant, and startup marker changes appear; `prepare.py`, evaluation, dependencies, validation frequency, seed, architecture, optimizer, LR schedule, normalization, and fixed time-budget behavior remain unchanged.
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

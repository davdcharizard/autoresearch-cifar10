# Plan EXP-072: Fan-Out Kaiming Conv Initialization
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-072.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-072` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Change `_weights_init` so `nn.Conv2d` uses `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`.
- [x] Keep `nn.Linear` on the current `init.kaiming_normal_(m.weight)` path so the experiment isolates convolution fan-out scaling.
- [x] Add a startup print marker documenting the conv and linear initialization modes used in this run.
- [x] Confirm validated EXP-064 anchor settings remain unchanged: `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, unit-std normalization, reflection crop padding, `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05`.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm log startup lines include the selected device, CutMix anchor, and initialization marker.
- [x] Monitor progress until the first LR drop at step 21000 is observed, or classify as failure if the run cannot make progress.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-072.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py**: Split `_weights_init` into explicit `nn.Conv2d` and `nn.Linear` branches. Conv layers use fan-out ReLU Kaiming normal initialization; linear layers keep the current default Kaiming normal call. This directly tests whether the residual CNN benefits from conv fan-out signal scaling without changing architecture, optimizer, schedule, augmentation, throughput settings, or the evaluation harness.
- **train.py**: Add one startup print marker after the CutMix print so the log identifies EXP-072's initialization variant.

## Configuration Changes
- Conv2d initialization: `init.kaiming_normal_(m.weight)` -> `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` (tests residual-conv fan-out scaling).
- Linear initialization: unchanged `init.kaiming_normal_(m.weight)` (keeps the classifier-head initialization out of scope).
- All training hyperparameters, augmentation settings, and CutMix settings: unchanged from the EXP-064 baseline.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM similar to the 660 MB CutMix anchor because parameter count and batch size are unchanged.
- Estimated runtime: About 6-8 minutes total, including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-072.md`.
- Tool skill: No remote submission skill is used. Prior infra notes require an attached foreground launch for local CIFAR training.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, or import errors appear in the log.
- The run misses the step-21000 first LR drop due to infrastructure slowdown or premature termination; classify scientific conclusions cautiously and follow the crash/invalid path if no final metric is available.

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
3. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
4. Confirm hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only `_weights_init` and the initialization marker changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, schedule, optimizer, CutMix, augmentation, and fixed time-budget behavior remain unchanged.
5. Compare against the current baseline:
   - Baseline from experiment index: `best_test_acc=94.11%` at commit `1119ff8`.
   - Improvement threshold with +0.10 percentage-point noise guard: `best_test_acc >= 94.21%`.
   - Pass condition for improvement: final `best_test_acc` is at least `94.21%`. Any value below `94.21%`, including smaller gains over 94.11%, is `no-improvement`.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final accuracy after the fixed training budget.
- final_test_loss: `grep "^final_test_loss:" run.log` — final loss context.
- training_seconds: `grep "^training_seconds:" run.log` — confirms fixed-budget usage.
- total_seconds: `grep "^total_seconds:" run.log` — confirms the run stayed below the 10-minute cap.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — tracks soft VRAM cost.
- num_epochs: `grep "^num_epochs:" run.log` — throughput/epoch count.
- num_steps: `grep "^num_steps:" run.log` — optimization-step budget.
- num_params: `grep "^num_params:" run.log` — confirms architecture size unchanged.

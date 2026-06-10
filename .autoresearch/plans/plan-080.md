# Plan EXP-080: Very Short Linear LR Warmup
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-080.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-080` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Add `LR_WARMUP_START = 0.02` and `LR_WARMUP_STEPS = 500` near the existing optimizer/LR constants.
- [x] Add a small helper or inline expression that returns a linear LR from 0.02 at step 0 to 0.1 at step 500, then keeps the normal scheduled LR path.
- [x] Set the optimizer LR before each batch through step 500 without changing `LR=0.1`, `LR_MILESTONES=[21000, 64000]`, `scheduler.step()` placement, or post-warmup scheduler behavior.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, clean-batch label smoothing 0.05, `STAGE_WIDTHS=(28, 56, 112)`, `WEIGHT_DECAY=2e-4`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Add a startup marker confirming the warmup, for example `LR warmup: 0.02 -> 0.1 over 500 steps`.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged CutMix settings, and the LR-warmup marker.
- [x] Confirm parameter count remains `822,790`.
- [x] Confirm early progress lines show LR rising during the first 500 steps and then reaching `0.1000`.
- [x] Monitor progress until the first LR drop at step 21000 is observed; if the run misses this milestone, treat the scientific comparison as suspect.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-080.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py / constants**: Add `LR_WARMUP_START = 0.02` and `LR_WARMUP_STEPS = 500` while preserving `LR = 0.1` as the post-warmup base LR.
- **train.py / helper**: Add a deterministic warmup helper, for example `current_warmup_lr(step)`, that returns `LR_WARMUP_START + (LR - LR_WARMUP_START) * step / LR_WARMUP_STEPS` through `step == LR_WARMUP_STEPS` and `None` after warmup.
- **train.py / training loop**: Before the optimizer update for each batch, if the helper returns a value, assign it to every optimizer param group's `lr`. Keep `optimizer.step()` followed by `scheduler.step()` exactly where it is now so the existing 21k drop still occurs.
- **train.py / startup log**: Print a clear marker for the warmup parameters so the execution log can verify this is EXP-080 and not the static anchor.

## Configuration Changes
- Initial LR behavior: static `0.1` becomes a 500-step linear warmup from `0.02` to `0.1`, then the existing `MultiStepLR` behavior takes over.
- LR milestones, CutMix alpha/probability/label smoothing, architecture, optimizer type, momentum, weight decay, transforms, compile/channels-last, batch size, seed, validation cadence, and evaluation harness: unchanged.

This plan intentionally differs from failed schedule approaches. EXP-040 and EXP-043 changed the full-run initial LR away from 0.1; EXP-080 restores 0.1 by step 500. EXP-003, EXP-024, EXP-030, EXP-046, and EXP-052 changed late schedule behavior; EXP-080 preserves the 21k first drop and flat 0.01 post-drop tail. It also differs from EXP-073 and EXP-079 because it does not reduce or delay CutMix exposure.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM and parameter count should match the current anchor.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-080.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, shape, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm unchanged CutMix anchor settings and the LR-warmup marker.
- Parameter count differs from `822,790`.
- Progress lines do not show LR reaching `0.1000` by or shortly after step 500.
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
   - Command: inspect `git diff train.py` and `grep "CutMix alpha:\|LR warmup:" run.log`.
   - Pass condition: the LR warmup is present, startup log confirms `0.02 -> 0.1 over 500 steps`, and CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
4. Confirm scheduler behavior from log:
   - Command: inspect progress lines around steps 50, 500, and 21000 in `run.log`.
   - Pass condition: early LR is below 0.1, LR reaches `0.1000` after warmup, and step 21000 switches to `lr: 0.0100`.
5. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
6. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only `train.py` changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, optimizer type, LR milestones, normalization, architecture, and fixed time-budget behavior remain unchanged.
7. Compare against the current baseline:
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

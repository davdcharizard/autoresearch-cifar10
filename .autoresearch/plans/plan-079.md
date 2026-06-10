# Plan EXP-079: Short CutMix Probability Ramp
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-079.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-079` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Add `CUTMIX_PROB_START = 0.25` and `CUTMIX_PROB_RAMP_STEPS = 1000` near the existing CutMix constants.
- [x] Add a small `current_cutmix_prob(step)` helper or equivalent inline expression that returns a linear ramp from 0.25 at step 0 to 0.5 at step 1000, then stays at `CUTMIX_PROB`.
- [x] Replace the static `torch.rand(...) < CUTMIX_PROB` sampling check with `torch.rand(...) < cutmix_prob`, where `cutmix_prob` is computed from the current optimizer-step count before the batch update.
- [x] Keep `CUTMIX_ALPHA=1.0`, final `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, clean-batch label smoothing 0.05, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, compile/channels-last, and batch size 128 unchanged.
- [x] Add a startup marker confirming the ramp, for example `CutMix prob ramp: 0.25 -> 0.5 over 1000 steps`.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup lines include selected device, unchanged CutMix alpha/prob/smoothing, and the probability-ramp marker.
- [x] Confirm parameter count remains `822,790`.
- [x] Monitor progress until the first LR drop at step 21000 is observed; if the run misses this milestone, treat the scientific comparison as suspect.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-079.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py / constants**: Add `CUTMIX_PROB_START = 0.25` and `CUTMIX_PROB_RAMP_STEPS = 1000` while preserving the existing `CUTMIX_PROB = 0.5` as the long-run target.
- **train.py / helper**: Add a deterministic probability schedule:
  - if `step >= CUTMIX_PROB_RAMP_STEPS`, return `CUTMIX_PROB`;
  - otherwise return `CUTMIX_PROB_START + (CUTMIX_PROB - CUTMIX_PROB_START) * step / CUTMIX_PROB_RAMP_STEPS`.
- **train.py / training loop**: Before sampling CutMix, compute `cutmix_prob = current_cutmix_prob(step)` and use that value in the Bernoulli check. Keep the existing CutMix box sampling, batch permutation, area-adjusted lambda, endpoint losses, and clean-batch loss unchanged.
- **train.py / startup log**: Print a clear marker for the ramp parameters so the execution log can verify this is EXP-079 and not the static anchor.

## Configuration Changes
- CutMix probability schedule: static `0.5` becomes a 1000-step linear ramp from `0.25` to `0.5`, then static `0.5`.
- CutMix alpha, label smoothing, architecture, optimizer, LR schedule, transforms, compile/channels-last, batch size, seed, validation cadence, and evaluation harness: unchanged.

This plan explicitly differs from prior failed CutMix variants: EXP-065 used `p=0.25` for the full run, EXP-066 used `p=0.75` for the full run, EXP-069 weakened CutMix after the first LR drop, and EXP-073 fully disabled CutMix for 2000 steps. EXP-079 keeps CutMix active from the start and restores the validated `p=0.5` well before the first LR drop.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM and parameter count should match the current anchor.
- Estimated runtime: About 6-7 minutes total including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-079.md`.
- Tool skill: No remote submission skill is used. Infrastructure notes require attached foreground launches for local CIFAR runs; detached/nohup launches are not trusted.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, import, shape, NaN, or non-finite-loss errors appear in the log.
- Startup markers do not confirm unchanged CutMix anchor settings and the probability-ramp marker.
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
   - Command: inspect `git diff train.py` and `grep "CutMix alpha:\|CutMix prob ramp:" run.log`.
   - Pass condition: the probability ramp is present, startup log confirms `0.25 -> 0.5 over 1000 steps`, and CutMix alpha/prob/smoothing remain 1.0/0.5/0.05.
4. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
5. Confirm the hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only `train.py` changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, optimizer, LR schedule, normalization, architecture, and fixed time-budget behavior remain unchanged.
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

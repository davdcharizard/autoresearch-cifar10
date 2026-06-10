# Plan EXP-073: Short Clean Warmup Before CutMix
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-073.md

## Milestones

### Milestone 1: Code Change Implemented and Locally Checked
- [x] Create branch `autoresearch/exp-073` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Add `CUTMIX_WARMUP_STEPS = 2000` near the existing CutMix constants.
- [x] Keep `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05` unchanged.
- [x] Change the training-loop CutMix sampling so the first 2000 updates are clean label-smoothed batches, then the existing probabilistic CutMix branch is enabled for all later updates.
- [x] Add startup and one-time runtime markers documenting the warmup schedule and the first enabled CutMix step.
- [x] Confirm validated anchor settings remain unchanged: `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, unit-std normalization, reflection crop padding, compile/channels-last, batch size 128, and full-run label smoothing 0.05.
- [x] Verify `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment Launched and First LR Drop Reached
- [x] Check GPU availability with `nvidia-smi` and select one free H20-class GPU.
- [x] Remove stale `run.log` before launch.
- [x] Launch the experiment in a foreground attached shell with `env CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm log startup lines include the selected device, CutMix anchor, and `CutMix warmup steps: 2000`.
- [x] Confirm the one-time `CutMix enabled after 2000 clean steps` marker appears before the first LR drop.
- [x] Monitor progress until the first LR drop at step 21000 is observed.

### Milestone 3: Final Metrics Captured and Classified
- [x] Let the run finish unless an abort criterion triggers.
- [x] Extract final metric lines from `run.log`.
- [x] Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` in `logs/exp-log-073.md`.
- [x] Classify as `improvement` only if `best_test_acc >= 94.21%`; classify smaller gains or regressions as `no-improvement`.

## Code Changes
- **train.py**: Add `CUTMIX_WARMUP_STEPS = 2000` to define a short clean warmup before the validated CutMix branch can activate.
- **train.py**: Replace the direct `apply_cutmix = torch.rand((), device=device).item() < CUTMIX_PROB` decision with a gated decision: `cutmix_enabled = step >= CUTMIX_WARMUP_STEPS`, then sample CutMix with probability 0.5 only if enabled. This makes updates 1-2000 clean and enables CutMix from update 2001 onward.
- **train.py**: Add a startup print for the warmup setting and a one-time runtime print when CutMix becomes enabled, so analysis can verify the temporal schedule from `run.log`.

## Configuration Changes
- `CUTMIX_WARMUP_STEPS`: new constant set to `2000` clean updates. This is about 5.1 epochs and less than 10% of the pre-drop window, intended to reduce only earliest mixed-label noise while preserving the step-21000 LR drop and the post-drop CutMix anchor.
- `CUTMIX_ALPHA`: unchanged at `1.0`.
- `CUTMIX_PROB`: unchanged at `0.5` after warmup.
- `CUTMIX_LABEL_SMOOTHING`: unchanged at `0.05`.
- Architecture, optimizer, schedule, batch size, normalization, augmentation, compile/channels-last, and evaluation: unchanged.

## Execution Environment
- Method: Local single-GPU foreground run from the project root.
- Resources: One available NVIDIA H20-class GPU; expected VRAM similar to the 660 MB CutMix anchor because parameter count and batch size are unchanged.
- Estimated runtime: About 6-7 minutes total, including startup and validation; kill if total wall time exceeds 10 minutes.
- Log output: Capture stdout/stderr to `run.log`; write structured observations and final metrics to `.autoresearch/logs/exp-log-073.md`.
- Tool skill: No remote submission skill is used. Prior infra notes require attached foreground launches for local CIFAR training.

## Abort Criteria
- The process crashes or exits before reporting a numeric `best_test_acc`.
- `run.log` remains empty or shows no training progress for more than 2 minutes after launch.
- The run exceeds 10 minutes total wall-clock time.
- CUDA, dataset, compile, or import errors appear in the log.
- The warmup marker does not appear, or the first LR drop at step 21000 is missed due to infrastructure slowdown or premature termination.

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
3. Confirm warmup implementation from the log:
   - Command: `grep "CutMix warmup steps:\\|CutMix enabled after" run.log`
   - Pass condition: output includes `CutMix warmup steps: 2000` and a one-time enabled marker at or just after step 2000.
4. Confirm the run completed and produced the primary metric:
   - Command: `grep "^best_test_acc:\\|^peak_vram_mb:" run.log`
   - Pass condition: output includes a numeric `best_test_acc`.
5. Confirm hard constraints:
   - Command: inspect `git diff train.py` and startup log lines.
   - Pass condition: only CutMix warmup gating and markers changed; `prepare.py`, evaluation, dependencies, validation frequency, seed, architecture, optimizer, LR schedule, normalization, and fixed time-budget behavior remain unchanged.
6. Compare against the current baseline:
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

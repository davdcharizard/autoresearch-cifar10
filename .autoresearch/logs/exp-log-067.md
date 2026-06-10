# EXP-067: CutMix Alpha 0.5

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md
- **Plan**: plans/plan-067.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-067
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-067` from `autoresearch/dev` and implemented the planned one-scalar alpha bracket by changing only `CUTMIX_ALPHA` in `train.py` from `1.0` to `0.5`. The validated `CUTMIX_PROB=0.5` setting, endpoint label smoothing, architecture, optimizer, LR schedule, transform stack, compile/channels-last path, and evaluation cadence were left unchanged.

### Surprises & Discoveries

No implementation surprises. The current CutMix implementation already exposes alpha as a top-level constant consumed by the beta distribution.

### Decisions

Kept the experiment to a one-line hyperparameter change to preserve clean attribution after EXP-065/066 closed the probability bracket. Preflight confirmed the tracked diff is limited to `train.py`; `python3 -m py_compile train.py` and `uv run ruff check train.py` both passed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 91897; shell PID 3691771, uv PID 3691772, main Python PID 3691776
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 20:15
- **Ended**: 2026-06-09 20:22 UTC

Description:
- Run the one-scalar `CUTMIX_ALPHA=0.5` bracket locally on a single available GPU using the fixed CIFAR-10 training harness. This tests whether a higher-variance CutMix patch-area distribution can improve the current `p=0.5` CutMix anchor. The required improvement threshold is `best_test_acc >= 94.21%`; output is captured to `run.log` for startup markers, LR milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 0.5, prob: 0.5, label smoothing: 0.05`, 300s budget, and 390 batches per epoch. GPU0 was selected; `/proc/3691776/cwd` confirmed the main Python process is in this project root. (source: run.log L1-L5; `/proc/3691776/cwd`)
- First LR drop reached as expected at `step 21000 ep 54` with `lr: 0.0100`; no traceback, OOM, or non-finite-loss markers were present during monitoring. Post-drop accuracy climbed from 91.41% at epoch 54 to 93.94% by epoch 68. (source: run.log around `step 21000`; eval lines ep 54-68)
- Run completed cleanly within the 10 minute hard limit. The best accuracy peaked at 94.07% at epoch 77 and the final summary reported `best_test_acc: 94.07%`, which is below the 94.21% improvement threshold. Verdict: valid no-improvement; baseline remains 94.11%. (source: run.log final summary)

Key Metrics:
- `best_test_acc`: 94.07%
- `final_test_acc`: 93.65%
- `final_test_loss`: 0.2739
- `training_seconds`: 300.0
- `total_seconds`: 394.7
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39006
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch/diff: PASS. `git status --short --branch` showed `autoresearch/exp-067`; tracked code diff was limited to `train.py` with the single intended `CUTMIX_ALPHA = 0.5` change. Persistent untracked `data/` ignored.
- Syntax/lint: PASS. `python3 -m py_compile train.py` exited 0; `uv run ruff check train.py` reported all checks passed.
- Run completion: PASS. Foreground process exited cleanly; final summary metrics were present in `run.log`.
- Startup/scope markers: PASS. Log showed CUDA, ResNet-20 with 822,790 params, `CutMix alpha: 0.5, prob: 0.5, label smoothing: 0.05`, 300s budget, and 390 batches per epoch.
- LR milestone: PASS. `grep "step 21000" run.log` showed `step 21000 ep 54` with `lr: 0.0100`.
- Primary metric extraction: PASS. `grep "^best_test_acc:" run.log` returned `94.07%`.
- Improvement threshold: FAIL for improvement, PASS for no-improvement classification. `94.07% < 94.21%`, so EXP-067 does not count as an improvement under the 0.10pp noise guard.
- Hard constraints: PASS. `git diff --name-only` listed only `train.py`.

### Informational Metrics
- Final summary: `best_test_acc=94.07%`, `final_test_acc=93.65%`, `final_test_loss=0.2739`, `training_seconds=300.0`, `total_seconds=394.7`, `startup_seconds=2.4`, `peak_vram_mb=660.4`, `num_epochs=101`, `num_steps=39006`, `num_params=822,790`.

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.

# EXP-069: Post-Drop CutMix Probability Taper to 0.25

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md
- **Plan**: plans/plan-069.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-069
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-069` from `autoresearch/dev` and implemented the planned post-drop CutMix probability taper in `train.py`. The baseline `CUTMIX_ALPHA=1.0`, pre-drop `CUTMIX_PROB=0.5`, endpoint label smoothing, architecture, optimizer, LR milestones, reflection padding, compile/channels-last path, and validation cadence were preserved. The change adds `CUTMIX_POST_DROP_PROB=0.25`, ties `CUTMIX_TAPER_STEP` to `LR_MILESTONES[0]`, uses the reduced probability once `step >= CUTMIX_TAPER_STEP`, and prints startup plus one-time taper markers for log verification.

### Surprises & Discoveries

No implementation surprises. The training loop already has the global `step` available before CutMix sampling, so the schedule can be applied without touching the optimizer, scheduler, dataloader, or evaluator.

### Decisions

Kept the taper step tied to `LR_MILESTONES[0]` rather than duplicating a magic value in loop logic. Added an explicit runtime marker at the first post-drop tapered update so analysis can verify the schedule from `run.log` in addition to the code diff.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 19663; shell PID 3734524, uv PID 3734525, main Python PID 3734528
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 20:42 UTC
- **Ended**: 2026-06-09 20:49 UTC

Description:
- Run the one-file CutMix schedule experiment locally on a single available GPU using the fixed CIFAR-10 training harness. This preserves the EXP-064 `alpha=1.0, p=0.5` anchor before the first LR drop, then lowers CutMix probability to 0.25 for post-drop refinement. The required improvement threshold is `best_test_acc >= 94.21%`; output is captured to `run.log` for startup markers, LR/taper milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, `CutMix post-drop prob: 0.25 after step 21000`, 300s budget, and 390 batches per epoch. GPU0 was selected; `/proc/3734528/cwd` confirmed the main Python process is in this project root. (source: run.log L1-L6; `/proc/3734528/cwd`)
- First LR drop reached at step 21000 in epoch 54 with LR 0.0100, and the one-time `CutMix probability tapered to 0.25 after step 21000` marker printed immediately afterward. Post-drop accuracy climbed from 91.94% at epoch 54 to 93.72% by epoch 63, with no traceback, CUDA OOM, or non-finite markers during monitoring. (source: run.log L113-L136)
- Run exited cleanly with final `best_test_acc=93.73%`, which is below the 94.11% baseline and below the required 94.21% improvement threshold. Final checkpoint accuracy was 92.94%, confirming late post-drop tapering did not produce a competitive final plateau. (source: run.log L214-L223)

Key Metrics:
- `best_test_acc`: 93.73%
- `final_test_acc`: 92.94%
- `final_test_loss`: 0.2647
- `training_seconds`: 300.0
- `total_seconds`: 394.5
- `startup_seconds`: 2.3
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39606
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch and diff scope: passed. Branch was `autoresearch/exp-069`; `git diff --name-only` listed only `train.py`; persistent untracked `data/` was ignored. (source: `git status --short --branch`; `git diff --name-only`)
- Startup and scope markers: passed. CUDA, ResNet-20 with 822,790 params, baseline CutMix alpha/prob/smoothing, post-drop probability 0.25, 300s budget, and 390 batches/epoch all appeared in `run.log`. (source: run.log L1-L6)
- LR/taper milestone: passed. `run.log` showed the first LR drop at step 21000 with LR 0.0100 and printed the CutMix taper marker immediately afterward. (source: run.log L113-L114)
- Primary metric extraction: passed. `best_test_acc: 93.73%` was present in the final summary. (source: run.log L214)
- Improvement threshold: failed for improvement. Current baseline is 94.11%, and the goal requires at least +0.10pp, so EXP-069 needed `best_test_acc >= 94.21%`; observed 93.73% is a valid no-improvement result. (source: run.log L214; goal/index threshold)
- Hard constraints: passed. Only `train.py` was modified, the run used one GPU, training budget was 300.0 seconds, and architecture parameter count stayed 822,790. (source: `git diff --name-only`; run.log L217-L223)

### Informational Metrics
- Best accuracy emerged at epoch 89 and then degraded through the final checkpoint, suggesting the post-drop taper failed to stabilize late refinement. (source: run.log L188-L223)

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.

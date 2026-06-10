# EXP-019: Minimal Width Step 29/58/116 with 19k First LR Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-019
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-019 implements the planned minimal width/schedule retune in `train.py` only. `STAGE_WIDTHS` changed from `(28, 56, 112)` to `(29, 58, 116)`, and `LR_MILESTONES` changed from `[21000, 64000]` to `[19000, 64000]`. The rest of the training recipe, including depth, optimizer, augmentation, seed, batch size, compile/channels-last settings, fixed time budget, and once-per-epoch validation, was preserved.

### Surprises & Discoveries

The implementation was only a two-constant diff, and preflight confirmed no incidental changes to evaluation cadence or training flow.

### Decisions

No deviations from the plan were needed. The first LR drop was set to 19000 rather than 20000 to give the slightly wider model more LR 0.01 refinement time than EXP-017 had.

## Experimental Adjustments

- **Use physical GPU 1 for launch**: Pre-launch `nvidia-smi` showed GPU 0 active and GPU 1 idle, so the run uses `CUDA_VISIBLE_DEVICES=1` to preserve single-GPU isolation without sharing GPU 0. (ref: pre-launch GPU check, 2026-06-08 17:18 UTC)

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 73445; shell PID 1041272; uv PID 1041273; main Python PID 1041277
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 17:19 UTC
- **Ended**: 2026-06-08 17:26 UTC

Description:
- Run a 29/58/116 ResNet-20 with a 19k first LR drop under the fixed 300s training budget. This tests whether the validated width-scaling axis has a smaller usable increment after the broader 30/60/120 step lost too much throughput. The run must complete on one GPU, keep validation once per epoch, reach the 19k drop, and achieve at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Preflight passed: `python3 -m py_compile train.py` exited 0, `uv run ruff check train.py` reported all checks passed, `git diff -- train.py` showed only the `STAGE_WIDTHS` and `LR_MILESTONES` constant changes, and `rg -n "evaluator\\.evaluate|Eval\\(" train.py` showed one `Eval()` construction plus one epoch-level evaluate call.
- Startup confirmed on CUDA with `ResNet-20 | params: 882,451`, 300s training budget, and 390 batches per epoch. (source: run.log L1-L4)
- The first LR drop was reached at step 19000 during epoch 49 with 136s training budget remaining. Post-drop accuracy reached 92.59% by epoch 73, below the 93.33% threshold. (source: run.log L103-L152)
- The run completed normally before the 10-minute wall-clock cap with `training_seconds=300.0` and `total_seconds=390.2`. Final `best_test_acc` remained 92.59%, so EXP-019 is a valid no-improvement result against the 93.33% threshold.

Key Metrics:
- `best_test_acc`: 92.59%
- `final_test_acc`: 92.44%
- `final_test_loss`: 0.3267
- `training_seconds`: 300.0
- `total_seconds`: 390.2
- `startup_seconds`: 2.9
- `peak_vram_mb`: 691.8
- `num_epochs`: 93
- `num_steps`: 36139
- `num_params`: 882,451

## Verification Results

### Conditions Checked
- [x] Single GPU selected: launched with `CUDA_VISIBLE_DEVICES=1`; PyTorch preflight saw exactly one selected CUDA device.
- [x] Process exited successfully before 10 minutes total wall-clock: run completed with `total_seconds=390.2`.
- [x] Numeric primary metric reported: `best_test_acc=92.59%`.
- [x] Fixed training budget preserved: `training_seconds=300.0`.
- [x] First LR drop reached: log shows `step 19000 ... lr: 0.0100`.
- [x] Validation cadence preserved: one `Eval()` construction and one epoch-level `evaluator.evaluate(...)` call remain in `train.py`.
- [x] Scope preserved during run: tracked diff was limited to `train.py`; untracked `data/` was preserved.
- [ ] Improvement threshold met: failed because 92.59% is below the required 93.33% threshold (`93.23 + 0.10`).

### Informational Metrics
- EXP-019 completed 93 epochs and 36,139 optimizer steps, fewer than the 28/56/112 baseline family, and did not recover enough accuracy after the 19k LR drop.

## Errors & Dead Ends
- No crash, OOM, NaN, protected-file change, or evaluation-harness issue occurred.

## Human Notes

> No human intervention during autopilot execution.

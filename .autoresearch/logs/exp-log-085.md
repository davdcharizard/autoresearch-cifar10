# EXP-085: Crop Padding 3 on Flip p=0.4 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-085.md
- **Plan**: plans/plan-085.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-085
- **Commit**: 83d4e947d2b0781af6b227376aab0637ae1e7aed
- **PR**: N/A (no git remote configured in this local checkout)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved crop-strength interaction on the current EXP-082 spatial anchor. `train.py` changes only the training crop transform from `padding=4` to `padding=3` while preserving `RandomHorizontalFlip(p=0.4)`, and adds the startup marker `RandomCrop padding: 3 reflect`. All other augmentation, CutMix, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target crop transform was explicit and localized, and the existing flip marker made it straightforward to add the crop marker beside it.

### Decisions

Kept EXP-085 isolated to crop padding on the validated flip p=0.4 anchor, as planned. No coupled changes to CutMix, schedule, optimizer, normalization, or architecture were added, preserving a clean comparison against the 94.36% baseline and the 94.46% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 64922; launcher PID 4080636; uv PID 4080637; train PID 4080640
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 00:32 UTC
- **Ended**: 2026-06-10 00:39 UTC

Description:
- Local single-GPU foreground run of EXP-085 using the current CutMix anchor with `RandomHorizontalFlip(p=0.4)` preserved and reflection crop padding reduced from 4 to 3. This tests whether EXP-081's padding-3 near miss becomes useful on the validated p=0.4 flip anchor. The run is expected to preserve throughput, parameter count, the step-21000 LR drop, and fixed wall-clock training behavior while attempting to reach at least 94.46% `best_test_acc`.

Observations:
- GPU0 was selected because both H20 GPUs were free and `nvidia-smi` showed no running processes before launch. (source: `nvidia-smi` pre-launch output, 2026-06-10 00:31 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached on schedule at step 21000 with `lr: 0.0100`; pre-drop best was 88.43% at epoch 41 and still 88.43% through epoch 53. (source: run.log L89, L113-L115)
- Post-drop convergence crossed the improvement threshold at epoch 74 with `test_acc=94.51%`, above the required 94.46% noise-guard threshold. (source: run.log L153-L157)
- The run completed cleanly at the fixed 300s training budget with final epoch 102 and final `best_test_acc=94.51%`. No crash, CUDA, shape, NaN, or non-finite-loss signatures were found. (source: run.log L211-L222; error scan with `rg` returned no matches)

Key Metrics:
- `best_test_acc`: 94.51% (improvement; +0.15pp over 94.36% baseline and +0.05pp over the 94.46% threshold)
- `final_test_acc`: 94.10%
- `final_test_loss`: 0.2661
- `training_seconds`: 300.0
- `total_seconds`: 394.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,685
- `num_params`: 822,790
- Source: run.log L213-L222

## Verification Results

### Conditions Checked

1. Code-scope constraint: PASS. `git diff --name-only` listed only `train.py`.
2. Syntax and style: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
3. Implementation from code and log: PASS. `git diff train.py` shows only `RandomCrop(... padding=3 ...)` plus the startup marker; startup log confirms `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, and unchanged CutMix alpha/prob/smoothing. (source: `git diff train.py`; run.log L1-L5)
4. Scheduler behavior: PASS. Step 21000 switched to `lr: 0.0100`. (source: run.log L114)
5. Run completion and primary metric: PASS. Final summary includes numeric `best_test_acc: 94.51%`. (source: run.log L213)
6. Hard constraints: PASS. Only `train.py` changed; parameter count stayed 822,790; fixed 300s training budget, validation cadence, seed, optimizer, LR milestones, normalization, architecture, and CutMix settings were preserved by diff/log inspection. (source: `git diff --name-only`; `git diff train.py`; run.log L1-L7, L216-L222)
7. Baseline comparison: PASS. Baseline was 94.36% at commit `e859ac5`; the explicit +0.10pp threshold is 94.46%; EXP-085 reached 94.51%, so verdict is `improvement`.

### Informational Metrics

- `final_test_acc`: 94.10%
- `final_test_loss`: 0.2661
- `training_seconds`: 300.0
- `total_seconds`: 394.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,685
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.

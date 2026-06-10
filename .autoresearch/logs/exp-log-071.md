# EXP-071: CIFAR AutoAugment on CutMix Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-071.md
- **Plan**: plans/plan-071.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-071
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-071` from `autoresearch/dev` and implemented the planned CIFAR AutoAugment probe in `train.py`. The training transform now applies `transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)` after reflection crop and horizontal flip, before tensor conversion and unit-std normalization. A startup print was added so `run.log` verifies the augmentation policy; the EXP-064 CutMix anchor, architecture, optimizer, LR milestones, reflection crop padding, label smoothing, compile/channels-last path, and validation cadence were preserved.

### Surprises & Discoveries

No implementation surprises. A pre-plan API check confirmed this torchvision version exposes both `transforms.AutoAugment` and `transforms.AutoAugmentPolicy.CIFAR10`, so no import or dependency changes were needed.

### Decisions

Kept AutoAugment in the PIL-image portion of the transform pipeline, after geometric crop/flip and before `ToTensor()`, matching torchvision's expected usage. Added a startup marker rather than a new config flag to keep the experiment's code diff minimal and auditable.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 69141; shell PID 3774909, uv PID 3774910, main Python PID 3774913
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 21:09 UTC
- **Ended**: 2026-06-09 21:17 UTC

Description:
- Run the one-file policy-augmentation experiment locally on a single available GPU using the fixed CIFAR-10 training harness. This preserves unit-std normalization and the EXP-064 CutMix anchor while adding CIFAR AutoAugment to the training transform. The required improvement threshold is `best_test_acc >= 94.21%`; output is captured to `run.log` for startup markers, LR milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, `Train augmentation: CIFAR AutoAugment`, 300s budget, and 390 batches per epoch. GPU1 was selected because GPU0 was occupied by another repo's run; `/proc/3774913/cwd` confirmed the main Python process is in this project root. (source: run.log L1-L6; `/proc/3774913/cwd`)
- First LR drop was reached at `step 21000 ep 54` with `lr: 0.0100`, so AutoAugment overhead did not prevent the anchor schedule transition. Pre-drop best was 87.50% at epoch 48; post-drop refinement reached 92.59% by epoch 58. No error markers were observed at this checkpoint. (source: run.log L102-L122)
- The run completed cleanly with no traceback, CUDA OOM, non-finite, `nan`, or `inf` markers. Late post-drop accuracy plateaued below the active threshold: best reached 93.58% at epoch 72, briefly improved to 93.62% at epoch 101, and ended at 93.35%. (source: run.log L150-L221)

Key Metrics:
- `best_test_acc`: 93.62%
- `final_test_acc`: 93.35%
- `final_test_loss`: 0.2809
- `training_seconds`: 300.0
- `total_seconds`: 457.7
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,585
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch/diff scope: passed. The run executed on `autoresearch/exp-071`; tracked code diff was limited to `train.py`. Persistent untracked `data/` was present and ignored. (source: `git status --short --branch`; `git diff --name-only`)
- Syntax/lint: passed before launch. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed.
- Experiment completion: passed. The foreground local run exited cleanly within the 10-minute wall-clock limit and wrote final summary metrics to `run.log`. (source: run.log L212-L221)
- Startup and scope markers: passed. `run.log` confirmed CUDA, ResNet-20 with 822,790 params, unchanged CutMix anchor settings, CIFAR AutoAugment active, 300s budget, and 390 batches per epoch. (source: run.log L1-L6)
- LR milestone: passed. The first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. (source: run.log L113)
- Primary metric extraction: passed. `best_test_acc` was numeric at 93.62%. (source: run.log L212)
- Improvement threshold: failed as an improvement gate, but valid as a completed no-improvement result. Baseline was 94.11% and the required threshold was 94.21%; EXP-071 reached 93.62%.
- Hard constraints: passed. Only `train.py` was modified among tracked files; `prepare.py`, dependency files, and the evaluation harness were unchanged.

### Informational Metrics
- Final checkpoint accuracy was 93.35%, below the best checkpoint but much closer than EXP-064's peak-to-final gap.
- Runtime stayed inside the hard cap: 300.0 training seconds and 457.7 total seconds, with AutoAugment adding substantial validation/transform overhead relative to lighter runs.
- Peak VRAM was 660.4 MB, matching prior local CutMix runs and indicating no resource pressure.
- The run completed 102 epochs / 39,585 steps, slightly fewer steps than the CutMix anchor but still reached the first LR drop.

## Errors & Dead Ends
- No infrastructure errors. The experimental result itself is a dead end: CIFAR AutoAugment stacked on the CutMix anchor underperformed the 94.11% baseline and did not approach the 94.21% improvement threshold.

## Human Notes

> Autopilot execution; no human intervention during implementation.

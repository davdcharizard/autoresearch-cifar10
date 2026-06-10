# EXP-031: Symmetric Padding for RandomCrop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-031
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-031 exactly as planned by changing only the CIFAR training crop padding mode in `train.py` from `padding_mode="reflect"` to `padding_mode="symmetric"`. The 28/56/112 ResNet-20 width, batch size, optimizer settings, LR schedule, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, torchvision transform construction, validation cadence, symmetric padding presence, and schedule preservation.

### Surprises & Discoveries

No implementation surprises. The installed torchvision accepts `padding_mode="symmetric"` for `transforms.RandomCrop`, so the experiment remains a one-line augmentation-boundary change.

### Decisions

No deviations from the plan were needed. Kept all non-padding settings unchanged to isolate symmetric versus reflected crop-boundary fill.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 57121
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 20:23 UTC
- **Ended**: 2026-06-08 20:30 UTC

Description:
- Run the EXP-029 reflection-padding 28/56/112 ResNet-20 anchor with only the crop padding mode changed to symmetric. The experiment tests whether symmetric edge mirroring preserves the no-zero-border benefit while improving crop-boundary statistics enough to raise `best_test_acc`. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.68%` to count as an improvement over the current `93.58%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no error patterns; best test accuracy reached 81.91% by epoch 10 while GPU 0 was active. (source: run.log L6-L24)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; the run remained on the preserved 21k/64k schedule. (source: run.log L111)
- Post-drop accuracy climbed to 93.48% by epoch 73 and remained below the 93.68% improvement threshold through epoch 99. No crash, OOM, NaN/Inf, or traceback pattern was observed in the monitored log slice. (source: run.log L144-L202)
- The run exited cleanly with `best_test_acc: 93.48%`, `total_seconds: 397.1`, `num_steps: 43464`, and `num_params: 822,790`; the preserved second LR milestone at step 64000 was not reached. This is a valid no-improvement result because it is below both the 93.58% baseline and the 93.68% improvement threshold. (source: run.log L230-L239)

Key Metrics:
- best_test_acc: 93.48%
- final_test_acc: 92.93%
- final_test_loss: 0.2925
- training_seconds: 300.0
- total_seconds: 397.1
- startup_seconds: 2.9
- peak_vram_mb: 660.4
- num_epochs: 112
- num_steps: 43464
- num_params: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. The experiment index reported `baseline=93.58`, so EXP-031 required `best_test_acc >= 93.68` under the goal's +0.10 percentage-point improvement rule.
- Scope before launch: passed. The only tracked source diff during the run was `train.py`, with `.autoresearch/` local-only and `data/` untracked.
- Syntax, lint, and transform validity: passed. `python3 -m py_compile train.py`, `uv run ruff check train.py`, and the torchvision `RandomCrop(..., padding_mode="symmetric")` construction check all exited 0.
- Validation cadence, augmentation, and preserved schedule: passed. The training file retained one `Eval()` construction, one epoch-level `evaluator.evaluate(...)` call, the symmetric padding mode, and `LR_MILESTONES = [21000, 64000]`.
- Preserved batch size, schedule behavior, and parameter count: passed. `run.log` reported `Batches per epoch: 390`; step 21000 had `lr: 0.0100`; step 64000 was absent; `num_params` was `822,790`.
- Experiment completion: passed. The process exited 0 before the 10-minute wall-clock cap and printed numeric summary metrics.
- Metric improvement: failed. `best_test_acc` was 93.48%, below the 93.68% improvement threshold. This makes the verdict no-improvement, not invalid, because all process and scope constraints passed.
- Hard constraints: passed. The fixed 300s training budget was used, total runtime was 397.1s, and no protected files were modified.

### Informational Metrics
- final_test_acc: 92.93%
- final_test_loss: 0.2925
- training_seconds: 300.0
- total_seconds: 397.1
- startup_seconds: 2.9
- peak_vram_mb: 660.4
- num_epochs: 112
- num_steps: 43464
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> No human notes yet.

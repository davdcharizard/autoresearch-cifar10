# EXP-044: Mild RandAugment After Crop/Flip

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md
- **Plan**: plans/plan-044.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-044
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - skipped if no remote exists)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned isolated augmentation probe in `train.py` by inserting `transforms.RandAugment(num_ops=1, magnitude=5)` into the training transform after `RandomHorizontalFlip()` and before `ToTensor()`. The architecture, optimizer, LR schedule, weight decay, label smoothing, reflected crop padding, compile/channels-last path, seed, and once-per-epoch validation path were preserved.

### Surprises & Discoveries

No implementation surprises. The local torchvision install exposes `transforms.RandAugment`, and the transform can be inserted without new imports because `torchvision.transforms` is already imported as `transforms`.

### Decisions

Used the conservative policy from the plan, `num_ops=1` and `magnitude=5`, rather than the torchvision default. This keeps EXP-044 focused on mild policy augmentation and reduces the risk of repeating prior over-regularization from cutout-style masking.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 46053; shell PID 2926938, `uv run train.py` PID 2926939, Python worker PID 2926942
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 12:35 UTC
- **Ended**: 2026-06-09 12:44 UTC

Description:
- Run the current CIFAR-10 training harness locally on a single selected GPU with mild RandAugment inserted after crop/flip. This tests whether policy augmentation improves generalization beyond the current `2e-4` label-smoothed reflection anchor. The success threshold is `best_test_acc >= 94.07%`, because the active baseline is 93.97% and the goal requires a +0.10 percentage-point improvement.

Observations:
- Startup is clean on GPU 0: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: `run.log` L1-L4)
- Early training is healthy: first progress lines show `lr: 0.1000`, no NaN/Inf/OOM/traceback patterns, and validation reached `best: 82.63%` by epoch 10. (source: `run.log` L5-L25)
- Pre-drop training was noisy but finite, reaching a best validation accuracy of 89.41% by epoch 37. The first LR drop occurred at step 21000 with `lr: 0.0100`, and post-drop accuracy reached 93.29% by epoch 56. (source: `run.log` L78-L116)
- The best result arrived at epoch 80 with `test_acc: 93.83%`; later evaluations fluctuated below that and the final evaluation was 92.80%. Mild RandAugment therefore did not clear the 94.07% improvement threshold. (source: `run.log` L164-L208)
- No second LR drop occurred: `grep -n "step 64000" run.log` returned no matches, as expected because the run completed at 39,015 steps. (source: `run.log` summary and grep exit 1/no output)

Key Metrics:
- `best_test_acc`: 93.83%
- `final_test_acc`: 92.80%
- `final_test_loss`: 0.2577
- `training_seconds`: 300.0
- `total_seconds`: 458.2
- `startup_seconds`: 3.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,015
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: baseline is 93.97%, so EXP-044 required `best_test_acc >= 94.07%` to count as improvement. Result did not meet threshold. (source: active goal/index baseline and `run.log` summary)
- Scope: tracked source diff during the run was only the planned `train.py` transform insertion. (source: `git diff -- train.py`)
- API/syntax/lint preflight: `transforms.RandAugment` was available, `python3 -m py_compile train.py` passed, and `uv run ruff check train.py` passed before launch.
- Transform placement and anchors: RandAugment was inserted after `RandomHorizontalFlip()` and before `ToTensor()`; architecture, optimizer, LR, weight decay, schedule, label smoothing, reflected crop, and validation cadence were preserved.
- Schedule and geometry: `Batches per epoch: 390`, initial progress used `lr: 0.1000`, step 21000 changed to `lr: 0.0100`, step 64000 was absent, and `num_params` remained 822,790. (source: `run.log`)
- Completion: process exited 0; total wall-clock was 458.2 seconds, below the 10-minute cap, and a numeric `best_test_acc` was printed. (source: `run.log` summary)
- Improvement rule: `best_test_acc=93.83%` is below `94.07%`, so verdict is no-improvement under the +0.10 percentage-point rule.

### Informational Metrics
- Best epoch signal: epoch 80 reached 93.83%, then accuracy regressed/fluctuated from 93.65% to a final 92.80%. (source: `run.log` L164-L208)
- Throughput/runtime: total runtime was 458.2s, notably slower than the non-RandAugment anchor runs but still inside the hard cap.

## Errors & Dead Ends

## Human Notes

> No human notes yet.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

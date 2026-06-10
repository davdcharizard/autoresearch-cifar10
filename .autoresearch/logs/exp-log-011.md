# EXP-011: ResNet-20 Width 1.25x

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-011
- **Commit**: 03cc708
- **PR**: skipped (no git remote configured)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned modest width increase on top of the EXP-002 FP32 compile/channels-last ResNet-20 baseline. `train.py` now defines `STAGE_WIDTHS = (20, 40, 80)`, wires those widths through `ResNet.__init__`, and defines `LR_MILESTONES = [24000, 64000]` for the scheduler.

### Surprises & Discoveries
No implementation surprises. The existing CIFAR shortcut padding already handles channel increases, so the width change only required replacing hardcoded stage widths in the model constructor.

### Decisions
Kept width as a single tuple hyperparameter to make the experiment easy to audit and revert. The second LR milestone is deliberately unreachable to avoid the LR 0.001 phase that hurt EXP-003.

## Experimental Adjustments
- The successful `autoresearch/exp-011` branch was fast-forward merged into `autoresearch/dev` at commit `03cc708`. Push and PR creation were skipped because `git remote -v` returned no configured remotes.

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 151248
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 13:52 UTC
- **Ended**: 2026-06-08 14:00 UTC

Description:
- Run the 1.25x wider ResNet-20 FP32 compile/channels-last recipe locally on one GPU with output redirected to `run.log`. This tests whether modest width capacity plus an earlier first LR drop can exceed the current 91.95% baseline by the required +0.10 points. Success requires `best_test_acc >= 92.05%`; a key diagnostic is whether the run reaches the planned step-24000 LR drop.

Observations:
- Physical GPU 0 was occupied by an unrelated run in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8`, so EXP-011 launched on physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Startup is clean: log reports `Device: cuda`, `ResNet-20 | params: 420,670`, `Time budget: 300s`, and `Batches per epoch: 390`. The known non-fatal Inductor TF32 warning appeared again. (source: `nvidia-smi`, `pgrep`, and `run.log` startup lines)
- Early monitoring is clean through epoch 27: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. Throughput is roughly 19k-22k img/s, GPU memory is 863 MiB on physical GPU 1, and the best early test accuracy is 86.85%. (source: `run.log` lines 8-60, `nvidia-smi`)
- The planned first LR drop occurred at step 24000, switching to `lr: 0.0100` during epoch 62. Accuracy jumped from an 88.04% pre-drop best to 91.79% by epoch 69, but it is still below the tightened 92.05% improvement threshold while the run continues. (source: `run.log` lines 120-146)
- The run completed successfully with `best_test_acc: 92.12%`, exceeding the 91.95% baseline by +0.17 percentage points and clearing the tightened 92.05% success threshold. It ended after 113 epochs, 43713 steps, and 300.0 training seconds; final test accuracy was also 92.12%. (source: `run.log` lines 234-243)

Key Metrics:
- best_test_acc: 92.12%
- final_test_acc: 92.12%
- final_test_loss: 0.3665
- training_seconds: 300.0
- total_seconds: 406.2
- startup_seconds: 2.5
- peak_vram_mb: 468.3
- num_epochs: 113
- num_steps: 43713
- num_params: 420,670

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. `exp-index.sh baseline` reports `baseline=91.95`, so the tightened success threshold is 92.05%.
- Single-GPU CUDA execution: passed. Because physical GPU 0 was occupied, the run and verification used `CUDA_VISIBLE_DEVICES=1`; PyTorch reported CUDA available, visible device count 1, and device `NVIDIA H20`.
- Experiment completion: passed. `run.log` contains parseable final summary metrics including `best_test_acc: 92.12%` and `peak_vram_mb: 468.3`.
- Primary metric condition: passed. `best_test_acc=92.12%` is greater than or equal to the 92.05% threshold.
- Schedule/throughput sanity: passed. The first LR drop occurred at step 24000 and the run reached 43713 total steps, giving 19713 LR 0.01 refinement steps without entering the unreachable second LR milestone.
- Scope review: passed. `git diff -- train.py` only changes `STAGE_WIDTHS`, `LR_MILESTONES`, `ResNet` channel wiring, and scheduler milestone wiring.
- Validation cadence review: passed. `train.py` has one `evaluator.evaluate(model, device)` call after the training dataloader loop, so validation remains once per epoch.

### Informational Metrics
- final_test_acc: 92.12%
- final_test_loss: 0.3665
- training_seconds: 300.0
- total_seconds: 406.2
- startup_seconds: 2.5
- peak_vram_mb: 468.3
- num_epochs: 113
- num_steps: 43713
- num_params: 420,670

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

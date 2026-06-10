# Plan EXP-003: Earlier Second LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md

## Milestones

### Milestone 1: Code changes implemented and checked
- [x] Change only the second LR milestone in `train.py` from `48000` to `40000`.
- [x] Preserve EXP-002's FP32 throughput setup: cuDNN benchmark, channels-last, and compile.
- [x] Preserve augmentation, loss, optimizer hyperparameters, model architecture, and evaluation cadence.
- [x] Run `uv run ruff check train.py` and `python3 -m py_compile train.py`.

### Milestone 2: Experiment run completed
- [x] Confirm PyTorch sees exactly one CUDA device with `CUDA_VISIBLE_DEVICES=0`.
- [x] Remove any stale `run.log`.
- [x] Run the experiment with one GPU and a 600 second wall timeout.
- [x] Confirm the log contains the final summary block and numeric `best_test_acc`.

### Milestone 3: Result parsed for analysis
- [x] Extract `best_test_acc`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, and `num_steps`.
- [x] Compare `best_test_acc` to the 91.95 baseline.
- [x] Confirm the run reached LR 0.001 after step 40,000.
- [x] Preserve parsed results in `logs/exp-log-003.md`.

## Code Changes
- **train.py**: Change `optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)` to use `milestones=[32000, 40000]`. Make no other source change.

## Configuration Changes
- scheduler second milestone: `48000` -> `40000` (EXP-002 reached 43,398 steps, so this creates a reachable final LR 0.001 phase).
- scheduler first milestone: remains `32000`.
- all throughput flags from EXP-002 remain unchanged.

## Execution Environment
- Method: local command from project root.
- Resources: one NVIDIA H20 GPU via `CUDA_VISIBLE_DEVICES=0`.
- Estimated runtime: 6-10 minutes including startup and evaluation; training budget remains 300 seconds.
- Log output: capture stdout/stderr to `run.log`; summarize metrics in `.autoresearch/logs/exp-log-003.md`.
- Tool skill: none.

## Abort Criteria
- Stop if PyTorch sees zero or more than one CUDA device under `CUDA_VISIBLE_DEVICES=0`.
- Stop if the run exceeds 600 seconds wall time.
- Stop if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` compiler failure, or no final summary block.
- Stop if the diff changes files other than `train.py`.
- Stop if any change besides the second LR milestone appears in `train.py`.

## Verification Protocol

### Verification Procedure
1. Confirm single-device PyTorch visibility:
   `CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"`
   Pass if output reports `1` and an NVIDIA H20.
2. Remove stale logs:
   `rm -f run.log`
   Pass if no stale `run.log` remains.
3. Run the experiment:
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Pass if the command exits zero before timeout.
4. Confirm the final metric exists:
   `grep "^best_test_acc:" run.log`
   Pass if a numeric percentage is present.
5. Compare against the current baseline:
   parse `best_test_acc` from `run.log`; pass the primary metric condition if it is greater than `91.95`.
6. Confirm scope and scheduler-only constraint:
   `git diff -- train.py`
   Pass if the diff only changes the second scheduler milestone from 48,000 to 40,000.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final epoch test accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` — final epoch test loss.
- training_seconds: `grep "^training_seconds:" run.log` — measured training budget use.
- total_seconds: `grep "^total_seconds:" run.log` — full process runtime.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` — epochs completed in the time budget.
- num_steps: `grep "^num_steps:" run.log` — optimizer steps completed.
- num_params: `grep "^num_params:" run.log` — should remain baseline model size.

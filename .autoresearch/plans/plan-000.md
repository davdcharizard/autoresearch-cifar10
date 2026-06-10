# Plan EXP-000: Cutout, Label Smoothing, and Cosine LR
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md

## Milestones

### Milestone 1: Code changes implemented and checked
- [x] Add cutout-style tensor augmentation in `train.py` using `transforms.RandomErasing`.
- [x] Add modest label smoothing to the training loss.
- [x] Replace `MultiStepLR` with cosine annealing over the existing `MAX_STEPS` horizon.
- [x] Enable Nesterov momentum in SGD.
- [x] Run `uv run ruff check train.py` if available through the project environment.

### Milestone 2: Experiment run completed
- [x] Confirm a single GPU is visible before starting.
- [x] Remove any stale `run.log`.
- [x] Run the experiment with one GPU and a 600 second wall timeout.
- [x] Confirm the log contains the final summary block and numeric `best_test_acc`.

### Milestone 3: Result parsed for analysis
- [x] Extract `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` from `run.log`.
- [x] Compare `best_test_acc` to the current baseline `91.52`.
- [x] Preserve the parsed result in `logs/exp-log-000.md` during execution.

## Code Changes
- **train.py**: Add `CUTOUT_SIZE`, `LABEL_SMOOTHING`, and `ETA_MIN` hyperparameters near the existing training constants. Add `transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(1.0, 1.0), value=0)` after `ToTensor()` and before normalization to approximate CIFAR cutout without a new helper class. Change `F.cross_entropy(outputs, targets)` to use `label_smoothing=LABEL_SMOOTHING`. Set SGD `nesterov=True`. Replace `optim.lr_scheduler.MultiStepLR(... milestones=[32000, 48000], gamma=0.1)` with `optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=ETA_MIN)`.

## Configuration Changes
- `CUTOUT_SIZE`: absent -> `16` conceptual target via RandomErasing scale range (CIFAR cutout-style masking; exact erased area varies).
- `LABEL_SMOOTHING`: absent -> `0.05` (modest regularization to avoid overconfidence without heavy underfitting).
- `ETA_MIN`: absent -> `0.001` (keeps a small learning rate floor through the fixed step horizon).
- `optimizer nesterov`: absent -> `True` (small SGD improvement with same optimizer family).
- `scheduler`: `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` -> `CosineAnnealingLR(T_max=MAX_STEPS, eta_min=0.001)` (smoother decay for fixed-time training).

## Execution Environment
- Method: local command from project root.
- Resources: one NVIDIA GPU only, expected H20 class GPU with ample VRAM.
- Estimated runtime: 6-10 minutes including startup and evaluation; training budget is 300 seconds.
- Log output: capture stdout/stderr to `run.log`; summarize parsed metrics in `.autoresearch/logs/exp-log-000.md`.
- Tool skill: none.

## Abort Criteria
- Stop if no GPU is visible before launch.
- Stop if the run exceeds 600 seconds wall time.
- Stop if `run.log` shows a Python traceback, CUDA OOM, dependency error, or no final summary block.
- Stop if the modified code attempts to touch files other than `train.py`.
- Stop if validation is added anywhere beyond the existing once-per-epoch `evaluator.evaluate(model, device)` call.

## Verification Protocol

### Verification Procedure
1. Confirm single-GPU visibility:
   `CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`
   Pass if the command reports one GPU; fail as infrastructure if no GPU is visible.
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
   parse `best_test_acc` from `run.log`; pass the primary metric condition if it is greater than `91.52`.
6. Confirm scope and harness constraints:
   `git diff --name-only`
   Pass if only `train.py` changed among project source files.
7. Confirm validation cadence:
   inspect the `train.py` diff and pass if the only evaluation call remains the existing once-per-epoch `evaluator.evaluate(model, device)`.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final epoch test accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` — final epoch test loss.
- training_seconds: `grep "^training_seconds:" run.log` — measured training budget use.
- total_seconds: `grep "^total_seconds:" run.log` — full process runtime.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` — epochs completed in the time budget.
- num_steps: `grep "^num_steps:" run.log` — optimizer steps completed in the time budget.
- num_params: `grep "^num_params:" run.log` — model parameter count.

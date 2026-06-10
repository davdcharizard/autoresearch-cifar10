# Plan EXP-002: FP32 Throughput Without AMP
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md

## Milestones

### Milestone 1: Code changes implemented and checked
- [x] Add throughput feature flags in `train.py` for cuDNN benchmark, channels-last, and compile only.
- [x] Enable `torch.backends.cudnn.benchmark = True` for fixed-shape CIFAR batches.
- [x] Convert model and input batches to channels-last memory format when CUDA is available.
- [x] Compile the model with `torch.compile` when CUDA is available.
- [x] Leave AMP/autocast disabled and keep FP32 forward/loss arithmetic.
- [x] Run `uv run ruff check train.py` and `python3 -m py_compile train.py`.

### Milestone 2: Experiment run completed
- [x] Confirm PyTorch sees exactly one CUDA device with `CUDA_VISIBLE_DEVICES=0`.
- [x] Remove any stale `run.log`.
- [x] Run the experiment with one GPU and a 600 second wall timeout.
- [x] Confirm the log contains the final summary block and numeric `best_test_acc`.

### Milestone 3: Result parsed for analysis
- [x] Extract `best_test_acc`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, and `num_steps`.
- [x] Compare `best_test_acc` to the 91.52 baseline.
- [x] Compare `num_steps` to EXP-001's 39,558 steps to quantify the cost of removing BF16.
- [x] Preserve parsed results in `logs/exp-log-002.md`.

## Code Changes
- **train.py**: Add constants `USE_CUDNN_BENCHMARK`, `USE_CHANNELS_LAST`, and `USE_COMPILE`. Set cuDNN benchmark after selecting the CUDA device. After constructing the ResNet, move it to channels-last memory format when enabled and then wrap with `torch.compile(model)` when enabled. In the training loop, move `inputs` to CUDA with channels-last memory format when enabled. Keep forward/loss in ordinary FP32 code with `outputs = model(inputs)` and `loss = F.cross_entropy(outputs, targets)`.

## Configuration Changes
- `USE_CUDNN_BENCHMARK`: absent -> `True` (fixed input shape should allow cuDNN autotuning).
- `USE_CHANNELS_LAST`: absent -> `True` (convolution-oriented memory format on NVIDIA GPU).
- `USE_COMPILE`: absent -> `True` (try PyTorch Inductor optimization on the simple static CNN).
- AMP/autocast: remains absent/disabled (preserve FP32 arithmetic to isolate EXP-001's precision confound).
- No statistical recipe changes: crop/flip augmentation, cross entropy, SGD settings, and `MultiStepLR([32000, 48000])` remain baseline.

## Execution Environment
- Method: local command from project root.
- Resources: one NVIDIA H20 GPU via `CUDA_VISIBLE_DEVICES=0`.
- Estimated runtime: 6-10 minutes including startup and evaluation; training budget remains 300 seconds.
- Log output: capture stdout/stderr to `run.log`; summarize metrics in `.autoresearch/logs/exp-log-002.md`.
- Tool skill: none.

## Abort Criteria
- Stop if PyTorch sees zero or more than one CUDA device under `CUDA_VISIBLE_DEVICES=0`.
- Stop if the run exceeds 600 seconds wall time.
- Stop if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` compiler failure, or no final summary block.
- Stop if the diff changes files other than `train.py`.
- Stop if AMP/autocast, augmentation, scheduler milestones, optimizer hyperparameters, model architecture, or validation cadence are changed.

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
   parse `best_test_acc` from `run.log`; pass the primary metric condition if it is greater than `91.52`.
6. Confirm scope and FP32-throughput constraints:
   `git diff -- train.py`
   Pass if the diff only adds cuDNN benchmark, channels-last, and compile mechanisms while leaving AMP/autocast absent and preserving augmentation, loss target semantics, optimizer hyperparameters, scheduler milestones, model architecture, and `evaluator.evaluate(model, device)` cadence.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final epoch test accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` — final epoch test loss.
- training_seconds: `grep "^training_seconds:" run.log` — measured training budget use.
- total_seconds: `grep "^total_seconds:" run.log` — full process runtime including compile/startup overhead.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` — epochs completed in the time budget.
- num_steps: `grep "^num_steps:" run.log` — optimizer steps completed; primary diagnostic for throughput.
- num_params: `grep "^num_params:" run.log` — should remain baseline model size.

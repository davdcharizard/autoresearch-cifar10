# Plan EXP-001: Time-Aligned Pre-Activation WRN-16-2
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Implement the WRN and time-aligned optimizer
- [x] Modify only `train.py` to implement pre-activation WRN-16-2 with stage widths 32/64/128 and two blocks per stage.
- [x] Set batch size 256, Nesterov SGD, selective weight decay, cuDNN benchmarking, efficient gradient clearing, and elapsed-training-time warmup/cosine LR.
- [x] Run `uv run ruff check train.py` and correct all reported issues.

### Milestone 2: Verify model and schedule mechanics
- [x] Run a CUDA forward/backward smoke test with a synthetic `(256, 3, 32, 32)` batch and verify logits shape `(256, 10)`, finite loss, and successful optimizer step.
- [x] After 20 untimed warmup steps, time 100 synthetic training steps with CUDA synchronization and project dataset-equivalent passes in 300 seconds; require at least 40 projected passes before committing to the full run.
- [x] Inspect `git diff -- train.py` and confirm no file other than `train.py` is modified by the experiment.

### Milestone 3: Execute the full experiment
- [x] Confirm exactly one NVIDIA H20 is selected with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
- [x] Remove stale `run.log`, execute `timeout 600s uv run train.py > run.log 2>&1`, and monitor only short progress/summary extracts.
- [x] Confirm the process exits zero and the final summary is present.

### Milestone 4: Verify and record the outcome
- [x] Query the moving baseline from `04-results.tsv` and compare the final `best_test_acc` against the concrete 91.64% success threshold.
- [x] Record all informational metrics from the final summary and verify evaluation cadence and budget compliance.
- [x] Remove `run.log` after analysis has captured the needed evidence.

## Code Changes
- **`train.py`**: Replace the post-activation ResNet-20 block/model with a pre-activation WRN-16-2. The stem remains a 3x3 3-to-16 convolution; residual stages use widths 32, 64, and 128 with two blocks each; stage transitions use learned 1x1 projection shortcuts applied to the pre-activated tensor; the head uses final BN/ReLU, global average pooling, and a 128-to-10 classifier.
- **`train.py`**: Initialize convolution weights with Kaiming-normal fan-out initialization, BN scales to one and biases to zero, and the classifier with the existing Kaiming-normal convention plus zero bias. This preserves explicit, reproducible initialization for every new module.
- **`train.py`**: Replace global optimizer decay with two parameter groups: convolution/linear weights (`ndim >= 2`) receive weight decay `5e-4`; BN affine parameters and biases receive zero decay. Use SGD with momentum 0.9 and Nesterov enabled.
- **`train.py`**: Remove `MultiStepLR` and `scheduler.step()`. Before each optimizer step, compute LR from the prior completed training time. Use LR 0.002 on the first step, linearly warm to 0.2 over the first 5% of 300 counted seconds, then cosine-decay to 0.002 at 100%. Assign the same LR to both optimizer groups.
- **`train.py`**: Set `torch.backends.cudnn.benchmark = True` and `torch.backends.cudnn.deterministic = True`, use `optimizer.zero_grad(set_to_none=True)`, and retain seed 42 and the existing crop/flip pipeline.
- **`train.py`**: Evaluate every fifth completed epoch and always evaluate the final budget-truncated epoch. This stays below the once-per-epoch limit while cutting excluded validation overhead enough to preserve the hard 10-minute total ceiling. Maintain `best_test_acc` over all evaluations performed.
- **`train.py`**: Preserve existing summary fields and progress logging so realized LR, steps, epochs, throughput, parameter count, and VRAM remain observable.

## Configuration Changes
- `NUM_BLOCKS`: 3 -> 2 (WRN-16 depth convention, reducing sequential residual blocks)
- stage widths: `16/32/64` -> `32/64/128` after the 16-channel stem (WRN width multiplier 2)
- `BATCH_SIZE`: 128 -> 256 (better H20 occupancy and fewer sequential updates per data pass)
- `LR`: 0.1 -> 0.2 peak, with 0.002 initial/floor (linear batch scaling plus bounded warmup/cosine tail)
- `WEIGHT_DECAY`: 1e-4 global -> 5e-4 on convolution/linear weights only (canonical CIFAR WRN regularization without decaying BN/bias)
- `MOMENTUM`: 0.9 -> 0.9 with `nesterov=True`
- schedule: fixed milestones `[32000, 48000]` -> 5% time warmup plus 95% time cosine decay (both decay phases become reachable regardless of step throughput)
- evaluation cadence: every epoch -> every fifth epoch plus the final epoch (prevents the wider evaluator from exhausting the total wall-clock limit)
- `MAX_STEPS`: retain 64000 as a safety ceiling; the 300-second counted training budget remains authoritative.

## Execution Environment
- Method: local single-process run from the project root using `timeout 600s uv run train.py > run.log 2>&1`
- Resources: exactly one NVIDIA H20 GPU with approximately 98 GB memory; existing 8-worker CIFAR-10 loader and local dataset cache
- Estimated runtime: 5 minutes counted GPU training plus approximately 30-90 seconds of startup and periodic validation, with a hard total ceiling of 10 minutes; evaluating every fifth epoch plus the final epoch provides explicit wall-time margin
- Log output: all stdout/stderr redirected to `run.log`; inspect bounded `rg`/`tail` extracts only
- Tool skill: none; execution is fully local and offline

## Abort Criteria
- Abort the smoke test immediately on shape mismatch, CUDA error, non-finite loss, or failed backward/optimizer step.
- Abort before the full run if the 100-step timed smoke benchmark projects fewer than 40 dataset-equivalent passes in 300 counted seconds; the chosen hypothesis depends on sufficient image exposure.
- Abort the full run on a CUDA/runtime traceback, non-finite training loss, or repeated absence of progress after 2 minutes.
- `timeout 600s` must terminate any run at the 10-minute hard limit; classify exit 124 as failure.
- Do not abort merely for weak intermediate test accuracy; the time-aligned low-LR phase is the core intervention and must finish before judging accuracy.
- Abort and classify invalid if any file other than `train.py` is modified by experiment implementation, evaluation occurs more than once per epoch, the seed is changed, or `prepare.py`/evaluation behavior changes.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and success threshold. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=91.54`, making the minimum successful result `best_test_acc >= 91.64`.
2. Confirm hardware before execution with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`; require the selected device to report `NVIDIA H20`. The process must use a single local GPU.
3. Remove stale output with `rm -f run.log`, then run `timeout 600s uv run train.py > run.log 2>&1`. Treat any nonzero exit as failure; exit 124 is specifically the 10-minute timeout failure.
4. Require a complete result using `rg '^(best_test_acc|training_seconds|total_seconds|peak_vram_mb):' run.log`. If `best_test_acc` is absent, classify as crash and inspect `tail -n 50 run.log`.
5. Require `training_seconds` to be approximately 300 seconds and `total_seconds <= 600.0`. Confirm source inspection shows one `evaluator.evaluate` call guarded to every fifth epoch plus the final epoch, and log inspection shows at most one `eval ep` entry per epoch.
6. Parse the numeric `best_test_acc` from the final summary and require `best_test_acc >= 91.64`. Stop verification on the first failed necessary condition.
7. Scope-check with `git status --short` and `git diff -- train.py`; only `train.py` may contain experiment code changes.

### Informational Metrics (Optional)
- `peak_vram_mb`: final summary in `run.log` via `rg '^peak_vram_mb:' run.log`
- `final_test_acc`: final summary via `rg '^final_test_acc:' run.log`
- `final_test_loss`: final summary via `rg '^final_test_loss:' run.log`
- `training_seconds`: final summary via `rg '^training_seconds:' run.log`
- `total_seconds`: final summary via `rg '^total_seconds:' run.log`
- `num_epochs`: final summary via `rg '^num_epochs:' run.log`
- `num_steps`: final summary via `rg '^num_steps:' run.log`
- `num_params`: final summary via `rg '^num_params:' run.log`

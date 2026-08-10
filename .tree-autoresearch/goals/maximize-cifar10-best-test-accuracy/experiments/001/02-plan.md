# Plan EXP-001: Time-Aware Pre-Activation WRN-16-4
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement the bounded WRN training package
- [x] Modify only `train.py`: replace ResNet-20 with a six-block pre-activation WRN-16-4 and expectation-preserving per-example drop path.
- [x] Add batch-256 BF16 autocast, channels-last model/input layout, Nesterov SGD, and the corrected piecewise time-based LR/drop-path schedules.
- [x] Preserve seed 42, baseline crop/flip and input normalization, `prepare.py` imports, evaluator, timing boundary, once-per-epoch maximum validation, and final summary format.
- [x] Verify syntax and style with `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `uv run ruff format --check train.py`.

### Milestone 2: Validate the implementation before the timed run
- [x] Confirm the diff is limited to `train.py` with `git status --short` and `git diff -- train.py`; repeat this check immediately before launch.
- [x] Confirm physical GPU 0 is the required H20 using `nvidia-smi -i 0 --query-gpu=index,name,memory.total --format=csv,noheader`.
- [x] Run the inline, no-file CUDA benchmark below. It constructs `PreActWideResNet`, verifies 2.6M-2.9M parameters, performs 20 BF16 synthetic training steps, reports median step latency, runs one full frozen `Eval.evaluate` pass, and confirms finite loss/logits. The timed experiment starts a new process and resets seed 42, so this preflight cannot alter its RNG state.
- [x] Project total runtime as `300 + ceil((300 / median_step_s) / 195) * eval_s + 30`. If it exceeds 570 seconds, set `EVAL_EVERY = 2` in `train.py` before launch and re-run syntax/style/scope checks; otherwise use `EVAL_EVERY = 1`. In either case, always evaluate the final partial epoch after the time budget is reached.

### Milestone 3: Execute one protocol-compliant experiment
- [x] Remove any stale `run.log`, then run exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- [x] Monitor process liveness without streaming the log. Every 30-60 seconds, run a bounded `grep -Ei 'loss: (nan|inf)|CUDA out of memory|illegal memory access|Traceback' run.log | tail -n 5`; abort on a match, timeout, or process failure.
- [x] Require a complete final summary, approximately 300 charged training seconds, fewer than 600 total seconds, and no more than one `eval ep` line per completed epoch.

### Milestone 4: Verify and preserve the result
- [x] Compare the parsed `best_test_acc` against the parent BASE metric 91.51%; success requires at least 91.61%.
- [x] Collect all approved informational metrics only after necessary conditions pass.
- [x] Record execution decisions and metrics in `03-execute.md`; leave `run.log` available for the analysis phase, which will remove it after recording evidence.

## Code Changes
- **`train.py`**: Replace `BasicBlock`/`ResNet` with a pre-activation wide residual model using a 16-channel stem, two residual blocks per stage, and stage widths 64/128/256. Shape-changing blocks use a learned 1x1 shortcut from the pre-activated tensor; identity blocks preserve the raw shortcut. Finish with BN, ReLU, adaptive average pooling, and a 256-to-10 classifier.
- **`train.py`**: Add expectation-preserving, per-example residual drop path. Block base probabilities increase linearly with depth to 0.08; the training loop supplies a time-dependent scale of 1.0 through 75% progress and linearly decays it to zero by 100%. Evaluation disables dropping through `model.eval()` and a default zero scale.
- **`train.py`**: Replace `MultiStepLR` and the binding `MAX_STEPS` limit with a piecewise schedule based only on `total_training_time / TIME_BUDGET_S`. During the first 5%, LR rises linearly from 0.02 to 0.20; over the remaining 95%, cosine decays from 0.20 to 0.002. Set param-group LR immediately before each forward pass.
- **`train.py`**: Set batch size 256, use Nesterov SGD with momentum 0.9 and the unchanged `1e-4` weight decay, enable cuDNN benchmark, store the model and inputs in channels-last layout, run training forward/loss under CUDA BF16 autocast, and keep FP32 parameters, optimizer state, BatchNorm state, backward, and evaluation.
- **`train.py`**: Use `optimizer.zero_grad(set_to_none=True)`, retain `torch.cuda.synchronize()` inside the charged step timer, and log LR/drop-path/progress diagnostics without changing final metric keys. Do not add input rescaling, Mixup/CutMix, EMA, label smoothing, `torch.compile`, or new dependencies in EXP-001.
- **`train.py`**: Add `EVAL_EVERY`, defaulting to 1 and changed to 2 only by the pre-authorized runtime projection above. Evaluate at the configured epoch cadence and unconditionally after the budget-ending partial epoch, never more than once in one epoch. Print a startup configuration line containing architecture, parameter count, peak LR, warmup fraction, maximum drop-path rate, and evaluation cadence.

## Configuration Changes
- Architecture: ResNet-20 (16/32/64, 3 blocks/stage) -> PreAct WRN-16-4 (64/128/256, 2 blocks/stage), targeting approximately 2.75M parameters.
- Batch size: 128 -> 256, balancing H20 utilization against update count and BatchNorm frequency.
- Precision/layout: FP32 contiguous -> BF16 autocast with FP32 state and channels-last tensors, to make the wider model affordable on H20.
- Optimizer: SGD momentum 0.9 -> SGD momentum 0.9 with Nesterov; weight decay remains `1e-4` to avoid an unsupported regularization confound.
- LR: step milestones at 32,000/48,000 -> time-indexed 5% warmup (0.02 to 0.20) plus cosine (0.20 to 0.002).
- Residual regularization: none -> depth-scaled drop path with maximum 0.08, annealed away from 75%-100% of charged training time.
- Evaluation cadence: every epoch -> every epoch unless the measured preflight projects more than 570 total seconds, in which case every second epoch plus the final partial epoch.
- Data/loss/seed: unchanged crop/flip, mean `(0.4914, 0.4822, 0.4465)`, std `(1, 1, 1)`, ordinary cross-entropy, and seed 42.

## Execution Environment
- Method: local single-process run from the repository root.
- Resources: physical GPU 0 only, NVIDIA H20 with 97,871 MiB; existing `.venv`, local CIFAR-10 data, and current locked dependencies.
- Estimated runtime: 300 seconds charged training plus startup and epoch-end evaluation, expected 400-525 seconds total. BASE spent 75.3 seconds outside its 300-second training timer across 89 evaluations; EXP-001 has a 225-second overhead allowance under the 600-second cap. If the wider model completes about 80-105 epochs, average evaluation overhead must remain below roughly 2.1-2.8 seconds per epoch.
- Log output: all stdout/stderr redirected to `run.log`; terminal context receives only liveness status and, after exit, bounded grep/tail summaries.
- Preflight command (inline and artifact-free):

  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
  import statistics
  import time

  import torch
  import torch.nn.functional as F

  from train import NUM_CLASSES, PreActWideResNet, evaluator

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  device = torch.device("cuda")
  model = PreActWideResNet(NUM_CLASSES).to(
      device, memory_format=torch.channels_last
  )
  optimizer = torch.optim.SGD(
      model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4, nesterov=True
  )
  inputs = torch.randn(256, 3, 32, 32, device=device).contiguous(
      memory_format=torch.channels_last
  )
  targets = torch.randint(NUM_CLASSES, (256,), device=device)
  times = []
  model.train()
  for _ in range(20):
      torch.cuda.synchronize()
      started = time.perf_counter()
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          logits = model(inputs, drop_scale=1.0)
          loss = F.cross_entropy(logits, targets)
      loss.backward()
      optimizer.step()
      torch.cuda.synchronize()
      times.append(time.perf_counter() - started)
  assert torch.isfinite(loss) and torch.isfinite(logits).all()
  param_count = sum(p.numel() for p in model.parameters())
  assert 2_600_000 <= param_count <= 2_900_000
  started = time.perf_counter()
  evaluator.evaluate(model, device)
  torch.cuda.synchronize()
  eval_s = time.perf_counter() - started
  print(f"param_count={param_count}")
  print(f"median_step_s={statistics.median(times[5:]):.6f}")
  print(f"eval_s={eval_s:.6f}")
  PY
  ```
- Tool skill: none; execution is local and does not use a job scheduler.

## Abort Criteria
- Abort before launch if GPU 0 is not an NVIDIA H20 with approximately 98 GB, the immediately pre-launch working diff includes anything except `train.py`, syntax/style checks fail, the inline benchmark produces an exception/non-finite value or parameter count outside 2.6M-2.9M, or median synthetic step time is at least 0.5 seconds.
- Abort the timed run if it exits non-zero, reports CUDA OOM/illegal memory access, produces NaN/Inf loss, or reaches the 600-second timeout. A timeout is an experiment failure even if charged training appears healthy.
- Do not abort merely because early accuracy is low; the hypothesis depends on the complete time-cosine trajectory. The preflight projection determines evaluation cadence before launch; the timed experiment does not change configuration in response to test accuracy.
- A completed run below 91.61% is not retried with a different seed. It proceeds to analysis as `no-improvement`.

## Verification Protocol

### Verification Procedure
1. Confirm the installed tree manager exists, then confirm the parent reference before execution:
   `test -x /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh`
   `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv BASE`
   Require the executable check to pass and `metric=91.51`; the concrete improvement threshold is therefore `best_test_acc >= 91.61`. This session's installed plugin root is fixed at the checked path; `${CLAUDE_PLUGIN_ROOT}` is unset in the Codex shell.
2. Confirm hardware with a 10-second timeout:
   `timeout 10s nvidia-smi -i 0 --query-gpu=index,name,memory.total --format=csv,noheader`
   Require index `0`, model `NVIDIA H20`, and approximately `97871 MiB`.
3. Execute with the frozen command and timeout:
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Exit code must be zero. Any timeout (`124`) or other non-zero exit fails the completion condition.
4. Require all ten final summary keys:
   `test "$(grep -Ec '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log)" -eq 10`
   Failure means the run crashed or did not finish cleanly.
5. Require the startup configuration line to report `PreActWideResNet`, `peak_lr=0.2`, `warmup_fraction=0.05`, `max_drop_path=0.08`, and the preflight-selected `eval_every`. Require `num_params` from the final summary to be between 2.6M and 2.9M. Inspect bounded progress log lines to confirm LR rises during the first 5%, then falls, and the effective maximum drop-path scale reaches zero near 100%.
6. Parse and validate timing using the values printed in `run.log`: `training_seconds` must be at least 299.5 seconds and at most `300 + max(1.0, 2 * preflight_median_step_s)`; `total_seconds` must be less than 600 seconds. The 1-second minimum tolerance safely covers one final charged step when the preflight median is below the 0.5-second abort threshold. Count `eval ep` records and require the count to be less than or equal to `num_epochs`; require the final budget-ending epoch to have an evaluation.
7. Parse `best_test_acc` by removing the trailing `%` and compare numerically against 91.61. A value below 91.61 immediately yields `no-improvement`; do not collect optional metrics as pass evidence.
8. After the necessary conditions pass, extract the bounded summary with `tail -n 12 run.log`, record it in the execution artifact, and hand off to analysis. Analysis removes `run.log` after recording all evidence.

### Informational Metrics (Optional)
- `final_test_acc`: final summary line in `run.log` - final epoch accuracy and convergence check.
- `final_test_loss`: final summary line in `run.log` - final cross-entropy.
- `training_seconds`: final summary line in `run.log` - charged training budget compliance.
- `total_seconds`: final summary line in `run.log` - 600-second outer-runtime compliance.
- `startup_seconds`: final summary line in `run.log` - setup overhead.
- `peak_vram_mb`: final summary line in `run.log` - memory cost of the wider model.
- `num_epochs`: final summary line in `run.log` - completed data passes and evaluation count reference.
- `num_steps`: final summary line in `run.log` - optimizer exposure and throughput diagnostic.
- `num_params`: final summary line in `run.log` - verifies the intended capacity increase.

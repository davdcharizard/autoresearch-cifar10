# Plan EXP-002: Front-Loaded Probabilistic CutMix
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement isolated CutMix
- [x] Modify only `train.py`: add fixed CutMix constants, a safe one-rectangle helper, the early-phase probabilistic gate, two-target loss, and exposure counters.
- [x] Preserve every EXP-001 architecture, optimizer, LR/drop-path schedule, precision/layout, seed, evaluation cadence, timing boundary, and final summary key.
- [x] Pass `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, and `git diff --check`.

### Milestone 2: Validate helper and scope
- [x] Run the exact inline no-file CPU/GPU helper smoke check in Execution Environment. Require shape preservation, finite pixels, paired-target alignment, changed pixels, `adjusted_lambda == 1 - clipped_area/(H*W)`, and a forced `lambda=1` zero-area case that leaves inputs unchanged and returns adjusted lambda 1.
- [x] Confirm `git diff --name-only` reports only `train.py`, GPU 0 is an H20 with 97,871 MiB, and the parent branch code otherwise remains unchanged.

### Milestone 3: Execute once on GPU 0
- [x] Remove stale `run.log`, then run `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` exactly once.
- [x] Monitor liveness and bounded error patterns every 30-60 seconds without streaming training output.
- [x] Require clean exit, complete summary, approximately 300 charged seconds, fewer than 600 total seconds, and at most one evaluation per epoch.

### Milestone 4: Verify parent-relative improvement
- [x] Require unchanged 2,748,890 parameter count and fixed CutMix configuration `prob=0.5`, `alpha=1.0`, `end=0.75`.
- [x] Require a nonzero eligible-batch count and realized applied/eligible ratio between 0.45 and 0.55.
- [x] Compare against parent 94.62%; success requires `best_test_acc >= 94.72%`, then record all informational metrics in `03-execute.md`.

## Code Changes
- **`train.py`**: Add `CUTMIX_PROB = 0.5`, `CUTMIX_ALPHA = 1.0`, and `CUTMIX_END = 0.75`. The alpha constant documents that `Beta(1,1)` is exactly the uniform lambda draw used by the dependency-free implementation.
- **`train.py`**: Add `cutmix_batch(inputs, targets, cpu_generator, cuda_generator, lam=None, center=None, permutation=None)`. For normal training, draw lambda/center from the dedicated CPU generator and permutation from the dedicated CUDA generator; smoke tests supply all three explicitly. Derive `cut_ratio = sqrt(1 - lambda)`, `cut_w = int(W * cut_ratio)`, and `cut_h = int(H * cut_ratio)`, then clip the centered bounds. Clone the paired source patch before in-place assignment, and return original targets, paired targets, adjusted lambda `1 - clipped_area/(H*W)`, and clipped area. A zero-area rectangle performs no assignment and returns adjusted lambda 1.
- **`train.py`**: Create dedicated CPU and GPU CutMix generators, both fixed once at seed 42. This prevents new gate/lambda/rectangle/permutation draws from consuming the parent recipe's global CPU shuffle/augmentation stream or global CUDA drop-path stream; the actual timed process still resets all seeds in a fresh process.
- **`train.py`**: Keep `t0 = time.time()` in its parent location before input transfer. Perform every CutMix operation after `t0` and before the existing post-optimizer CUDA synchronization, so augmentation cost is charged. Mark a batch eligible when progress is below 0.75 and gate it with the dedicated CPU generator at probability 0.5. Compute `adjusted_lambda * CE(outputs, targets_a) + (1-adjusted_lambda) * CE(outputs, targets_b)` from the same logits. Ineligible or gate-false batches use unchanged hard-label CE.
- **`train.py`**: Count eligible and applied batches, include their counts in periodic progress lines, and print one `cutmix: applied=X eligible=Y ratio=Z` line immediately before the existing `---` final summary separator. Extend the startup configuration line with fixed CutMix values. Do not add label smoothing, EMA, a second forward, per-image loops, input normalization changes, or any dependency.

## Configuration Changes
- CutMix probability: none -> 0.5 for eligible batches, a fixed hypothesis chosen before execution.
- CutMix lambda: none -> `Beta(1,1)`, implemented as `torch.rand(()).item()` because Beta(1,1) is uniform.
- CutMix phase: none -> progress `< 0.75`; the final 25% is fully clean.
- CutMix geometry: one shared rectangle per batch; side fractions are `sqrt(1-lambda)`, with safe paired-patch cloning and clipped-area lambda correction weighting the original target by the retained original-image area.
- CutMix RNG: dedicated CPU and CUDA generators, each seeded once with 42, so adding CutMix does not perturb the parent global RNG streams. This is a deterministic augmentation stream, not seed selection or rerolling.
- All EXP-001 settings remain fixed: PreAct WRN-16-4, batch 256, BF16 channels-last, Nesterov SGD, weight decay `1e-4`, LR 0.02 -> 0.20 -> 0.002 by charged time, maximum drop path 0.08 annealed after 75%, global seed 42, crop/flip, unit-std input scaling, and every-epoch evaluation. CutMix turning off and drop-path annealing beginning at the same 75% boundary is intentional but confounds attribution of final-quarter dynamics; analysis must treat that phase as a joint transition.

## Execution Environment
- Method: local single-process execution from the repository root.
- Resources: physical GPU 0 only, NVIDIA H20 97,871 MiB; existing environment and local CIFAR-10 data.
- Estimated runtime: near EXP-001's 471.9 total seconds. CutMix adds one patch copy and a second CE reduction but no second forward; the 600-second outer cap remains ample.
- Log output: all stdout/stderr to `run.log`; monitoring uses only bounded error grep and process status.
- Helper smoke command (importing `train.py` constructs the existing local evaluator but writes no tracked file; the timed run is a fresh process):

  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
  import torch

  from train import cutmix_batch

  device = torch.device("cuda")
  original = torch.arange(4 * 3 * 32 * 32, device=device).reshape(4, 3, 32, 32)
  original = original.to(torch.float32)
  targets = torch.arange(4, device=device)
  permutation = torch.tensor([1, 2, 3, 0], device=device)
  cpu_generator = torch.Generator().manual_seed(42)
  cuda_generator = torch.Generator(device=device).manual_seed(42)
  mixed, targets_a, targets_b, adjusted, area = cutmix_batch(
      original.clone(), targets, cpu_generator, cuda_generator,
      lam=0.5, center=(16, 16), permutation=permutation,
  )
  changed_area = mixed.ne(original).any(dim=1).sum(dim=(1, 2))
  assert mixed.shape == original.shape and torch.isfinite(mixed).all()
  assert torch.equal(targets_a, targets)
  assert torch.equal(targets_b, targets[permutation])
  assert area > 0 and torch.all(changed_area == area)
  assert abs(adjusted - (1.0 - area / (32 * 32))) < 1e-7
  unchanged, _, _, adjusted_zero, area_zero = cutmix_batch(
      original.clone(), targets, cpu_generator, cuda_generator,
      lam=1.0, center=(16, 16), permutation=permutation,
  )
  assert area_zero == 0 and adjusted_zero == 1.0
  assert torch.equal(unchanged, original)
  print(f"cutmix_smoke=pass adjusted={adjusted:.6f} area={area}")
  PY
  ```
- Tool skill: none.

## Abort Criteria
- Abort before launch if any tracked file except `train.py` changed, GPU 0 is not the required H20, code/style checks fail, or the helper smoke check fails.
- Abort the run on nonzero exit, timeout 124, NaN/Inf loss, traceback, CUDA OOM, or illegal memory access. Normalize progress carriage returns before bounded monitoring: `tr '\r' '\n' < run.log | grep -Ei 'loss: (nan|inf)|CUDA out of memory|illegal memory access|Traceback' | tail -n 5`. Detection may lag by at most the 50-step progress interval.
- Do not abort or modify configuration based on intermediate test accuracy. A clean result below 94.72% is a research no-improvement and is not retried with another seed or CutMix hyperparameter.

## Verification Protocol

### Verification Procedure
1. Confirm parent reference with the installed tree manager:
   `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 001`
   Require `metric=94.62`; threshold is 94.72%.
2. Confirm hardware with `timeout 10s nvidia-smi -i 0 --query-gpu=index,name,memory.total --format=csv,noheader`; require `0, NVIDIA H20, 97871 MiB`.
3. Execute with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`; require exit 0.
4. Require all ten final summary keys with `test "$(grep -Ec '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log)" -eq 10`.
5. Require the startup config line to report PreActWideResNet, 2,748,890 parameters, `cutmix_prob=0.5`, `cutmix_alpha=1.0`, `cutmix_end=0.75`, `cutmix_seed=42`, and the unchanged EXP-001 schedule/drop-path values. Inspect `git diff -U0 1feed19 -- train.py` during execution/analysis to confirm `t0` remains before CutMix and synchronization remains after the optimizer. Parse the final `cutmix:` line; require eligible > 0 and `0.45 <= applied/eligible <= 0.55`.
6. Require `299.5 <= training_seconds <= 301.0`, `total_seconds < 600`, and `num_params = 2,748,890`. Run `eval_count=$(grep -c 'eval ep' run.log); epochs=$(awk '/^num_epochs:/ {print $2}' run.log); test "$eval_count" -le "$epochs"` and `grep -Eq "eval ep +${epochs} " run.log` to confirm cadence and final evaluation.
7. Parse `best_test_acc`; require at least 94.72%. On failure, stop remaining pass-only verification and classify `no-improvement`.
8. On pass, collect the final 12 log lines and all approved informational metrics, record values in `03-execute.md`, and preserve `run.log` until analysis records the verdict.

### Informational Metrics (Optional)
- `final_test_acc`: final summary line in `run.log`.
- `final_test_loss`: final summary line in `run.log`.
- `training_seconds`: final summary line in `run.log`.
- `total_seconds`: final summary line in `run.log`.
- `startup_seconds`: final summary line in `run.log`.
- `peak_vram_mb`: final summary line in `run.log`.
- `num_epochs`: final summary line in `run.log`.
- `num_steps`: final summary line in `run.log`.
- `num_params`: final summary line in `run.log`.
- `cutmix_exposure`: final `cutmix:` line in `run.log`, tracked for mechanism audit only.

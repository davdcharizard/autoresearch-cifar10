# Plan EXP-045: Buy net-new epochs — compile-warmup off the timed budget + max-autotune

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md

## Milestones

### Milestone 1: Code changes implemented and seed-safety verified
- [ ] Edit `train.py` L190: `mode="reduce-overhead"` → `mode="max-autotune"`.
- [ ] Insert a compile-warmup block before `t_start_training` (L206): one forward+backward on a `torch.zeros(128,3,32,32)` channels_last batch through `compiled_model` under bf16 autocast, then `optimizer.zero_grad(set_to_none=True)`, then reset every BatchNorm2d's running stats, then `torch.cuda.synchronize()`. Do NOT call `optimizer.step()`.
- [ ] Verify seed-safety: dummy input is `torch.zeros` (consumes no RNG), model has no dropout, weights never stepped (no `optimizer.step` in warmup), BN buffers reset to pristine → the training-loop RNG state and initial weights/BN are byte-identical to baseline (NOT seed hacking; the run stays comparable).
- [ ] Sanity-check: `python -c "import ast; ast.parse(open('train.py').read())"` parses; warmup is before the `while` loop so per-step `dt` (and thus `total_training_time`) excludes compile.

### Milestone 2: Run launched on idle GPU and confirmed healthy
- [ ] Pick an idle GPU (`nvidia-smi`); shared node — avoid contention (infra-errors).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in background.
- [ ] Confirm within ~4 min: banner `ResNet-20 | params: 4,299,866`, compile completes (training steps begin), no traceback, loss falling. (max-autotune compile may take 1-3 min during startup — expected.)

### Milestone 3: Epoch count and dt verified — the KEY MEASUREMENT
- [ ] Extract dt distribution (`tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`), `num_epochs`, `num_steps`, `startup_seconds`, `total_seconds`.
- [ ] **This is the experiment's core signal**: did epochs rise above baseline ~91? Record steady dt (did max-autotune cut it below 8ms?) and startup_seconds (should now include the compile cost, confirming it moved off the budget). Confirm `total_seconds < 600` (10-min wall).

### Milestone 4: Accuracy verified against baseline
- [ ] Extract `best_test_acc`; compare to bar 96.32.

## Code Changes
- **train.py L190 (compile mode)**: `torch.compile(model, mode="reduce-overhead")` → `torch.compile(model, mode="max-autotune")`. More exhaustive kernel/epilogue autotuning; may lower steady-state dt below the 8ms floor (EXP-040 flagged max-autotune as the one untried dt reducer). Update the adjacent comment to note EXP-045.
- **train.py (compile-warmup, inserted before L206 `t_start_training`)**: add a warmup that triggers `torch.compile` compilation during startup so its one-time cost is charged to wall-clock (`startup_seconds`), NOT to the per-step-timed `total_training_time` (the 300s budget). Conceptual block:
  ```python
  # Compile warmup (EXP-045): trigger torch.compile compilation OUTSIDE the timed loop
  # so the one-time compile cost is billed to startup, not total_training_time. Seed-safe:
  # zeros input (no RNG), no optimizer.step (weights unchanged), BN buffers reset after.
  model.train()
  _wx = torch.zeros(BATCH_SIZE, 3, 32, 32, device=device, memory_format=torch.channels_last)
  _wy = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      _wloss = F.cross_entropy(compiled_model(_wx), _wy, label_smoothing=LABEL_SMOOTHING)
  _wloss.backward()
  optimizer.zero_grad(set_to_none=True)
  for _m in model.modules():
      if isinstance(_m, nn.BatchNorm2d):
          _m.reset_running_stats()
  torch.cuda.synchronize()
  ```
  This tests the hypothesis by reclaiming the ~14s compile cost (EXP-007) as net-new epochs, with the recipe otherwise byte-identical so any accuracy delta is attributable solely to epoch count.
- **Risks/edge cases**: (a) Warmup must run in `model.train()` mode to compile the exact training-path graph (train-mode BN + backward). (b) `reset_running_stats()` undoes the dummy forward's BN-buffer updates → BN starts pristine, identical to baseline. (c) Weights are never stepped → identical init. (d) Shape (128,3,32,32) matches every training batch (`drop_last=True`); eval uses the eager `model` handle (L267), so no recompile. (e) CUDA-graph capture (reduce-overhead/max-autotune) is warmed by the warmup; real steps replay — standard usage.

## Configuration Changes
- compile `mode`: `reduce-overhead` → `max-autotune` (rationale: the one untried dt reducer per EXP-040; reduce-overhead reaches the conv floor, max-autotune does a more exhaustive Triton search that may find faster conv kernels).
- New compile-warmup pass before the timed loop (rationale: moves the one-time compile cost off the 300s per-step budget → guaranteed ≈ +4-5 epochs; EXP-007 measured ~14s compile cost previously billed to the budget).
- No recipe/hyperparameter changes (optimizer, LR schedule, augmentation, label smoothing, Cutout, batch size, seed all identical).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background; harness re-invokes on completion.
- Resources: single idle NVIDIA H20 (shared node, idx 0/1). VRAM trivial. Fixed 300s training budget.
- Estimated runtime: ~7-9 min wall (max-autotune compile ~1-3 min in startup + 300s training + per-epoch evals). Must be < 10 min total.
- Log output: `run.log` in project root. dt lines use `\r` — extract via `tr '\r' '\n'`.
- Tool skill: none (local).

## Abort Criteria
- max-autotune compile errors out (Triton/autotune failure) → treat as code/infra failure; **retry once with `mode="reduce-overhead"`** (keeps the warmup → still gains ~5 epochs). This is the planned fallback for the brainstorm's wall-clock risk.
- Total wall-clock approaches/exceeds 10 min (the compile + training overruns the limit) → kill; retry once with `mode="reduce-overhead"` (much faster compile, ~14s).
- Loss diverges (NaN/inf) or fails to fall below ~1.0 in the first few epochs.
- No `dt:`/epoch-eval output after ~4 min (silent hang beyond expected compile time).
- GPU contention mid-run (dt steady-state ≫ expected while another job is co-resident) → discard as contention-confounded, rerun on idle GPU.

## Verification Protocol

### Verification Procedure
Baseline (from experiment index) = **96.22%**; bar = **96.32%** (baseline + 0.1).

1. **Run completes cleanly within budget** — `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:|^num_epochs:|^num_steps:|^startup_seconds:|^peak_vram_mb:" run.log`. Pass: `best_test_acc` present/non-empty, `total_seconds` < 600, `training_seconds` ≈ 300. Empty `best_test_acc` ⇒ crash (`tail -n 50 run.log`). Run timeout: 600s wall.
2. **Epoch-count signal (core measurement, not a pass/fail gate)** — `num_epochs` and dt distribution. Record whether epochs rose above ~91 and whether steady dt dropped below 8ms. `startup_seconds` should now include the compile cost (confirming it moved off the per-step budget). This determines whether the test of the saturation hypothesis was actually delivered (epochs genuinely added) — essential for interpreting the accuracy result in analysis.
3. **Primary necessary condition** — `grep -aE "^best_test_acc:" run.log`. Pass iff `best_test_acc ≥ 96.32`.
4. **No hard-constraint violations** — `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; warmup is seed-safe (zeros input, no optimizer.step, BN reset) so the run remains a fair comparison (NOT seed hacking).
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — the central signal (did we add epochs?).
- startup_seconds: `grep -aE "^startup_seconds:" run.log` — should rise (now includes compile), confirming the budget reclaim.
- dt distribution: from the `tr '\r' '\n'` extraction — did max-autotune cut steady dt below 8ms?
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.

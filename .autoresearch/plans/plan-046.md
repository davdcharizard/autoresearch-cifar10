# Plan EXP-046: Clean +5-epoch test — off-budget compile-warmup, reduce-overhead kernels

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md

## Milestones

### Milestone 1: Code change implemented and seed-safety verified
- [ ] Insert the seed-safe compile-warmup block before `t_start_training` (L206), KEEPING L190 `mode="reduce-overhead"` unchanged (the baseline's exact conv kernels — NO max-autotune).
- [ ] Use the EXP-045-debugged tensor construction: `torch.zeros(BATCH_SIZE,3,32,32,device=device).to(memory_format=torch.channels_last)` (NOT `memory_format=` inside `torch.zeros`).
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` parses; confirm L190 still reads `mode="reduce-overhead"` and the warmup is before the `while` loop.

### Milestone 2: Run launched on idle GPU and confirmed healthy
- [ ] Pick an idle GPU (`nvidia-smi`); shared node.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in background.
- [ ] Confirm within ~90s: banner `ResNet-20 | params: 4,299,866`, reduce-overhead compile completes (~14s, faster than max-autotune), training steps begin, loss falling, no traceback.

### Milestone 3: Epoch count / dt / startup verified — the KEY MEASUREMENT
- [ ] Extract `num_epochs`, `num_steps`, `startup_seconds`, dt distribution (`tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`).
- [ ] Confirm the warmup worked: `startup_seconds` rose above baseline ~2s (now includes the ~14s reduce-overhead compile), dt steady at **8ms** (baseline kernels, NOT the 7ms max-autotune saw), and `num_epochs` rose above baseline ~91 (target ~96). This confirms a clean +epochs-only test.

### Milestone 4: Accuracy verified against baseline
- [ ] Extract `best_test_acc`; compare to bar 96.32.

## Code Changes
- **train.py (compile-warmup, inserted before L206 `t_start_training`)**: identical to EXP-045's warmup but with `mode` left at `reduce-overhead` (L190 unchanged). Block:
  ```python
  # Compile warmup (EXP-046): trigger torch.compile compilation OUTSIDE the timed loop so the
  # one-time compile cost is billed to startup, NOT to total_training_time (timer starts after the
  # dataloader yields, L218). Seed-safe: zeros input (no RNG), no optimizer.step (weights unchanged),
  # BN buffers reset → training loop starts byte-identical to baseline. Reclaims ~14s → ~+5 epochs.
  model.train()
  _wx = torch.zeros(BATCH_SIZE, 3, 32, 32, device=device).to(
      memory_format=torch.channels_last
  )
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
  This tests the hypothesis by reclaiming the ~14s compile cost (EXP-007) as ~5 net-new epochs at byte-identical reduce-overhead kernels and recipe, so any accuracy delta is attributable solely to epoch count (de-confounding EXP-045's max-autotune kernel-numerics penalty).
- **NO other changes** — L190 compile mode stays `reduce-overhead` (baseline kernels); recipe/optimizer/schedule/aug/seed all identical.
- **Risks/edge cases**: warmup must run in `model.train()` (compile the training-path graph); `reset_running_stats()` undoes the dummy forward's BN updates; no `optimizer.step()` → weights stay at kaiming init; shape (128,3,32,32) matches every training batch (`drop_last`); eval uses the eager `model` handle → no recompile. All verified working in EXP-045 Run 2.

## Configuration Changes
- New compile-warmup pass before the timed loop (rationale: moves the ~14s one-time compile cost off the 300s per-step budget → ~+5 epochs; validated in EXP-045 where startup rose 2→79s, but here with the lighter reduce-overhead compile startup should be ~15-20s).
- compile `mode`: UNCHANGED at `reduce-overhead` (baseline kernels — the deliberate difference from EXP-045, to isolate the epoch effect from kernel numerics).
- No recipe/hyperparameter changes.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background; harness re-invokes on completion.
- Resources: single idle NVIDIA H20 (shared node, idx 0/1). VRAM trivial. Fixed 300s training budget.
- Estimated runtime: ~6-7 min wall (reduce-overhead compile ~14s in startup + 300s training + per-epoch evals). Must be < 10 min.
- Log output: `run.log` in project root. dt lines use `\r` — extract via `tr '\r' '\n'`.
- Tool skill: none (local).

## Abort Criteria
- Loss diverges (NaN/inf) or fails to fall below ~1.0 in the first few epochs.
- Traceback / OOM / shape error in `run.log`.
- No `dt:`/epoch-eval output after ~120s (silent hang).
- Total wall-clock approaches 10 min without the summary (reduce-overhead compile is ~14s, so this is unlikely; if seen, infra issue).
- GPU contention mid-run (dt steady-state ≫ 8ms while another job co-resident) → discard as contention-confounded, rerun on idle GPU.

## Verification Protocol

### Verification Procedure
Baseline (from experiment index) = **96.22%**; bar = **96.32%** (baseline + 0.1).

1. **Run completes cleanly within budget** — `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:|^num_epochs:|^num_steps:|^startup_seconds:|^peak_vram_mb:" run.log`. Pass: `best_test_acc` present/non-empty, `total_seconds` < 600, `training_seconds` ≈ 300. Empty `best_test_acc` ⇒ crash (`tail -n 50 run.log`). Run timeout: 600s wall.
2. **Clean +epochs verification (core measurement)** — `num_epochs` (target > ~91), `startup_seconds` (should include the ~14s compile, i.e. > baseline ~2s), dt distribution (should be steady **8ms** — baseline reduce-overhead kernels, NOT 7ms). This confirms the test was a clean single-variable +epochs change at baseline numerics.
3. **Primary necessary condition** — `grep -aE "^best_test_acc:" run.log`. Pass iff `best_test_acc ≥ 96.32`.
4. **No hard-constraint violations** — `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch; no new deps; seed 42 unchanged; warmup seed-safe (zeros/no-step/BN-reset).
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — did the +5 clean epochs materialize?
- startup_seconds: `grep -aE "^startup_seconds:" run.log` — confirms the compile cost moved off-budget (target ~15-20s for reduce-overhead).
- dt distribution: should be steady 8ms (baseline kernels) — distinguishes this from EXP-045's 7ms max-autotune.
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (expect ≈ baseline 491).

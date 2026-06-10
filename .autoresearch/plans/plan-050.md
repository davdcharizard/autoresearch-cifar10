# Plan EXP-050: Smaller batch size (128→64) for SGD gradient-noise regularization
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md

Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. Single-variable test of the only genuinely-untested axis left: batch-size DOWNWARD. Halving the batch at fixed LR doubles relative SGD gradient noise (∝ LR/√B) → Keskar "flatter minima → better generalization" regime. EXP-025's compute-bound finding predicts batch-64 ADDS ~2× updates (inverse of the batch-256 update-collapse) at similar total images; epoch-saturation (EXP-007/045/046) bounds underfit risk.

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] Change `BATCH_SIZE = 128` → `BATCH_SIZE = 64` in the hyperparameter block (single line). Nothing else changes.
- [ ] Smoke check: `python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff` shows exactly one changed line (`BATCH_SIZE`).
- [ ] Confirm num_params unchanged (4,299,866) — batch size does not affect param count — and `len(train_loader) == 50000 // 64 == 781` batches/epoch (drop_last=True).

### Milestone 2: Experiment running and throughput characterized
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm run.log is being written.
- [ ] Early signal: record dt (expect ~4.5-6ms, roughly half of baseline 8ms, since the net is compute-bound per EXP-025) and ep1 test_acc (normal range). No NaN/divergence in early steps despite the higher relative noise.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary block; `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `num_steps`, dt distribution, `peak_vram_mb`; compare to bar 96.32; record updates (num_steps, expect ~2× baseline ~35.5k) and epochs (expect ~70-85).

## Code Changes
- **train.py** (one line): `BATCH_SIZE = 128` → `BATCH_SIZE = 64`.
  - **Why this tests the hypothesis**: at fixed `PEAK_LR=0.2`, halving the batch doubles the relative gradient noise while keeping the mean update magnitude the same — the canonical small-batch→flat-minima generalization test. Single variable → clean attribution.
  - **Risks / edge cases**: (a) the `reduce-overhead` CUDA graph captures a fixed batch shape — batch 64 is still static, so it recompiles once cleanly (no graph break, dt stays low); (b) `drop_last=True` so the last partial batch is dropped (781 full batches/epoch); (c) if a launch-overhead floor keeps dt above ~half, epochs drop more (~70) → mild underfit (acceptable, and epoch-saturated); (d) LR/warmup deliberately left UNCHANGED — the goal is MORE noise at the same mean step, not the linear-scaling rule (which would preserve dynamics and defeat the test).

## Configuration Changes
- `BATCH_SIZE`: 128 → 64 (the single experimental variable; gradient-noise regularization).
- Unchanged: PEAK_LR 0.2, WARMUP_FRAC 0.05, MOMENTUM 0.9, WEIGHT_DECAY 1e-4, LABEL_SMOOTHING 0.1, CUTOUT_SIZE 16, width k=4, depth, Nesterov SGD, time-fraction cosine schedule, seed 42, compile mode reduce-overhead, TrivialAugment + Cutout.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (Bash `run_in_background: true`).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi` and launch on an idle GPU (GPU 1 used this session; both idle at last check).
- Estimated runtime: ~300s training + ~2s startup + eval ≈ 400s wall (< 600s limit). Smaller batch = more eval calls (more epochs of more steps) but eval is outside the training budget.
- Log output: all stdout/stderr → `run.log` in the project root.
- Tool skill: none (local run).

## Abort Criteria
- Loss goes NaN/inf or diverges (debiased loss climbing many steps) — the main risk from higher relative gradient noise at LR 0.2.
- dt rises ABOVE baseline 8ms (would indicate a graph break or contention) — kill and diagnose.
- No output / log not advancing for > 3 minutes after launch.
- Total wall-clock approaching 600s without a summary → kill (constraint breach).

## Verification Protocol

### Verification Procedure
Run after the experiment completes; stop at the first failed necessary condition.

1. **Get baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → baseline 96.22, bar **96.32**.
2. **Necessary condition 1 — `best_test_acc >= 96.32`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.32`; else no-improvement. (Absent `best_test_acc:` ⇒ crash → inspect `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, `total_seconds < 600`, `num_params == 4,299,866`. No NaN/traceback in run.log.
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only (and the diff is the single `BATCH_SIZE` line); prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.22: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~70-85 ep and ~2× baseline steps (~60-70k); characterizes the compute-bound throughput at batch 64.
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect ~4.5-6ms (vs baseline 8ms).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to baseline 0.195 (under-train would inflate it).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — expect lower than baseline (smaller batch).

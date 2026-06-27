# Plan EXP-000: Budget-matched modern training recipe (same ResNet-20)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md

## Milestones

### Milestone 1: Experiment branch + code changes implemented
- [x] Create experiment branch `exp/000-budget-matched-recipe` from `autoresearch/dev`
- [x] Implement all train.py changes listed under Code Changes below
- [x] Static check passes: `uv run python -c "import ast; ast.parse(open('train.py').read())"` and `uv run ruff check train.py`
- [x] Confirm `prepare.py`, `pyproject.toml`, `uv.lock` untouched: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal within ~90s: `grep "eval ep" run.log` shows epoch-1 eval completed and loss is finite (not nan/inf)

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty (empty = crash → read `tail -n 50 run.log`)
- [x] Total wall clock ≤ 10 min (check `total_seconds:` in summary)
- [x] Record full summary block into logs/exp-log-000.md

## Code Changes

- **train.py** (the only file modified — hard constraint):
  1. **Time-keyed one-cycle LR schedule** (fixes the demonstrated schedule-truncation defect). Remove `MultiStepLR`. Each step, compute `progress = min(total_training_time / TIME_BUDGET_S, 1.0)` and set LR on all param groups: linear warmup from ~0 to `PEAK_LR` over the first 15% of the budget, then cosine anneal to ~0 at 100% of the budget. Keying to elapsed-budget-fraction (not predicted step count) guarantees the anneal completes regardless of realized throughput. Tests hypothesis mechanism (a).
  2. **bf16 autocast**: wrap forward + loss in `torch.autocast("cuda", dtype=torch.bfloat16)`; backward/step outside the autocast context. No GradScaler needed for bf16. BN runs in fp32 under autocast (safe numerics).
  3. **TF32 + cuDNN autotune**: `torch.set_float32_matmul_precision("high")`, `torch.backends.cudnn.benchmark = True`.
  4. **channels_last**: `model.to(memory_format=torch.channels_last)` and convert input batches with `.to(memory_format=torch.channels_last)`. Tests hypothesis mechanism (b) together with items 2-3.
  5. **Large batch + scaled LR + Nesterov**: `BATCH_SIZE 128 → 512`, SGD with `nesterov=True`, peak LR 0.4 (linear scaling rule: 0.1 × 4; matches cifar10-fast's batch-512 recipe).
  6. **No weight decay on BN/bias**: two optimizer param groups — params with `ndim <= 1` (BN weight/bias, linear bias) get `weight_decay=0`; conv/linear weights get `weight_decay=5e-4` (cifar10-fast value for batch 512).
  7. **Label smoothing**: `F.cross_entropy(outputs, targets, label_smoothing=0.1)`.
  8. **DataLoader**: `persistent_workers=True` (epochs shrink to a few seconds; per-epoch worker respawn would dominate otherwise). Keep `num_workers=NUM_WORKERS`, `pin_memory=True`, `drop_last=True`.
  9. **MAX_STEPS**: raise to a non-binding value (`1_000_000`) so only the time budget governs run length.
  10. Keep everything else intact: seed 42 (no seed hacking — hard constraint), eval exactly once per epoch (hard constraint), the summary print block, per-step `synchronize()` timing (it meters the budget — do not touch).

## Configuration Changes
- BATCH_SIZE: 128 → 512 (better GPU utilization on H20; recipe basis: cifar10-fast)
- LR: 0.1 (constant base, step decay) → 0.4 peak, time-keyed one-cycle (linear scaling ×4 for batch 512; super-convergence/cifar10-fast)
- Momentum: 0.9 → 0.9 with `nesterov=True` (cifar10-fast)
- WEIGHT_DECAY: 1e-4 (all params) → 5e-4 (weights only), 0.0 (BN/bias) (standard modern CIFAR recipe; cifar10-fast)
- MAX_STEPS: 64000 → 1_000_000 (non-binding; time budget governs)
- Loss: plain CE → CE with label_smoothing=0.1
- Precision/layout: fp32/NCHW → bf16 autocast + TF32 + channels_last
- Schedule horizon: 64k steps (never reached) → 100% of TIME_BUDGET_S elapsed time

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM expected < 2GB (soft constraint, huge headroom)
- Estimated runtime: ~6–9 min total (300s training budget + startup + per-epoch evals; epoch count may rise to ~150–250 with higher throughput, adding eval overhead — see Abort Criteria for the 10-min cap)
- Log output: all stdout/stderr redirected to `run.log` in project root (do NOT tee/stream). Metrics extracted afterward via grep. `run.log` deleted after the experiment concludes (goal procedure).
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill the process, treat as failure (hard constraint from TASK.md)
- Training loss becomes NaN/inf, or epoch-1 eval accuracy < 15% (divergence under high peak LR) → kill, diagnose, fix LR/warmup before any rerun
- No output written to run.log within 120s of launch (hang — dataloader/driver issue) → kill and diagnose
- `grep "^best_test_acc:" run.log` empty after process exit → crash; read `tail -n 50 run.log` for the stack trace

## Verification Protocol

### Verification Procedure
Run from project root after the process exits. Baseline from `exp-index.sh baseline` on `.autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv` = **91.97%** at verification time of writing.

1. **Condition: run completes without crashing within the time budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` line present AND `total_seconds:` ≤ 600. Fail: empty grep (crash) or > 600s.
   - Timeout: if the process is still alive 10 min after launch, kill it → fail.
2. **Condition: best_test_acc exceeds baseline by ≥ 0.1 pp**
   - Command: `grep "^best_test_acc:" run.log` → parse the percentage.
   - Pass: value ≥ 92.07 (= 91.97 + 0.1). Fail otherwise. (Necessary-condition evaluation stops at first failure.)
3. **Condition: validation executed at most once per epoch**
   - Check: code review of the diff — the eval call remains the single `evaluator.evaluate(model, device)` per epoch-loop iteration; confirm `num_epochs:` ≥ count of `eval ep` lines in run.log (`grep -c "eval ep" run.log`).
   - Pass: eval-line count ≤ num_epochs. (Self-evident from unchanged loop structure; checked anyway.)

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — soft-constraint awareness
- num_epochs: `grep "^num_epochs:" run.log` — throughput proxy (baseline: 97)
- num_params: `grep "^num_params:" run.log` — should remain ~270k (architecture unchanged)

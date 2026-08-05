# Plan EXP-014: torch.compile throughput (off-budget warmup), headlined by compile-funded layer2 256→320

- **Created**: 2026-06-29

## Summary

Add `torch.compile` to the training forward, paying the one-time compilation OFF-BUDGET via a warmup
before `t_start_training`, and test whether the bought throughput (a) directly adds anneal epochs and
(b) funds a mild capacity widen (layer2 256→320) that previously under-annealed (EXP-007). All changes
are env-toggled so an unmodified invocation reproduces the baseline byte-for-byte. Three same-session
cells, each a SEPARATE `train.py` process under its own `timeout 600` (no single process breaches the
10-min wall): cell-0 (no-compile/256 control), cell-A (compile/256), cell-B (compile/320, headline).

## Milestones

### Milestone 1: Code changes implemented, baseline-preserving
- [ ] Add `import os` and four env-var reads (`USE_COMPILE`, `COMPILE_MODE`, `LAYER2_WIDTH`, `WARMUP`) with baseline defaults (off / "default" / 256 / off).
- [ ] Parametrize `ResNet9` layer2 width (layer2 conv_bn + GatedResidual + layer3 stem-in).
- [ ] Add `torch.compile` of a SEPARATE training-forward handle `train_fwd` + off-budget warmup gated on `USE_COMPILE or WARMUP` (exact static shape, BN-buffer snapshot/restore, local-RNG dummies, no optimizer.step, `warmup_seconds` print), placed AFTER ema construction and BEFORE `t_start_training`.
- [ ] Switch the in-loop training forward from `model(inputs)` to `train_fwd(inputs)`; leave EMA/eval on the uncompiled `model`/`ema_model`. Add first-10-step dt logging when `USE_COMPILE`.
- [ ] Add summary prints: `use_compile`, `compile_mode`, `layer2_width`, `warmup_on`.
- [ ] Verify byte-identical baseline reproduction: the unmodified default (`USE_COMPILE=0 WARMUP=0 LAYER2_WIDTH=256`) path must be logically identical to current code (the `train_fwd=model` alias, no warmup block, `layer2_width=256` default).
- **Check**: `git diff --stat` shows only `train.py`; `python -c "import ast,sys; ast.parse(open('train.py').read())"` parses clean.

### Milestone 2: Smoke test (compile correctness + off-budget invariant)
- [ ] Write `experiments/014/smoke.py` (pre-registered, exact — concern #2) that imports the REAL `ResNet9`/`conv_bn`/`compute_whitening_weight` from `train.py`, builds the real model+optimizer+criterion+`AveragedModel` EMA, and runs the ACTUAL warmup block. It must, **for BOTH `LAYER2_WIDTH=320` (headline) and `LAYER2_WIDTH=256`** (concern #3), assert:
  1. compile+warmup runs under bf16/channels_last at batch 512 with finite outputs; print `warmup_seconds` (cold-compile cost — concern #5).
  2. **param aliasing**: capture `train_fwd(x_fixed)` logits; do one REAL `optimizer.step()` on a nonzero grad; recapture — assert logits CHANGED (optimizer.step updates what the compiled forward reads).
  3. **BN restore**: assert every BN `running_mean/running_var/num_batches_tracked` equals its pre-warmup snapshot (dummy-data pollution undone).
  4. **off-budget invariant**: after warmup, time 3 train steps on real dataloader-shaped input — each `dt` < 60ms (no in-loop recompile).
  5. **eval-boundary recompile guard (4b)**: `evaluator.evaluate(ema_model, device)` (or `model` raw) → then `model.train()` + a `train_fwd` timed step → assert that step's `dt` < 60ms (train→eval→train transition does NOT recompile `train_fwd`).
  6. global RNG untouched: `torch.initial_seed()`/a post-`manual_seed(42)` draw matches the no-warmup path (local generator isolation).
  - Run: `CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$(pwd)/experiments/014/.inductor_cache uv run python experiments/014/smoke.py` → must print `SMOKE: ALL PASS` and the two `warmup_seconds`.
- [ ] **Wall-cap gate (concern #5)**: if either `warmup_seconds` is such that `warmup + ~300 (train) + ~150 (eval/startup)` approaches 600s (i.e. `warmup_seconds > 120`), STOP and reconsider (e.g. confirm cache reuse, or accept the risk explicitly) before launching official cells.
- **Check**: `SMOKE: ALL PASS`; both `warmup_seconds` recorded.

### Milestone 3: Run the three cells + verify
- [ ] **Before EACH cell** (concern #7): log `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` for GPU 1; if a foreign job is present, wait/note (contention confounds ABSOLUTE accuracy — same-session ranks still hold).
- [ ] Run cell-0, cell-A, cell-B as SEPARATE processes (each `timeout 600 env … uv run train.py > run_cX.log 2>&1`), sharing `TORCHINDUCTOR_CACHE_DIR=$(pwd)/experiments/014/.inductor_cache` (concern #4 — cache reuse is legitimate; it speeds the off-budget warmup but does NOT affect `training_seconds`/per-step throughput; cold cost already recorded in smoke).
- [ ] **Record the EXACT launched commands verbatim** in `03-execute.md` (concern #9 — any driver is gitignored under `.autoresearch/`, so the execute artifact is the integrity record; assert ONLY c0/cA/cB + the mandatory confirmation were run, no hidden retries/selection).
- [ ] Extract `best_test_acc`, `num_epochs`, steady `img/s`, `training_seconds`, `total_seconds`, `warmup_seconds`, `num_params` per cell; check first-10-step dt on compiled cells (no leak).
- [ ] Apply verification protocol; if any cell wins, run the mandatory PAIRED confirmation (winning cell + a fresh cell-0 control, same session — concern #6).
- **Check**: all three cells exit 0, `training_seconds`≈300, wall<600s each, valid metrics, no NaN, no first-step compile leak.

## Code Changes

- **train.py** (the ONLY editable file):
  1. **`import os`** at top (with the other stdlib imports).
  2. **Env-var reads** after the hyperparameter block (~line 31):
     ```python
     USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"
     COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
     LAYER2_WIDTH = int(os.environ.get("LAYER2_WIDTH", "256"))
     WARMUP = os.environ.get("WARMUP", "0") == "1"  # off-budget cudnn-autotune/allocator warmup, independent of compile
     ```
     Baseline defaults (all off / 256) → an unmodified `uv run train.py` reproduces EXP-008 exactly (no warmup, no compile).
     **Why `WARMUP` is separate (plan-review concern #8)**: with `cudnn.benchmark=True`, the off-budget warmup fwd/bwd ALSO prepays cuDNN kernel autotune + allocator growth for the static (512,3,32,32) shape. If only the compiled cells warmed up, cell-A-vs-cell-0 would conflate compile-fusion with prepaid-autotune. So ALL cells run with `WARMUP=1` (cell-0 included), making torch.compile the ONLY difference between cell-0 and cell-A. The warmup block runs whenever `USE_COMPILE or WARMUP`.
  3. **`ResNet9.__init__`** — add `layer2_width=256` param; use it for layer2 and the layer3 stem-in:
     ```python
     def __init__(self, num_classes=10, scale_out=SCALE_OUT, layer2_width=256):
         ...
         self.layer2 = nn.Sequential(conv_bn(128, layer2_width), nn.MaxPool2d(2), GatedResidual(layer2_width))
         self.layer3 = nn.Sequential(conv_bn(layer2_width, 512), nn.MaxPool2d(2), Residual(512))
     ```
     Construct with `ResNet9(NUM_CLASSES, layer2_width=LAYER2_WIDTH)`. `fc` stays `Linear(512,…)` (layer3 out unchanged). GatedResidual keeps ReZero α=0 identity init → no LR retune.
  4. **Compile + off-budget warmup**, placed AFTER `ema_model` construction (~line 257) and BEFORE `t_start_training` (~line 268). Do NOT rebind `model` (keeps eval/EMA uncompiled → zero eval recompile). The warmup runs on `USE_COMPILE or WARMUP` so cell-0 gets the SAME off-budget cudnn-autotune prepay as the compiled cells (concern #8):
     ```python
     train_fwd = torch.compile(model, mode=COMPILE_MODE) if USE_COMPILE else model
     if USE_COMPILE or WARMUP:
         t_warm = time.time()
         model.train()
         bn_backup = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                      for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
         gen = torch.Generator(device=device).manual_seed(0)  # LOCAL rng — global seed(42) untouched
         dummy = torch.randn(BATCH_SIZE, 3, 32, 32, generator=gen, device=device).to(memory_format=torch.channels_last)
         dtgt = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), generator=gen, device=device)
         for _ in range(3):
             optimizer.zero_grad(set_to_none=True)
             with torch.autocast("cuda", dtype=torch.bfloat16):
                 warm_loss = criterion(train_fwd(dummy), dtgt)
             warm_loss.backward()
         optimizer.zero_grad(set_to_none=True)  # discard warmup grads (no optimizer.step → params/momentum pristine)
         for m, rm, rv, nbt in bn_backup:  # undo dummy-data BN running-stat pollution
             m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nbt)
         torch.cuda.synchronize()
         print(f"warmup_seconds:   {time.time()-t_warm:.1f} (use_compile={USE_COMPILE})")  # OFF-budget; on the wall clock
     ```
     The `warmup_seconds` print is the cold-compile cost for the wall-cap audit (concern #5).
  5. **Training forward** (line ~301): `outputs = model(inputs)` → `outputs = train_fwd(inputs)`. This is the ONLY change in the timed loop. EMA update (`ema_model.update_parameters(model)`) and eval (`eval_target = ema_model` or `model`) stay on the uncompiled `model` — `train_fwd` shares the same parameter tensors via `_orig_mod`, so `optimizer.step()` on `model.parameters()` updates exactly what the compiled forward reads.
  6. **First-step dt logging to catch in-loop compile leakage (concern #1)**: a leaked compile inflates ONE step's `dt` (≈20–60s) and shows up as fewer steps, NOT as `training_seconds≪300`. So print `dt`/`img/s` for the FIRST 10 steps when `USE_COMPILE` (gate on `USE_COMPILE and step <= 10` alongside the existing `step % 50` print). A normal first step is ~15–25ms; a multi-second step-1 = compile leaked on-budget → abort.
  7. **Summary prints** (~line 379): add `print(f"use_compile:      {USE_COMPILE}")`, `print(f"compile_mode:     {COMPILE_MODE}")`, `print(f"layer2_width:     {LAYER2_WIDTH}")`, `print(f"warmup_on:        {USE_COMPILE or WARMUP}")`.

  **Why this tests the hypothesis**: the off-budget warmup isolates the compile cost from `training_seconds`; cell-A vs cell-0 measures the pure throughput→epochs effect; cell-B vs cell-A/cell-0 measures whether the bought throughput lets the 256→320 capacity anneal and clear the bar — directly attacking the EXP-007 under-anneal failure.

  **Risks/edge cases**: in-loop recompile if the warmup shape/dtype/layout mismatches the loop (mitigated: exact (512,3,32,32) channels_last bf16, `drop_last=True`); BN pollution from dummy data (mitigated: snapshot/restore); eval recompile toward the wall cap (avoided: eval is uncompiled); param aliasing (smoke-verified); `COMPILE_MODE="default"` avoids cudagraphs (reduce-overhead's static-input-address requirement conflicts with fresh dataloader tensors + EMA).

## Configuration Changes

Per-cell env (all three cells set `WARMUP=1` so the off-budget cudnn-autotune prepay is identical across them — concern #8):
- **cell-0** (control): `USE_COMPILE=0 WARMUP=1 LAYER2_WIDTH=256`
- **cell-A** (compile): `USE_COMPILE=1 WARMUP=1 LAYER2_WIDTH=256`
- **cell-B** (compile+capacity, headline): `USE_COMPILE=1 WARMUP=1 LAYER2_WIDTH=320`

- `USE_COMPILE`: `0` → `1` (cells A, B) — enable torch.compile of the training forward.
- `WARMUP`: `0` → `1` (ALL cells) — off-budget fwd/bwd to prepay cuDNN autotune + allocator (and, on compiled cells, the compilation). Equalizes the off-budget prepay so cell-A−cell-0 isolates compile-fusion.
- `COMPILE_MODE`: `default` (all compiled cells) — Inductor fusion without cudagraphs (safe with fresh per-step inputs + EMA). max-autotune deferred (higher off-budget compile, marginal extra fusion).
- `LAYER2_WIDTH`: `256` (cells 0, A) → `320` (cell B) — +1.03M params at the proven 8×8 stage; the EXP-007 pre-registered milder step (~1.25× layer2 cost vs 384's ~2.25×).

## Execution Environment

- **Method**: local, `CUDA_VISIBLE_DEVICES=1 uv run train.py` per cell with env-var overrides; three separate processes launched sequentially by one driver script, each wrapped in `timeout 600`.
- **Resources**: single GPU (NVIDIA H20), GPU **1** only (GPU 0 busy — hard constraint). VRAM ~1.6–2.0 GB of 98 GB (non-binding). 8 dataloader workers (from prepare.py).
- **Estimated runtime**: ~447s wall per cell (300s train + ~140s eval/startup) + ~30–90s off-budget compile warmup on the compiled cells → ~8–9 min per compiled cell wall (under the 600s cap; the warmup is off the 300s budget but ON the wall clock — watch it). Total ~22–25 min for three cells.
- **Log output**: each cell redirects to `experiments/014/run_c0.log` / `run_cA.log` / `run_cB.log` (`> run_cX.log 2>&1`). Driver script `experiments/014/run_cells.sh`. The per-cell log is the source of truth for metrics.
- **Tool skill**: none (local run).

## Abort Criteria

- **NaN/inf** in train loss or `best_test_acc` → abort that cell (research failure, do not retry).
- **In-loop recompile (compile leaked on-budget)** (concern #1): a leak inflates ONE step's `dt` (~20–60s) and surfaces as FEWER steps/epochs (NOT `training_seconds≪300`, since the loop runs until 300s). Detect via the first-10-step `dt` print on compiled cells: if any of steps 1–10 has `dt` > 1s → the warmup didn't cover the loop graph; abort, fix warmup shape/mode, re-run. Cross-check: cell-A `num_epochs` should be ≥ cell-0's; if cell-A has FEWER epochs than cell-0, suspect a leak (or zero throughput gain) — inspect the dt trace.
- **GPU 1 contention** (infra-errors EXP-010): if `nvidia-smi` shows a foreign job on GPU 1 OR a cell's img/s <18k / num_epochs trending <100 → results' ABSOLUTE accuracy is confounded (under-anneal). Same-session cells remain rank-comparable, but defer the comparable verdict to an uncontended re-run. Note in the log; do not trust absolute numbers.
- **Wall > 600s** for any cell → `timeout` kills it (exit 124); treat as failure (likely compile/eval-recompile blowup), investigate.
- **No output after 120s** from a cell → inspect; likely a hang in compile.

## Verification Protocol

### Verification Procedure

Operationalizes the goal's three necessary conditions. Baseline = **96.38** (`exp-index.sh baseline`), bar = **≥96.48** (+0.10pp).

1. **Pre-run host check**: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` — confirm GPU 1 has no foreign job before launching (infra-errors EXP-010). Record occupancy.

2. **Smoke (Milestone 2)** before timed runs — inline python on GPU 1, asserts compile correctness + off-budget invariant (param-aliasing, BN-restore, no in-loop recompile, finite outputs). Must print `SMOKE: ALL PASS`.

3. **NC1 — completion & budget** (per cell): from each `run_cX.log`,
   `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^num_params:" run_cX.log`.
   PASS iff exit 0, `training_seconds`∈[298,302], `total_seconds`<600, a valid `best_test_acc` printed, and no `nan`/`inf` in the log (`grep -ic "nan\|inf" run_cX.log` on the loss stream = 0). Empty grep ⇒ crash ⇒ FAIL (read `tail -50`).

4. **NC2 — improvement gate** (the win condition): let `c0 = cell-0 best`, `cB = cell-B best` (headline), `cA = cell-A best`. PASS iff the best compiled cell **≥ 96.48 AND > c0 + 0.10pp** (clears the stored baseline by ≥0.1pp AND beats the same-session no-compile control by >0.1pp — the stored 96.38 alone is too weak at the ~0.1pp noise floor). **Any win triggers a MANDATORY PAIRED confirmation (concern #6)**: re-run the winning cell AND a fresh `cell-0` control back-to-back in a NEW session (re-checking GPU-1 occupancy adjacent to both). The win COUNTS only if, in the confirmation session, the winning cell **≥ 96.48 AND > (confirmation cell-0) + 0.10pp** — i.e. it must reproduce BOTH the absolute bar AND the over-control margin (selecting the best of cell-A/cell-B in the first session inflates a single-session margin; the paired re-run controls for that + for contention drift). If no cell clears the bar → `no-improvement`.

5. **NC3 — integrity** (genuine method change): `git diff --quiet -- prepare.py` (byte-unchanged eval harness, else FAIL/invalid); only `train.py` modified (`git status --porcelain`); `manual_seed(42)`/`cuda.manual_seed(42)` intact (warmup uses a LOCAL generator); ≤1 `evaluator.evaluate` per epoch (unchanged from baseline); `num_params` = 7,784,627 for the 256 cells, and the expected +1.03M (~8.81M) for cell-B (320). No seed hacking, no eval circumvention.

6. **Pre-registered diagnostics (first-class reads, not pass/fail but decisive for interpretation)**: per-cell `img/s` (steady-state, from the per-step prints) and `num_epochs`.
   - cell-A `img/s` and `num_epochs` MUST exceed cell-0's (confirms compile bought throughput); the +% quantifies the throughput lever.
   - cell-B `num_epochs` ≥ ~120 = valid (above the ~110 under-anneal cliff); if cell-B `best==final` (monotone-rising to the end), that is the EXP-007 under-anneal signature → compile didn't buy back enough epochs and the capacity axis is then strongly exhausted at this budget.

### Informational Metrics (Optional)
- `peak_vram_mb`: `grep "^peak_vram_mb:" run_cX.log` — VRAM headroom (soft constraint).
- `training_seconds` / `num_epochs` / `num_steps`: `grep "^training_seconds:\|^num_epochs:\|^num_steps:" run_cX.log` — budget-fill + throughput diagnostic (the decisive read for this throughput experiment).
- `num_params`: `grep "^num_params:" run_cX.log` — confirms the 256 vs 320 capacity delta.
- Steady-state `img/s`: from the per-step progress prints (`grep "img/s" run_cX.log | tail -5`) — the throughput-lever magnitude.

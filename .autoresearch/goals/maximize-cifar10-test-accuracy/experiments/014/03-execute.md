# EXP-014: torch.compile throughput (off-budget warmup) + compile-funded layer2 256→320

## Execution

Overall Status & Info:
- **Created**: 2026-06-29
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-014
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (Run 2 clean; verdict no-improvement)

## Implementation Notes

### Summary
Implemented the plan's six `train.py` changes (Milestone 1), all env-toggled so the unmodified
default (`USE_COMPILE=0 WARMUP=0 LAYER2_WIDTH=256`) reproduces the EXP-008 baseline byte-for-byte:
(1) `import os` + four env reads (`USE_COMPILE`, `COMPILE_MODE`, `LAYER2_WIDTH`, `WARMUP`);
(2) `ResNet9.__init__` gains a `layer2_width` param threaded through layer2 (`conv_bn(128,W)` +
`GatedResidual(W)`) and the layer3 stem (`conv_bn(W,512)`); (3) a SEPARATE compiled training handle
`train_fwd = torch.compile(model)` plus the OFF-BUDGET warmup block (3 fwd+bwd at the exact static
(512,3,32,32) channels_last/bf16 shape on local-RNG dummies, no `optimizer.step`, BN buffers
snapshot+restored, `warmup_seconds` printed) placed after EMA construction and before
`t_start_training`; the warmup runs on `USE_COMPILE or WARMUP` so the no-compile control gets the
same off-budget cudnn-autotune prepay; (4) the in-loop forward switched `model(inputs)`→
`train_fwd(inputs)` (the ONLY timed-loop change; EMA/eval stay on uncompiled `model`); (5) first-10-step
dt trace on compiled cells (in-loop compile-leak detector); (6) summary prints for the new knobs.
Milestone 2 smoke (`experiments/014/smoke.py`) PASSED for BOTH widths. Milestone 3 driver
`experiments/014/run_cells.sh` runs the three cells as separate `timeout 600` processes.

### Surprises & Discoveries
- **Smoke needed `PYTHONPATH=$(pwd)`**: the smoke lives in `experiments/014/` so `from train import …`
  failed with `ModuleNotFoundError: train` until the project root was put on the path. The official
  cells run `train.py` from the project root directly, so they don't need it — but the driver sets
  `PYTHONPATH=$(pwd)` anyway for uniformity.
- **Compile cost is small & off-budget-clean**: cold `warmup_seconds` = 24.0s (width 320) / 12.4s (256),
  far under the 120s wall-cap gate. The post-warmup compiled step ran ~17ms @256 vs the historical
  uncompiled ~19.7ms (≈13% faster) — squarely in the predicted 7–15% band, and crucially the first
  post-warmup steps were NOT inflated (no in-loop recompile → compile genuinely stayed off-budget).
- **Eval-boundary is clean**: evaluating the uncompiled EMA copy + raw model, then resuming a
  `train_fwd` step, kept dt ~20–22ms — the train→eval→train transition does NOT recompile `train_fwd`
  (it's only ever called in train mode; eval goes through the uncompiled `model`).

### Decisions
- **Separate `train_fwd` handle, `model` never rebound** (plan-review concern #3): keeps all eval/EMA on
  the uncompiled module → zero eval recompile, no `OptimizedModule` deep-copy hazards in `AveragedModel`.
- **`WARMUP` as a distinct toggle from `USE_COMPILE`** (concern #8): all three cells warm up off-budget,
  so cell-A−cell-0 isolates torch.compile fusion from the cudnn-autotune/allocator prepay that the
  warmup also performs (with `cudnn.benchmark=True`).
- **`COMPILE_MODE=default`** (not reduce-overhead/max-autotune): avoids cudagraphs, whose static-input-
  address requirement conflicts with fresh per-step dataloader tensors + the EMA; default still fuses.
- **Smoke skips the ZCA whitening load**: frozen-conv values are irrelevant to compile/aliasing/BN/dt
  behavior; skipping the eigendecomp keeps the smoke fast. The official cells load whitening normally.

## Experimental Adjustments

- **Added `PYTHONPATH=$(pwd)` to smoke + driver invocations**: smoke import path fix. (ref: Run 1 setup — `ModuleNotFoundError: train`)

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash (driver `experiments/014/run_cells.sh`)
- **Log file(s)**: `experiments/014/run_c0.log`, `run_cA.log`, `run_cB.log` (+ driver stdout)
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-29
- **Ended**: pending

Description:
- Runs the three same-session cells back-to-back as separate `timeout 600` processes on GPU 1
  (uncontended at launch: GPU1 3 MiB / 0%): **cell-0** (`USE_COMPILE=0 WARMUP=1 LAYER2_WIDTH=256`, the
  no-compile control with off-budget warmup parity), **cell-A** (`USE_COMPILE=1 WARMUP=1 LAYER2_WIDTH=256`,
  pure compile throughput), **cell-B** (`USE_COMPILE=1 WARMUP=1 LAYER2_WIDTH=320`, compile-funded
  capacity — headline). Expected: cell-A shows higher img/s + more epochs than cell-0 (confirms compile
  bought throughput); cell-B holds epochs ≥~120 WITH +1.03M annealing capacity and is the only cell with
  a path to clear 96.48. Pre-registered first-class reads: per-cell `num_epochs` and steady `img/s`.

Observations:
- **INFRA-CONFOUNDED by GPU-1 contention (infra-error EXP-010).** GPU 1 was uncontended at launch (3 MiB/0%), but a foreign job (PID 1723342) appeared during cell-0 and RAMPED UP through the session (14→20+ GB, 100% util), escalating contention. Epoch counts fell monotonically across the session as the foreign job grew: cell-0 127 ep (mild), cell-A 74 ep (heavy), cell-B 64 ep (heavy) — vs the clean ~142–150 band. The compiled cells' first-10-step dt trace showed ~37–43ms (≈12–15k img/s) vs the smoke's uncompiled ~17ms — i.e. the slowdown is CONTENTION, not compile (smoke proved compile is FASTER on an idle GPU). (source: run_{c0,cA,cB}_contended.log; driver_contended.log [smi] lines; live nvidia-smi during run showed PID 1723342 at 20576 MiB/100%).
- All three cells executed END-TO-END correctly (exit 0, training_seconds 300.0, valid best_test_acc, no NaN) — the CODE is correct; only the absolute/comparative numbers are infra-confounded. The compile machinery worked (warmup off-budget, first-step dt not spiked beyond the uniform contention level, no in-loop recompile signature).
- **Verdict on Run 1: discard as infra-confounded; re-run all 3 cells uncontended (Run 2).** Per EXP-010, the cells are not even rank-comparable here because they were UNEQUALLY slowed (127 vs 74 vs 64 ep).

Key Metrics (Run 1 — CONTENDED, not comparable):
- Smoke (pre-run): `warmup_seconds` 24.0s@320 / 12.4s@256; post-warmup step ~17ms@256 (~13% over uncompiled ~19.7ms); BN-restore/param-aliasing/eval-boundary all PASS (source: smoke stdout).
- cell-0: best 96.25% @ 127 ep, warmup 8.5s (source: run_c0_contended.log).
- cell-A (compile/256): best 95.92% @ 74 ep, warmup 14.6s (source: run_cA_contended.log) — under-annealed by contention.
- cell-B (compile/320): best 95.93% @ 64 ep, warmup 13.4s, num_params 8,817,203 (source: run_cB_contended.log) — under-annealed by contention.

### Run 2 (clean re-run after Run 1 GPU-1 contention)

Metadata:
- **Job ID**: background bash (driver `experiments/014/run_cells.sh`, task bi4b4my3f)
- **Log file(s)**: `experiments/014/run_c0.log`, `run_cA.log`, `run_cB.log` (Run-1 logs preserved as `*_contended.log`)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-29
- **Ended**: 2026-06-29

Description:
- Identical 3-cell protocol, re-run because Run 1 was GPU-1-contention-confounded. GPU 1 was idle (0% util; foreign PID 1723342 holding memory but not computing) at Run-2 launch; cell-0 confirmed running at full ~26.6k img/s (clean). Same cells: c0 (no-compile/256 control), cA (compile/256), cB (compile/320 headline). Watching `num_epochs` per cell — a clean session should put cell-0 back in the ~142–150 band; if any cell drops far below, contention recurred and the run is again confounded.

Observations:
- **CLEAN run — GPU 1 uncontended throughout** (driver [smi] snapshots before each cell: GPU1 0–3% util; foreign PID 1723342 held memory but stayed idle). cell-0 back to 154 epochs (clean band), all exit 0, no NaN, prepare.py unchanged.
- **Throughput hypothesis CONFIRMED**: compile (cell-A) bought +19 epochs over cell-0 (154→173, +12%) at matched config — the off-budget warmup worked (warmup_seconds 11.8s, first compiled step 38ms then 17–22ms steady ~27–30k img/s, NO in-loop recompile). (source: run_cA.log compile-step trace + num_epochs).
- **But throughput does NOT convert to accuracy** — the net is anneal-saturated AND capacity-saturated at ~150 ep:
  - cell-A (compile/256, 173 ep): 96.32 vs same-session cell-0 96.29 = **+0.03pp** (sub-noise). Extra anneal epochs past ~150 give nothing → net is anneal-saturated (as the hypothesis pre-registered was likely).
  - cell-B (compile/320, 143 ep, headline): 96.21 vs cell-0 96.29 = **−0.08pp**. Crucially 143 ep is a HEALTHY anneal count (compile bought back the epochs 320 would've cost — without compile, 256→320 was ~120-130 region). Since cell-A proved epochs near 150 are worth ~0 (+0.03pp for +19 ep), cell-B's 11-epoch deficit vs cell-0 cannot explain −0.08pp → the 256→320 capacity GENUINELY doesn't help even when properly annealed. This resolves the EXP-007 under-anneal ambiguity: the layer2/8×8 capacity axis is EXHAUSTED, not epoch-starved.
- best==final for cell-A (96.32) and cell-B (96.21): not under-anneal (cell-A at 173 ep can't be epoch-limited and still only ties) — just where the noise/ceiling landed.

Key Metrics (Run 2 — CLEAN, comparable):
- cell-0 (no-compile/256): best **96.29%**, final 96.24, 154 ep, 14866 steps, warmup 8.9s, vram 1641.9 MB (source: run_c0.log).
- cell-A (compile/256): best **96.32%**, final 96.32, 173 ep (+12% vs c0), 16715 steps, warmup 11.8s, vram 1642.2 MB (source: run_cA.log).
- cell-B (compile/320): best **96.21%**, final 96.21, 143 ep, 13862 steps, warmup 11.8s, vram 1656.3 MB, num_params 8,817,203 (+1,032,576 = +1.03M as predicted) (source: run_cB.log).
- All cells: training_seconds 300.0, total_seconds 456/487/450s (<600 wall), exit 0, 0 nan/inf.

## Verification Results

(Run 2 = the clean comparable session; Run 1 discarded as GPU-1-contention-confounded.)

### Conditions Checked

- **NC1 — completion & budget**: PASS (all cells). All three exit 0; training_seconds = 300.0 each; total_seconds 456.5 / 487.3 / 450.7 (<600 wall); valid best_test_acc printed; `grep -ic nan|inf` = 0 for all. (source: run_{c0,cA,cB}.log summaries; driver.log exits).
- **NC2 — improvement gate**: **FAIL** → no-improvement. Best compiled cell = cell-A 96.32%. Required ≥96.48 AND > cell-0 (96.29) + 0.10pp = 96.39. cell-A 96.32 < 96.39 and < 96.48; cell-B 96.21 < cell-0. No cell clears the bar → NO win → no confirmation re-run triggered. (source: run_cA.log/run_c0.log best_test_acc).
- **NC3 — integrity**: PASS. `git status --porcelain` = only ` M train.py` (experiment artifacts live under gitignored `.autoresearch/`); `git diff --quiet -- prepare.py` clean (eval harness byte-unchanged); `manual_seed(42)`/`cuda.manual_seed(42)` both intact (warmup used a LOCAL generator); 1 `evaluator.evaluate` call (≤1/epoch); num_params 7,784,627 (256 cells) and 8,817,203 (320, +1,032,576 ≈ predicted +1.03M). No seed hacking, no eval circumvention.
- **Pre-registered diagnostics**: cell-A epochs 173 > cell-0 154 ✓ (compile bought ~+12% throughput, confirmed). cell-B epochs 143 ≥ 120 ✓ (valid, above the ~110 under-anneal cliff — NOT under-annealed). The throughput lever is real (~12%); it just doesn't convert to accuracy at this saturated operating point.

**Verdict**: no-improvement (NC2 result-quality gate failed; NC1/NC3 pass — a valid, clean negative).

### Informational Metrics

- peak_vram_mb: 1641.9 (c0) / 1642.2 (cA) / 1656.3 (cB) MB — non-binding (of 98 GB) (source: run_*.log).
- num_epochs: 154 (c0) / 173 (cA) / 143 (cB) — the decisive throughput diagnostic (source: run_*.log).
- num_steps: 14866 / 16715 / 13862 (source: run_*.log).
- num_params: 7,784,627 (256) / 8,817,203 (320) (source: run_*.log).
- warmup_seconds (off-budget): 8.9 / 11.8 / 11.8s — well under the 120s wall-cap gate (source: run_*.log).
- steady img/s: cell-0 ~26k; cell-A ~29k (+12%); cell-B ~24k (320 capacity cost partly offset by compile) (source: per-step prints).

## Errors & Dead Ends

### 2026-06-29 — smoke ModuleNotFoundError (resolved)
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: smoke.py runs from `experiments/014/`, so the project root (where `train.py` lives) was not on `sys.path`.
- Source: smoke stdout (first attempt).
- Do NOT retry: run smoke/any experiments-dir script with `PYTHONPATH=$(pwd)` from the project root.

### 2026-06-29 — Run 1 GPU-1 contention (recurrence of infra-error EXP-010)
- Error: no crash — foreign job (PID 1723342, ramping 14→20+ GB, 100% util) appeared on GPU 1 (`CUDA_VISIBLE_DEVICES=1`) during Run 1 cell-0 and escalated, halving our throughput. Epoch counts fell across the session (cell-0 127, cell-A 74, cell-B 64) — unequally slowed → not even rank-comparable.
- Root cause: GPU 1 is intermittently borrowed (GPU 0 always busy on this box). Pre-run nvidia-smi was clean; contention arrived mid-run.
- Source: live nvidia-smi (PID 1723342 @ 20576 MiB/100%); run_*_contended.log (127/74/64 ep); driver_contended.log.
- Do NOT retry under contention: re-ran all 3 cells (Run 2) once GPU 1 was idle → clean (cell-0 back to 154 ep, all comparable). Always read num_epochs vs the ~142–150 clean band as the contention tell; same-session controls only hold if ALL cells were equally (un)contended. Matches infra-errors.md EXP-010.

## Human Notes

> (none — autopilot)

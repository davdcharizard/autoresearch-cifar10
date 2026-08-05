# EXP-021: Compile-funded DEPTH at the 8×8 stage (2nd ReZero GatedResidual @ layer2)

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (verdict no-improvement — see 04-analysis.md)

## Implementation Notes

### Summary
Implemented all Milestone-1 changes in `train.py`, env-gated so the default (`USE_COMPILE=0 DEPTH=0 WARMUP=0`) reproduces the EXP-008 baseline: (1) `import os` + four env reads (`USE_COMPILE`, `DEPTH`, `WARMUP`, `COMPILE_MODE`); (2) `ResNet9(extra_depth=)` appends ONE extra `GatedResidual(256)` to `layer2`; (3) **RNG-isolated init** — build the BASE layer2/layer3/fc identically to the no-depth net, construct the extra block under `torch.random.fork_rng(devices=[])` and `.append()` it, then init all SHARED leaves via a `modules()` loop (skipping the extra block) and init the extra block under a second `fork_rng`; (4) the EXP-014 compile wrapper — separate `train_fwd = torch.compile(model)`, uncompiled `model`/EMA/eval, off-budget warmup (BN snapshot/restore, local-RNG dummies, 3 fwd+bwd, no optimizer.step) before `t_start_training`; (5) in-loop forward `model(inputs)`→`train_fwd(inputs)`; (6) per-epoch first-step dt recompile monitor; (7) summary prints (`use_compile`, `depth`, `warmup_seconds`). The official cells hold compile+warmup constant and toggle DEPTH only, so cA−c0 isolates depth.

### Surprises & Discoveries
- **Construction-time RNG divergence (caught by the smoke).** First RNG-isolation attempt built the extra `GatedResidual` *inside* the `layer2` Sequential. That consumes main-stream RNG at CONSTRUCTION (Conv2d `reset_parameters` → kaiming_uniform), shifting the RNG state before layer3/fc are built — so c0 and cA's SHARED weights diverged before the init loop even ran (smoke checks b/c FAILED). Fix: build the base layer2 identically, then construct the extra block under `fork_rng` and `.append()` it after fc, so its construction consumes ZERO main-stream draws. After the fix all 8 smoke checks PASS, including bit-equality of every shared param AND an identical post-construction global-RNG probe (proves c0/cA see the identical DataLoader-shuffle/aug stream → depth is truly isolated).
- **GPU-1 foreign contention present at execution start** (infra-errors EXP-010/014): a foreign compute job (PID 2724456 ~10 GB, 100% util; plus the persistent idle 3.8 GB holder PID 1723342) was on GPU 1. The timed sizing + official cells must wait for it to clear (correctness smoke is contention-tolerant and already passed).

### Decisions
- **Both official cells compiled + warmed** (`USE_COMPILE=1 WARMUP=1`), DEPTH toggled — isolates depth from compile/warmup (compile is math-equivalent per EXP-014). c0 = compiled control, cA = compiled depth (verdict cell).
- **No single-conv fallback as a verdict cell** (plan-review #2): if the full block fails the Milestone-2 sizing gate (predicted num_steps < 13095 ≈135 ep), abort EXP-021 to analysis as a valid negative; the single-conv probe is a separate future experiment.
- **Confirmation pair runs REVERSED** (cAb then c0b) to counterbalance monotonic GPU-drift bias (plan-review #6). Run logs in `/tmp` to keep the integrity gate clean (plan-review #7).

## Experimental Adjustments

- **Extra block constructed under `fork_rng` and appended after fc** (not inline in layer2) to fix construction-time RNG divergence between c0 and cA. (ref: smoke checks b/c — FAIL inline → PASS appended.)

## Run Log

### Milestone 1 — correctness smoke (no benchmark machinery)
- **Status**: completed — ALL 8 checks PASS (`/tmp/exp021_smoke.py`): forward shape/finite; shared-param bit-equality c0 vs cA; post-init RNG-stream identity; identity-init (forward bit-equal at init); +1,180,673 params; compile aliasing; BN snapshot/restore; eval-boundary; 2-step compiled loop. Ran under GPU-1 contention (correctness-only, tolerant).

### Milestone 2 — throughput sizing gate
- **Status**: completed — **GATE PASSES**. Compiled depth/base throughput ratio = 0.893 → predicted cA ~14,923 steps (~154 epochs) > the 13,095-step (135-ep) floor → run the FULL extra GatedResidual block. (Absolute img/s in the sizing script is low because it omits `cudnn.benchmark`; the back-to-back ratio is the robust quantity. Actual num_steps measured in the official run.) Source: /tmp/exp021_size.py stdout.

### Run 1 — Milestone 3 (official same-session pair c0/cA)
Metadata:
- **Job ID**: local background (detached `/tmp/exp021_run.sh`), GPU 1
- **Log file(s)**: `/tmp/exp021_run_c0.log`, `/tmp/exp021_run_cA.log`; sampler `/tmp/exp021_smi.log`
- **WandB**: N/A
- **Status**: completed (clean same-session pair via the retry-until-clean orchestrator)
- **Started**: 2026-06-30 12:12
- **Ended**: 2026-06-30 12:29

Description:
- Same-session pair on GPU 1 via `/tmp/exp021_orchestrate.sh` (a retry-until-clean wrapper added because GPU 1 was being aggressively borrowed — see Errors & Dead Ends): c0 (`USE_COMPILE=1 WARMUP=1 DEPTH=0`, compiled control) then cA (`DEPTH=1`, compiled depth — verdict cell), each `timeout 600`, with continuous foreign-job + img/s monitoring. Tested whether compile-funded depth lifts cA ≥96.48 AND >c0+0.1pp at ≥135-epoch anneal.

Observations:
- **Clean session** (orchestrator caught a clear window after the foreign job exited at 12:12; monitored throughout, no abort). c0 steady ~28.5k img/s, cA ~24.9k img/s (ratio 0.877 ≈ the 0.893 sizing estimate). GPU1 mem 3843→5872 MiB (all ours); no foreign >5GB spike during the pair. Recompile monitor: cA first-step-dt steady 20–24ms with only 2 early compile spikes (60–62ms) — no in-loop recompile leak. Both fully annealed (peaked-then-settled, not still-rising).
- **The depth cell TIES the control**: cA (compiled +1 GatedResidual@layer2, +1.18M params) 96.26 vs c0 (compiled, proven net) 96.29 = **−0.03pp**, at a HEALTHY 152-epoch anneal (num_steps 14671 ≥ 13095 gate). c0 96.29 is a normal control draw (matches EXP-014's compiled control 96.29/96.32). Depth at the proven 8×8 stage does not move the metric even when properly annealed.

Key Metrics:
- **c0 (compiled, no-depth)**: best 96.29 @173ep/16716 steps, final 96.11, 479.4s, peak_vram 1642.9 MB, num_params 7,784,627, per-epoch-max 96.29 ✓, ~28.5k img/s. (Source: /tmp/exp021_run_c0.log)
- **cA (compiled, depth)**: best 96.26 @152ep/14671 steps, final 96.22, 485.0s, peak_vram 1661.1 MB, num_params **8,965,300 (+1,180,673 = +1.18M as predicted)**, per-epoch-max 96.26 ✓, ~24.9k img/s, warmup 16.6s. (Source: /tmp/exp021_run_cA.log) → **−0.03pp vs c0** (tie, below floor).
- Properly annealed (NOT under-anneal): cA num_steps 14671 > 13095 (135-ep floor) and > 14065 (145-ep strong-anneal threshold); best 96.26 within 0.04 of final 96.22 → annealed, so this null CLOSES depth-at-8×8 (not inconclusive).

## Verification Results

### Conditions Checked

- **NC1 — completion & budget**: PASS (both cells). training_seconds=300.0 each; total_seconds 479.4 (c0) / 485.0 (cA) < 600; valid best_test_acc printed; `grep -ic nan|inf` = 0 both. (Source: /tmp/exp021_run_{c0,cA}.log)
- **NC3 — integrity**: PASS. `git status --porcelain` = only ` M train.py` (run logs in /tmp, autoresearch under gitignored `.autoresearch/`); `git diff --quiet -- prepare.py` clean (eval harness byte-unchanged); 2× `manual_seed(42)` intact (warmup + extra-block init used LOCAL/forked generators); 1 `evaluator.evaluate` call (≤1/epoch); per-epoch `best:` max == reported best both cells (no eval-cache/seed anomaly); cA num_steps 14671 ≥ 13095 (≈135-ep anneal floor — NOT under-annealed, the result is a fair depth verdict); num_params 8,965,300 = 7,784,627 + 1,180,673 (exactly the predicted +1.18M for one GatedResidual(256)). No seed hacking, no eval circumvention.
- **NC2 — improvement gate**: **FAIL** → no-improvement. Verdict cell cA (compiled depth) = **96.26%**. Required ≥96.48 AND > c0 (96.29) + 0.1pp = 96.39. cA 96.26 < 96.39 and < 96.48, and cA − c0 = −0.03pp (tie, marginally below the control). No cell clears the bar → NO win → no confirmation pair triggered.
- **Contention guard**: PASS (session clean). The retry-until-clean orchestrator monitored continuously and did NOT abort the winning attempt; both cells ran at clean throughput (c0 ~28.5k, cA ~24.9k img/s, both ≫ the 20k floor), GPU1 mem all ours (≤5872 MiB), no foreign >5GB job during the pair. c0's 173 ep matches EXP-014's clean compiled control (173 ep) — confirming the throughput was uncontended.

**Verdict**: no-improvement (NC2 result-quality gate failed at full anneal; NC1/NC3 pass — a valid, clean, properly-annealed negative).

### Informational Metrics

- peak_vram_mb: 1642.9 (c0) / 1661.1 (cA) MB — non-binding (depth adds ~18 MB).
- num_epochs / num_steps: 173/16716 (c0) / 152/14671 (cA) — cA fully annealed, throughput-ratio 0.877 (the depth block costs ~12% throughput, ≈ the 0.893 sizing prediction).
- num_params: 7,784,627 (c0) / 8,965,300 (cA, +1.18M) — confirms the single extra GatedResidual(256).
- warmup_seconds (off-budget): 11.3 (c0) / 16.6 (cA) — well under the 120s wall-cap margin.

## Errors & Dead Ends

### 2026-06-30 — GPU-1 aggressive foreign contention (recurrence of infra-errors EXP-010/014)
- Error: no crash — GPU 1 (`CUDA_VISIBLE_DEVICES=1`) was being repeatedly borrowed by churning foreign compute jobs (PIDs 2643063 → 2724456 → 2807153, each ~10–12 GB / 100% util). The first official launch ran the compiled c0 at ~12.6k img/s (HALF the ~28.5k clean rate) → would fit only ~76 epochs (severe under-anneal). Clocks were at max (1980 MHz, not throttled) — pure contention. A new foreign job landed within ~1 min of each clean window.
- Root cause: GPU 1 is intermittently borrowed on this box (GPU 0 always busy); contention can arrive MID-RUN even when the pre-launch check is clean.
- Source: live nvidia-smi (foreign PIDs >10GB on GPU1 UUID a5c2…); first run_c0 log (12.6k img/s, [compile] first_step_dt 41ms steady).
- Resolution: killed the contended run; built `/tmp/exp021_orchestrate.sh` — a retry-until-clean loop that (1) waits for no foreign >5GB job + util<30 (×2 checks), (2) launches the c0/cA pair, (3) monitors every 20s for a foreign >5GB job OR steady img/s < 20000 and ABORTS+retries the whole pair on either. It caught a clean window on attempt 1 once the foreign job exited (c0 173 ep = EXP-014 clean band; cA 152 ep; both ~25–28k img/s). Do NOT trust any GPU-1 run whose steady img/s is far below the ~26–29k clean compiled band — re-run the full same-session set once clean.

## Human Notes

> (none — autopilot)

# Plan EXP-059: Late batch-size step 512 → 1024 at p ≥ 0.75, LR unchanged — the noise-SCHEDULE measurement
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md

Baseline (exp-index): 96.71 @ 1990397 → bar = 96.81. σ ≈ 0.16, mean ≈ 96.57 (EXP-027); reads in (96.73, 96.81) are no-improvement by protocol.

**Idea**: tail-only gradient-noise reduction by pairing loader batches (effective 1024) after elapsed-budget fraction 0.75, LR schedule untouched (Smith et al. 2018 mechanism + fixed-time throughput dividend: 1024 measured ~41ms vs 2×22.5ms ⇒ ~9% more tail images). This is a GRAPH/SHAPE change → full EXP-055/056 machinery: dual-shape uncharged warmup, two-shape GPU probe before launch, phase-aware watchdog bands, probe-revised step-ledger integrity bands.

## Milestones

### Milestone 1: Code change + CPU sanity PASS
- [ ] Branch `autoresearch/exp-059` from `autoresearch/dev`; edit train.py (4 hunks, see Code Changes)
- [ ] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp059_sanity.py`): (a) params 4,286,026; (b) pairing logic unit test — the fetch wrapper yields 1024-shaped concat pairs when the switch condition holds and falls back to a single 512 batch on StopIteration (tested WITHOUT pin_memory — `pin_memory()` needs a CUDA context, so the CPU test exercises the concat/fallback path only; the pin call is exercised on GPU in Milestone 2's probe environment); (c) eager forward/backward at BOTH shapes (512 and 1024) succeeds with identical param count; (d) 3-step smoke at lr 0.01, losses decreasing

### Milestone 2: GPU probe — price BOTH compiled shapes (~2 min, gated)
- [ ] Probe script (`/tmp/exp059_gpu_probe.py`, EXP-055/056 pattern): gate-check GPU 0 free + load < 60; compile; warm 3 iters @512 then 3 @1024; time 40 steps @512 and 40 @1024 (`(time.time()-t0)/40*1000` each)
- [ ] **Pre-registered probe branches** (probe-to-run offset +0.5–1.0ms applies): P512 ∈ [21.5, 23.5] required (else environment suspect — re-gate); P1024 ≤ 46ms → full mechanism, launch; P1024 ∈ (46, 50] → throughput dividend dead, launch anyway as a noise-only test (record the revision); P1024 > 50ms → COST-CLOSURE: do not launch; verdict `invalid` (NaN) per the EXP-045 gate-kill precedent, class closed on cost
- [ ] Derive launch bands from probe: TAIL_THRESH = max(56, P1024 × 1.35); integrity steps band = 10,090 + 75,000/P1024_run ± 400; epochs band = 104 + (75,000/P1024_run)/49 ± 4 (P1024_run = P1024 + 0.5–1.0 offset). Record revised bands in the exp-log BEFORE launch.

### Milestone 3: Phase-aware launcher + gated launch
- [ ] Write `/tmp/exp059_composite.sh` = exp046 launcher verbatim EXCEPT the contention check: windows with PCT ≥ 75 compare against TAIL_THRESH instead of THRESH (straddle windows land below TAIL_THRESH — safe); everything else (gates, D0 gate-kill at 26ms on PRE-switch windows, NaN/divergence guards, 44×15s wall cap) unchanged
- [ ] Launch via Bash `run_in_background` + background until-grep watcher + TaskOutput(block=true, timeout=600000); GATE_DECISION D0 ∈ [21.5, 23.5] (pre-switch phase is byte-identical to family)

### Milestone 4: Run completes with two-phase clean signatures
- [ ] rc=0, no kill markers, no NaN; 300.0s charged; total ≤ 600s; params 4,286,026; evals ≤ epochs; ep1 ≥ 30
- [ ] Pre-switch windows 21.5–23.5ms; post-switch windows within [P1024_run − 1, P1024_run + 2]; steps and epochs within the probe-revised bands; a visible dt step-change at pct ≈ 75 in the watchdog telemetry (mechanism engagement BY PHYSICAL SIGNATURE per the EXP-055 rule — the dt jump to ~41ms IS the pairing proof; no recompile stall (no multi-second step at the switch — dual warmup covers both shapes)

### Milestone 5: Verification (first-failure-stop) + exp-log complete

## Code Changes
- **train.py** (only file; 4 hunks):
  1. Constants: `BATCH_SWITCH_FRAC = 0.75  # after this budget fraction, pair loader batches (eff. 1024); LR unchanged` after BATCH_SIZE.
  2. Compile warmup: after the existing 3-iter 512 warmup (before `optimizer.zero_grad`), add a second 3-iter loop at `2 * BATCH_SIZE` random data (same autocast/backward pattern, fresh `warm_x2`/`warm_y2`, deleted after) so BOTH shapes land in the inductor cache and cudnn.benchmark autotunes both — uncharged. zero_grad + synchronize after both loops.
  3. Training loop fetch path: `batch_iter = iter(train_loader)` then `for inputs, targets in batch_iter:`; BEFORE `t0 = time.time()`:
  ```python
  if total_training_time >= BATCH_SWITCH_FRAC * TIME_BUDGET_S:
      try:
          inputs2, targets2 = next(batch_iter)
      except StopIteration:
          pass  # odd batch at epoch end: run the unpaired 512 step (shape pre-warmed)
      else:
          inputs = torch.cat((inputs, inputs2), 0).pin_memory()
          targets = torch.cat((targets, targets2), 0).pin_memory()
  ```
  CPU concat + re-pin happens in the UNCHARGED fetch region (~2ms/step CPU, ~3.7s total over the tail — far inside the ~110s wall headroom; EXP-013 margin respected since no per-image transform is added). The charged region is byte-identical: the existing `.to(device, non_blocking=True).to(channels_last)` handles either shape; pin_memory preserves async H2D.
  4. Honest telemetry: `img_per_sec = int(inputs.size(0) / dt)` replacing `BATCH_SIZE / dt` (cosmetic; watchdog uses step/pct deltas, not img/s).
  Risks/edge cases: recompile-at-switch if warmup missed a shape (covered by hunk 2; detectable as a multi-second step in telemetry); concat drops channels_last (irrelevant — conversion happens on GPU in the charged region exactly as baseline); the unpaired-512 fallback step keeps both graphs warm so no shape thrash; lr_at(progress) is time-keyed and unaffected.

## Configuration Changes
- Effective tail batch: 512 → 1024 at p ≥ 0.75 (single switch; LR, schedule, WD, aug all unchanged). Rationale: canonical class representative — noise-down WITH the anneal, on-lattice shapes only, per brainstorm-059 evaluation.

## Execution Environment
- Method: local, GPU 0 ONLY (wait if busy), `/tmp/exp059_composite.sh` (phase-aware revision of the exp046 launcher) via Bash `run_in_background`; until-grep watcher on /tmp/exp059_composite_run1.log; TaskOutput block. GPU probe FIRST (Milestone 2) — also gated on free GPU + load.
- Resources: 1× H20 (~2–3GB), host load < 60
- Estimated runtime: probe ~2 min; run ~480–510s total (300.0s charged; ~141 epochs ≈ ~120s eval overhead)
- Log output: `run.log` (training, source of truth; deleted after analysis); /tmp/exp059_composite_run1.log (gate/watchdog)

## Abort Criteria
- Launcher-enforced: GATE_KILL D0 > 26ms (pre-switch windows); CONTENTION_KILL 4 consecutive windows over the PHASE threshold (pre: max(26, D0×1.25); post: TAIL_THRESH); STARTUP_KILL no steps by tick 12; NaN loss; divergence (eval < 15% after ep5); WALL_CAP 660s
- Experiment-specific: post-switch windows persistently ABOVE TAIL_THRESH ⇒ contention or recompile thrash, kill; a multi-second single step at the switch ⇒ recompile leak (warmup failed) — kill and fix warmup before relaunch (counts as the code-error retry)

## Verification Protocol

### Verification Procedure
First-failure-stop. Integrity pre-condition gates Condition 1; integrity failure → relaunch byte-identically (max 2), never analyzed. Probe cost-closure branch (P1024 > 50) bypasses launch entirely → verdict `invalid`, metric NaN, description records the measured price.

0. **Integrity pre-condition** (run.log + watchdog log; timeout 2 min): rc=0; D0 ∈ [21.5, 23.5]; no kill markers; pre-switch windows ≤ 27, post-switch windows ≤ TAIL_THRESH with a clean dt step-change at pct ≈ 75; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds ≤ 600`; eval lines ≤ num_epochs; steps and epochs within the PROBE-REVISED bands recorded in the exp-log at Milestone 2 (provisional at P1024=41: steps ≈ 11,920 ∈ [11,400, 12,400], epochs ≈ 141 ∈ [135, 145]); ep1 ≥ 30; no NaN.
1. **Condition 1 — metric beats baseline by ≥ 0.1pp**: `grep "^best_test_acc:" run.log` ≥ 96.81 PASSES. Pre-registered branches (all terminal):
   - (i) ≥ 96.81 → replicate-pair escalation (byte-identical second run; decision = MEAN ≥ 96.81)
   - (ii) ∈ [96.41, 96.73] → no-improvement; noise axis closed in BOTH level and schedule; documented frontier EMPTY
   - (iii) ∈ (96.73, 96.81) → no-improvement by protocol
   - (iv) < 96.41 → no-improvement, sign-down: tail gradient noise at lr→0 is load-bearing regularization; schedule class closed from below
   - (v) infra contamination → relaunch byte-identically (max 2)
2. **Condition 2 — completes within budget**: `grep "^total_seconds:" run.log` ≤ 600
3. **Condition 3 — validation ≤ once/epoch**: eval-line count ≤ num_epochs (structural)

Delete run.log after metrics are extracted.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~2,600–3,200 (1024-shape activations roughly double the 512 peak)
- num_epochs: `grep "^num_epochs:" run.log` — expect ~135–145 (two-phase ledger)
- num_params: must be 4,286,026

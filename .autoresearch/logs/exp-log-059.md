# EXP-059: Late batch-size step 512 → 1024 at p ≥ 0.75, LR unchanged

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-059
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Four hunks to train.py exactly as planned: (1) `BATCH_SWITCH_FRAC = 0.75` constant; (2) second 3-iter compile warmup at `2*BATCH_SIZE` random data (uncharged) so both shapes land in the inductor/cudnn caches; (3) fetch-path pairing — explicit `batch_iter = iter(train_loader)`, and before `t0` when `total_training_time >= BATCH_SWITCH_FRAC * TIME_BUDGET_S`, pull a second batch, `torch.cat(...).pin_memory()` (StopIteration → unpaired 512 step, shape pre-warmed); (4) honest `img_per_sec = inputs.size(0)/dt`. The charged region is byte-identical (existing `.to(device, non_blocking).to(channels_last)` handles either shape; pin preserved async H2D). CPU sanity ALL PASS: params exact; pairing unit test pre `[8×5]` / post `[16,16,8]` with fallback; eager fwd/bwd both shapes; smoke decreasing.

### Surprises & Discoveries
- First GPU probe attempt failed its own gate (load 60.59); second attempt at load 57 ran but came back CONTAMINATED: P512 = 26.40ms, outside the [21.5, 23.5] family band — host load just under the 60 gate still inflates a short probe. Ratio P1024/P512 = 1.893 → scaled to clean P512 ≈ 22.5 implies P1024 ≈ 42.6ms (consistent with EXP-012/022's ~41ms). Probe re-run gated at load < 40 per the plan's "P512 in band required, else re-gate" rule.

### Decisions
- Probe re-gate threshold tightened to load < 40 (vs the launcher's 60) — probes are 40-step samples and far more sensitive to host-side inflation than full runs averaged over 15s watchdog windows.
- Pairing condition uses `total_training_time` (charged seconds) rather than step-derived progress — identical to the lr_at progress key, so the switch lands at exactly p = 0.75 of the charged budget.

## Experimental Adjustments

## Run Log

### Run 0 (GPU probes — uncharged, pre-launch)

Metadata:
- **Job ID**: background tasks bvzhbk8sb (probe attempt 1, contaminated), by1zqajnn (probe attempt 2, load < 40)
- **Log file(s)**: task outputs (probe prints P512/P1024)
- **Status**: attempt 2 running (gate wait)

Description:
- Two-shape probe per plan M2: P512 must read [21.5, 23.5] (family band); P1024 branches — ≤ 46 full mechanism; (46, 50] noise-only; > 50 cost-closure, no launch.

Observations:
- Attempt 1 (load 57): P512 26.40 / P1024 49.98 — P512 out of band; initially suspected load contamination.
- Attempt 2 (load 37, task by1zqajnn): P512 26.45 / P1024 49.98 — IDENTICAL at low load ⇒ NOT contamination. Root cause: torch.compile's default automatic-dynamic-shapes — after warming a second shape, dynamo recompiles ONE dynamic graph covering both, whose kernels are ~18% slower at BOTH shapes (26.4 vs 22.5 family at 512).
- Fix: `torch.compile(model, dynamic=False)` → one static graph per shape, both pre-warmed uncharged. Attempt 3 (load 16, task bzat1evih): **P512 = 21.73ms ∈ [21.5, 23.5] PASS; P1024 = 40.89ms ≤ 46 → FULL-MECHANISM branch, launch approved.** Per-image dividend confirmed: 40.89 vs 2×21.73 = 43.46 (~6%).
- **Probe-revised bands (recorded pre-launch per plan M2)**: P1024_run ≈ 41.4–41.9 (probe + 0.5–1.0 offset); TAIL_THRESH = 57 (max(56, 41.9×1.35)); post-switch window band [40.7, 43.9]; steps ≈ 10,090 + 75,000/41.7 ≈ 11,889 → band [11,400, 12,350]; epochs ≈ 104 + 37 ≈ 141 → band [135, 145]; startup may rise a few seconds (two static compiles), still uncharged.

Key Metrics:
- P512 = 21.73ms | P1024 = 40.89ms (dynamic=False, load 16) — source: task bzat1evih output

### Run 1

Metadata:
- **Job ID**: background task bh0w5v7ez (composite), bnjlptk6l (gate watcher)
- **Log file(s)**: run.log (training); /tmp/exp059_composite_run1.log (gate/watchdog)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11
- **Ended**: 2026-06-11 (RC=0 at watchdog tick 35, ~514s)

Description:
- Single gated run of the late-batch-step recipe via /tmp/exp059_composite.sh (phase-aware watchdog: pre-switch threshold max(26, D0×1.25); post-switch (pct ≥ 75) TAIL_THRESH = max(56, P1024×1.35)). Expected two-phase signatures: pre-switch dt 22.0–22.8ms; post-switch dt ≈ P1024 + 0.5–1.0; steps ≈ 10,090 + 75,000/P1024_run; epochs ≈ 104 + (75,000/P1024_run)/49; params 4,286,026; ~480–510s total. Decision read vs pre-registered branches (plan-059): ≥ 96.81 replicate-pair; [96.41, 96.73] schedule-null (noise axis closed level AND schedule, frontier empty); (96.73, 96.81) no-improvement; < 96.41 tail-noise load-bearing, closed from below; infra → relaunch (max 2).

Observations:
- PRISTINE two-phase run. Gates poll 1; GATE_DECISION D0=22.5 ∈ band, tail_thresh armed 57. Pre-switch windows 21.7–22.7ms (ticks 6–25). **Mechanism engagement by physical signature (EXP-055 rule): clean dt step-change at tick 26 (pct 75.5, straddle 24.0ms) → ticks 27–34 read 40.5–42.0ms, inside the probe-predicted [40.7, 43.9] band (40.5 a hair under — probe offset ~0.4 smaller than the +0.5–1.0 historical). No recompile stall at the switch (no multi-second step) — dynamic=False dual-static-graph warmup worked.** slow_streak 0 throughout; RC=0 (source: /tmp/exp059_composite_run1.log)
- Step/epoch ledger ON the probe-revised bands: 11,933 steps ∈ [11,400, 12,350]; 142 epochs ∈ [135, 145]; startup 15.2s (two static compiles, uncharged); VRAM 3,157MB ≈ predicted ~3,100 (source: run.log summary)
- Trajectory family-shaped: ep1 34.08; converged plateau 96.44–96.51 at family test_loss (~0.186); final 8 evals tight (source: run.log eval lines)

Key Metrics:
- best_test_acc: 96.51 @ ep136 (source: run.log summary; branch (ii) — family band [96.41, 96.73], mean−0.4σ)
- final_test_acc: 96.50 / final_test_loss: 0.1856 @ ep142 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 513.8 | num_epochs: 142 | num_steps: 11,933 | num_params: 4,286,026 | peak_vram_mb: 3,157 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Integrity pre-condition** (plan-059 step 0): PASS — RC=0; D0 22.5 ∈ [21.5, 23.5]; no kill markers; pre-switch windows ≤ 22.7 (< 27), post-switch 40.5–42.0 ≤ 57 with the clean dt step-change at pct ≈ 75; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds: 513.8` ≤ 600; 142 evals ≤ 142 epochs; steps 11,933 ∈ [11,400, 12,350] and epochs 142 ∈ [135, 145] (probe-revised bands); ep1 34.08 ≥ 30; no NaN. (source: run.log + /tmp/exp059_composite_run1.log)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 96.51. Pre-registered branch (ii): ∈ [96.41, 96.73] family band → schedule-null; the noise axis is closed in BOTH level and schedule; the documented frontier is empty. First-failure-stop: no escalation.
- **Condition 2 — completes within budget**: PASS informationally (513.8 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (142/142, structural).

### Informational Metrics

- peak_vram_mb: 3,157.1 (1024-shape tail, as predicted) | num_epochs: 142 | num_params: 4,286,026 (informational; Condition 1 failed)

## Errors & Dead Ends

### 2026-06-11 — GPU probe contaminated at host load 57 (under the 60 launcher gate)
- Error: `no error — P512 26.40ms vs family band [21.5, 23.5]; probe gate passed at load 57.1`
- Root cause: 40-step probe samples are far more sensitive to host-load inflation than 15s watchdog windows; load 57/180 cores inflated both shapes ~17%.
- Source: task bvzhbk8sb output
- Do NOT retry: never accept a probe whose P512 anchor is out of band even if its own gate passed; re-gate at load < 40 for probes (launcher full-run gate stays at 60).

## Human Notes

> {Researcher can add comments, corrections, or context here}

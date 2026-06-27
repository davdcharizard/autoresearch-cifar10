# Plan EXP-065: Warmup-phase augmentation lightening (head-side quadrant of the pressure-profile law)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md

Baseline 96.71 @ 1990397 (bar ≥ 96.81; family mean 96.57, σ 0.16). Hypothesis per brainstorm-065: TA+RE off during the LR warmup (p < 0.15, ~21 epochs), on at full strength for the entire anneal; crop+flip always-on. ZERO dt toll (aug is CPU-worker-side; EXP-048) → no probe gate, family bands apply byte-identically. The intervention is a time-profile change of existing pressures — no graph change, no new pressure, no constants changed.

## Milestones

### Milestone 1: Code changes + CPU sanity
- [ ] train.py: shared-memory gate tensor `AUG_GATE = torch.zeros(1).share_memory_()` (0 = light, 1 = full; EXP-041-validated worker-propagation pattern); two tiny wrapper classes — `GatedTA(gate)` (applies TrivialAugmentWide iff gate ≥ 0.5, else identity; PIL stage) and `GatedErase(gate)` (applies the baseline RandomErasing iff gate ≥ 0.5; tensor stage) — replacing the bare TA/RE entries in train_tf at the SAME pipeline positions. Main loop: at each epoch top, `if AUG_GATE[0] == 0 and min(total_training_time / TIME_BUDGET_S, 1.0) >= WARMUP_FRAC: AUG_GATE[0] = 1.0; print(f"AUG_ON ep {epoch} ...")` — flip at the FIRST epoch boundary with progress ≥ 0.15 (one-time, monotone; up to 1 epoch + prefetch-depth lag is pre-accepted). Everything else byte-identical.
- [ ] CPU sanity (`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run python /tmp/exp065_sanity.py`): (a) gate=0 → GatedTA returns the input PIL object unchanged and GatedErase returns the input tensor unchanged (identity check); gate=1 → GatedErase modifies ~50% of a batch of tensors (p=0.5 statistical check over 200 samples, bound [0.35, 0.65]); (b) worker propagation: a probe Dataset whose __getitem__ returns float(gate[0]) through a persistent-workers DataLoader (workers=2, prefetch default) — after flipping the gate mid-iteration, fetched values become 1.0 within ≤ 2×prefetch batches; (c) full train_tf pipeline output shape (3,32,32), dtype float32 at both gate values; (d) lr_at unchanged at p ∈ {0.05, 0.15, 0.5, 1.0}; (e) wrappers are picklable/fork-safe (construct DataLoader with workers=2 and iterate without error using the real train_tf).
- [ ] Quick GPU spot-check NOT required (no graph change — the probe law applies to graph changes; the composite watchdog covers dt anomalies).

### Milestone 2: Gated composite run (family bands, no probe revision)
- [ ] /tmp/exp065_composite.sh (exp061-standard single-phase): dual gates (GPU-0 apps==0 AND load<60, poll 30s×240) → rm -f run.log → train background → watchdog 44×15s; GATE_KILL D0 > 26; CONTENTION_KILL 4 consecutive > max(26, D0×1.25); STARTUP_KILL no step lines by tick 12; NaN guard; divergence guard test acc < 20 at/after ep 10; WALL_CAP backstop.
- [ ] Launch via Bash run_in_background (stdout → /tmp/exp065_composite_run1.log) + until-grep watcher + TaskOutput(block=true).
- [ ] Clean completion: RC=0, no kill markers, D0 ∈ [21.5, 23.5], steps ∈ [13,100, 13,600].

### Milestone 3: Verification (first-failure-stop; integrity gates Condition 1)
- [ ] Integrity: RC=0; steps ∈ [13,100, 13,600] (family band, BINDING); epochs 135–141; eval lines = epochs (+ final partial); `num_params: 4,286,026`; `training_seconds: 300.0`; zero NaN; VRAM < 2,000MB; **mechanism engagement (EXP-055 law — marker + physical signature)**: exactly one `AUG_ON` line at the first epoch with progress ≥ 0.15 (expect ep 21–23), AND the debiased train loss VISIBLY rises within 2 epochs after AUG_ON (data hardening signature) after running visibly below-family before it (light-aug epochs train easier). A marker without the loss signature = mechanism not engaged → `invalid`.
- [ ] Condition 1: best_test_acc ≥ 96.81. Near-bar protocol (EXP-052): Run-1 ≥ 96.81 → one byte-identical replicate; PAIR MEAN decides.
- [ ] Condition 2: total_seconds ≤ 600.
- [ ] Condition 3: validation ≤ once/epoch (structural: unchanged every-epoch cadence).

## Code Changes

- **train.py** (only editable file):
  1. **Gate tensor** (module or main scope, created BEFORE the DataLoader so fork shares it): `AUG_GATE = torch.zeros(1).share_memory_()`.
  2. **GatedTA / GatedErase wrappers** (~12 lines): hold the gate + the baseline transform instance; apply iff `self.gate[0].item() >= 0.5`. Inserted in train_tf exactly where bare TA / RandomErasing sit today (PIL stage / post-Normalize) — pipeline order unchanged.
  3. **Flip logic** (3 lines at the top of the epoch loop): one-time gate set + `AUG_ON` print with epoch and progress. Uses the existing `total_training_time` accounting — no timer changes.
  4. Nothing else changes: constants, lr_at, model, optimizer, compile/warmup, timed step, eval, summary all byte-identical.
- **Why this tests the hypothesis**: the only delta vs baseline is WHEN TA+RE pressure applies (epochs ~22–139 instead of 1–139). Any metric movement is attributable to the head-phase pressure profile.
- **Risks/edge cases**: prefetched batches straddle the flip (pre-accepted, ≤ ~2 batches); workers must see the flip — guaranteed by fork + shared memory (EXP-041) and sanity-checked at (b); compile warmup uses random tensors, unaffected; BN sees a distribution shift at the flip — transient, >100 full-aug epochs follow (EXP-029 satisfied at eval).

## Configuration Changes
- None. All constants byte-identical (PEAK_LR 0.4, WARMUP_FRAC 0.15 — reused as the aug boundary, batch 512, WD/LS/momentum, transforms' own parameters). The change is the aug-pressure time profile only.

## Execution Environment
- Method: local, GPU 0 only; single gated run via /tmp/exp065_composite.sh; `uv run train.py > run.log 2>&1`
- Resources: GPU 0; VRAM ~1,613MB (family); host load gates (launch <60)
- Estimated runtime: ~450–540s total (300 charged + ~25s startup + ~139 evals × ~0.9s + stalls); charged exactly 300.0s
- Log output: run.log (truth); /tmp/exp065_composite_run1.log (gate/watchdog); delete run.log after the experiment

## Abort Criteria
- GATE_KILL: D0 > 26 at GATE_DECISION (dt should be family-band — any excess is contamination, not the intervention)
- CONTENTION_KILL: 4 consecutive windows > max(26, D0×1.25)
- STARTUP_KILL: no step lines by tick 12
- NaN in loss prints → kill (crash); test acc < 20% at/after ep 10 → kill (divergence)
- WALL_CAP backstop at watchdog window end

## Verification Protocol

### Verification Procedure
First-failure-stop; integrity gates Condition 1.

0. **Integrity** (2 min, run.log + composite log): RC=0, no kill markers; `num_steps` ∈ [13,100, 13,600]; `num_params: 4,286,026`; `training_seconds: 300.0`; zero NaN; VRAM < 2,000MB; exactly one AUG_ON marker at first epoch with p ≥ 0.15 AND the loss-rise engagement signature within 2 epochs of it (marker alone insufficient — EXP-055).
1. **Condition 1** — `grep "^best_test_acc:" run.log` ≥ 96.81 (baseline 96.71 + 0.1 via `exp-index.sh baseline`). Near-bar protocol (EXP-052): any Run-1 read ≥ 96.81 → one byte-identical replicate; the PAIR MEAN decides; max never decides.
2. **Condition 2** — `grep "^total_seconds:" run.log` ≤ 600.
3. **Condition 3** — eval lines ≤ epochs (structural).

Pre-registered branches: (i) pair-mean ≥ 96.81 → improvement, commit; (ii) ∈ [96.41, 96.73] → no-improvement; head quadrant is profile-neutral (banked gain and transition cost cancel or are both ~0) — pressure-profile law complete at four quadrants; (iii) < 96.41 → no-improvement; EXP-018-class transition damage confirmed on the data side — law extends to "pressure full-on from step 0"; (iv) > 96.73 single-draw but pair-mean < 96.81 → no-improvement, record the near-miss; (v) infra/contamination → byte-identical relaunch ≤ 2.

Mechanism checks (informational): early-phase debiased train loss vs family at matched epochs (light-aug speedup realized?); magnitude of the loss jump at AUG_ON; eval-acc trajectory across the flip (transient dip depth/recovery); plateau eval count within 0.15 of best.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1,613MB
- num_epochs: expect 137–140; num_steps: [13,100, 13,600]; num_params: 4,286,026
- AUG_ON epoch (expect 21–23)

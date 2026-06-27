# Plan EXP-061: Stage-1-heavy depth reallocation [3,3,3] → [4,3,2] at equal FLOPs
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md

Current baseline (exp-index.sh baseline): **96.71 @ 1990397** → success bar **≥ 96.81**. Family stats (EXP-027): mean ≈ 96.57, σ ≈ 0.16, family band [96.41, 96.73]. Uniform-net family signatures: dt 22.0–22.8ms, 13,400–13,515 steps, 138–140 ep; THIS variant changes the model, so step/epoch bands are PROBE-REVISED at launch (formula below). Params change by design: **4,286,026 → 3,179,338** (verified arithmetically in brainstorm-061; the same formula reproduces the uniform count exactly).

**Single variable**: per-stage block allocation — (3,3,3) → (4,3,2) at constant total depth 20, constant lattice widths 64/128/256, constant FLOPs (ResNet stage invariant: every block ≈ 151 MFLOPs), constant recipe (heat, noise, schedule, loss, augmentation, batch, numerics all untouched). Tests the favorable direction of EXP-017's measured allocation slope.

## Milestones

### Milestone 1: Code changes + CPU sanity
- [ ] train.py edited per Code Changes below (3 hunks)
- [ ] `/tmp/exp061_sanity.py` passes on CPU (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp061_sanity.py`):
  (a) params == 3,179,338; (b) structure: len(layer1)==4, len(layer2)==3, len(layer3)==2; stage widths 64/128/256; stride pattern (transition blocks stride-2 at layer2[0], layer3[0] only); (c) eager fwd/bwd at batch 16, output shape (16, 10); (d) 3-step smoke at lr 0.01, losses decreasing

### Milestone 2: GPU probe (dt pricing of the new shape, ~2 min)
- [ ] `/tmp/exp061_gpu_probe.py` on GPU 0 — gate: zero compute apps AND load < 40; `torch.compile(model)` exactly as train.py, 3-iter warmup, time 40 steps (P ms/step)
- [ ] **Pre-registered probe branches** (per-block law predicts P ≈ family 21.5–23.5; stage-1 blocks carry 4× activations so a modest rise is plausible):
  - P ≤ 23.5 → launch; probe-revised bands: steps ∈ [300000/(P+1.5), 300000/(P+0.1)] (run dt = probe + 0.5–1.0 historical offset, ±margin), epochs = steps/97.65 rounded ±2
  - 23.5 < P ≤ 26 → launch with the same band formula; record the priced deferral toll ((P − 22.4) ms ≈ −7 ep/ms ≈ −0.08pp/ms) in the exp-log so the read can be toll-adjusted in analysis (EXP-056 precedent)
  - P > 26 → DO NOT LAUNCH (would GATE_KILL anyway): the [4,3,2] point is starvation-priced like the on-lattice width increases — record as the experiment outcome (allocation-toward-stage-1 unmeasurable at acceptable dt; axis closed on cost grounds), verdict no-improvement pathway via the pre-registered kill branch (EXP-040/042 precedent)

### Milestone 3: Gated launch + watchdog
- [ ] Copy `/tmp/exp060_composite.sh` → `/tmp/exp061_composite.sh` (single-phase launcher with neutralized tail threshold, verbatim — thresholds: dual gates apps==0 & load<60 poll 30s×240; GATE_KILL D0>26; contention 4 windows > max(26, D0×1.25); STARTUP_KILL tick 12; NaN/divergence guards; WALL_CAP)
- [ ] Launch via Bash run_in_background + until-grep watcher + TaskOutput(block=true, timeout=600000); re-block on timeout (gate waits can exceed 1h)
- [ ] GATE_DECISION D0 within [probe−0.3, probe+1.3]; no kill markers

### Milestone 4: Completion + verification
- [ ] RC=0; summary block present in run.log
- [ ] Integrity ledger: steps/epochs within the probe-revised bands; `num_params: 3,179,338` (THE assert for this experiment — any other value means the architecture change didn't land); training_seconds 300.0; total ≤ 600; evals ≤ epochs; no NaN; ep1 ≥ 25 (relaxed for the architecture change; if ∈ [25, 30) note as architecture-shift observation — contamination is judged by the step ledger, not ep1)
- [ ] Verification per protocol; verdict per pre-registered branches

## Code Changes

All in `train.py` (the only editable file), 3 hunks:

1. **Constants**: replace `NUM_BLOCKS = 3` with
   ```python
   STAGE_BLOCKS = (4, 3, 2)  # per-stage block counts; uniform (3,3,3) is the baseline.
   # EXP-017 measured the mirror (2,3,4) at -0.28 with the deficit isolated to stage-1
   # depth; blocks are FLOPs-equal across stages, so this reallocation is FLOPs-neutral.
   ```

2. **ResNet.__init__**: accept per-stage counts —
   ```python
   def __init__(self, stage_blocks, num_classes=10, width_mult=1):
       ...
       b1, b2, b3 = stage_blocks
       self.layer1 = self._make_layer(w1, w1, b1, stride=1)
       self.layer2 = self._make_layer(w1, w2, b2, stride=2)
       self.layer3 = self._make_layer(w2, w3, b3, stride=2)
   ```
   `_make_layer` is already per-stage-count-capable — no change to it or BasicBlock.

3. **Call sites**: `ResNet(STAGE_BLOCKS, NUM_CLASSES, WIDTH_MULT)` in main(); the banner print becomes `ResNet-{2 * sum(STAGE_BLOCKS) + 2} [{...}x wide, stages {STAGE_BLOCKS}]` (still depth 20).

No other changes. Transforms, loss, optimizer, schedule, warmup, timed loop, eval all byte-identical to baseline.

## Configuration Changes
- STAGE_BLOCKS: (3,3,3) implicit → (4,3,2) (the variable under test; EXP-017 mirror evidence)
- num_params: 4,286,026 → 3,179,338 (consequence, not a tuned choice; capacity-down at constant FLOPs/depth is a never-visited corner)

## Execution Environment
- Method: local, GPU 0 only, via `/tmp/exp061_composite.sh` (gated launcher + inline watchdog), training command inside: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (no tee)
- Resources: GPU 0 (dual gate apps==0 AND load<60), VRAM expected ~1.8–2.2GB (more 32×32 activations than uniform's 1,613MB; soft constraint), ~10 of 180 host cores
- Estimated runtime: ~460–510s total (300s charged) at probe-band dt; gate wait variable
- Log output: `run.log` in project root (deleted after analysis); launcher telemetry → /tmp/exp061_composite_run1.log
- Tool skill: none (local). Launch pattern: Bash run_in_background → until-grep watcher → TaskOutput(block=true, timeout=600000), re-block on timeout.

## Abort Criteria
- GATE_KILL: D0 > 26ms (projected epochs make the bar arithmetically implausible — and fires the M2 P>26 cost-closure branch if the probe somehow passed)
- CONTENTION_KILL: 4 consecutive 15s windows > max(26, D0×1.25)
- STARTUP_KILL: no step lines by tick 12 (180s)
- NaN/divergence guards; WALL_CAP at watchdog end (~660s); >600s total = failure per goal
- Post-hoc step-ledger gate (EXP-058): num_steps below the probe-revised band ⇒ contaminated — relaunch byte-identically (≤2), never analyze

## Verification Protocol

### Verification Procedure

Step 0 — **Integrity pre-condition** (gates Condition 1):
`RC=0`; no kill markers in /tmp/exp061_composite_run1.log; D0 ∈ [probe−0.3, probe+1.3]; windows under threshold; steps/epochs within probe-revised bands; **`num_params: 3,179,338` exactly**; `training_seconds: 300.0`; `total_seconds ≤ 600`; eval count ≤ num_epochs; no NaN; ep1 ≥ 25. Ledger failure ⇒ branch (v) contamination ⇒ byte-identical relaunch (≤2), never analyze.

Condition 1 — **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1):
`tr '\r' '\n' < run.log | grep "^best_test_acc:"` — numeric compare. First-failure-stop applies.

Condition 2 — **completes within budget**: `total_seconds ≤ 600` from the summary block.

Condition 3 — **validation at most once per epoch**: eval-line count ≤ num_epochs (structural).

**Pre-registered decision branches** (all terminal):
- (i) best ≥ 96.81 → run ONE byte-identical replicate (EXP-052 protocol); decide on the PAIR MEAN ≥ 96.81 → improvement (commit) else no-improvement. (A structural change with an in-project slope prior is exactly the near-bar case the replicate-pair protocol exists for.)
- (ii) best ∈ [96.41, 96.73] → no-improvement; allocation curve is flat-topped at uniform — axis closed BIDIRECTIONALLY ([2,3,4] −0.28 / [3,3,3] mean / [4,3,2] band); note: if probe priced a toll (M2 branch 2), report the toll-adjusted read informationally (EXP-056 precedent) but the verdict stays band-based
- (iii) best ∈ (96.73, 96.81) → no-improvement (noise-band high read; never single-draw promoted)
- (iv) best < 96.41 → no-improvement; stage-3 block COUNT is load-bearing (its added params were not, per EXP-017 — the distinction is the new datum); allocation axis closed from below
- (v) integrity failure → infra; relaunch byte-identically (≤2)
- (vi) M2 P > 26 probe branch → no-improvement via cost-closure (recorded without a full run, EXP-040/042 precedent)

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect 1.8–2.2GB; soft constraint)
- num_epochs: `grep "^num_epochs:"` (probe-revised band)
- num_params: `grep "^num_params:"` (must be exactly 3,179,338)

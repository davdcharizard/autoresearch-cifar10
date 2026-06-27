# Plan EXP-046: Anti-aliased shortcut — avg-pool the identity path at stage transitions
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

Projection: zero parameters added (params stay **4,286,026**); `F.avg_pool2d` fwd+bwd at the two transition sites is a fraction of a ms → expected dt ≈ 22.4–23.0ms, ~136–139 epochs — baseline-identical signatures. The change is pure information-routing on the identity path.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py` `BasicBlock.forward`: replace `shortcut = shortcut[:, :, :: self.stride, :: self.stride]` with `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)` (channel zero-pad line unchanged; nothing else in the file touched)
- [x] CPU sanity (CUDA_VISIBLE_DEVICES="", `sys.path.insert(0, <project root>)`):
  - param count == **4,286,026** exactly (zero-param change)
  - forward (4,3,32,32) → (4,10) finite
  - semantic check: on layer2[0] with a CONSTANT-valued input tensor, new shortcut == old sliced shortcut (both reduce a constant to itself); on a random input they DIFFER (anti-aliasing active) while shapes match
  - stride-1 blocks bit-identical to baseline behavior (need_pad False everywhere except layer2[0]/layer3[0] — assert)
  - 2-step train smoke, finite decreasing loss

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Build `/tmp/exp046_composite.sh` from `/tmp/exp045_composite.sh` with GATE threshold **26ms** and contention floor 26 (sed, diff-confirm 3 lines); launch in background
- [x] Confirm run.log being written / step prints appearing (GATE_DECISION D0=22.5ms projected_epochs=137)

### Milestone 3: Run resolved — full completion OR pre-registered gate/abort branch
- [ ] If GATE_KILL (D0 median > 26ms): record measured dt → branch (iii): avg_pool2d misprices under compile+channels_last (unexpected for a stock dense-regime kernel), verdict `invalid`, NO relaunch
- [ ] If CONTENTION_KILL/STARTUP_KILL: relaunch byte-identically once dual gates re-clear (max 2, infra)
- [x] If completed: rc=0 + summary block present (best 96.65, 139 ep, dt 22.0–22.7ms throughout)

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition (PASS, pristine), then necessary conditions in order (cond 1 FAIL: 96.65 < 96.81, below replicate band); results in exp-log-046.md

## Code Changes
- **train.py** (only file, one line of logic in `BasicBlock.forward`): the 2016 pad shortcut downsamples by strided slice `[::2, ::2]`, discarding 75% of the identity signal and aliasing the rest; replace with `F.avg_pool2d(shortcut, self.stride)` (the 2×2 box filter — simplest anti-aliasing, information-preserving). Channel zero-padding stays as-is (EXP-020 proved it harmless). Affects exactly two forward sites (layer2[0], layer3[0]); zero parameters, no init change, no schedule/optimizer/noise interaction. Risks: stock-kernel mispricing (gate covers); subtle train/eval asymmetry NONE (the op is deterministic and mode-independent).

## Configuration Changes
- None. Every training constant, the architecture's parameter set, and all signatures remain baseline (recipe certified: EXP-007…036). This is intentionally a pure function-quality probe.

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` — the validated exp045 script with thresholds 31→26:
  1. Dual launch gates (GPU-0 zero compute apps AND 1-min load < 60), poll 30s × 240 (infra-errors EXP-032/011)
  2. `rm -f run.log`; `uv run train.py > run.log 2>&1 &` (no tee)
  3. Watchdog 44 × 15s, windowed dt from pct deltas (≥200-step windows only):
     - **GATE_KILL**: median of first 3 windows > **26ms** (off-rung; expected ~22.6; >26 means the added kernel costs >3.4ms = something is wrong)
     - **CONTENTION_KILL**: 4 consecutive windows > max(D0×1.25, 26ms)
     - **STARTUP_KILL**: no step prints by tick 12 (~180s); **NaN guard**; **divergence** (eval < 15% after ep5); **WALL_CAP** tick 44 (~660s)
  4. On exit: rc, summary greps, last-8 evals
- Resources: GPU 0 only (H20); VRAM ~1.5GB (baseline-like); CIFAR-10 cached in `data/`
- Estimated runtime: ~480s clean run (startup ~12–25s [new graph → possible inductor cache miss], 300s charged, ~138 evals ≈ 120–180s uncharged); GATE_KILL branch ~90s
- Log output: `<project root>/run.log` + composite stdout
- Tool skill: none (local)

## Abort Criteria
- GATE_KILL D0 > 26ms → pre-registered branch (iii): verdict `invalid`, metric NaN, record the kernel datum; do NOT relaunch or rework the op this loop
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- NaN loss or eval < 15% after epoch 5 → research failure, no retry
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition** (guards false readings): pristine profile — ≥200-step windows mean ≤ ~24ms, none > 27 (off-rung); num_epochs within 130–142 (projection ~138; below ~125 implies contention or hidden kernel cost — cross-check pct deltas); printed params == 4,286,026; training_seconds == 300.0; eval lines ≤ num_epochs. Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → run the pre-registered replicate pair (two byte-identical gated runs; improvement only if mean of runs ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log` (vs ~138 — also the avg-pool dt datum)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — unchanged)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185 ⇒ absorbed-null branch; below ⇒ information gain real)

# Plan EXP-045: 64/192/256 gate-first — the last kernel-lattice capacity point
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

Projection arithmetic (dense law, valid ONLY if 192 is on the fast lattice): ΔFLOPs ≈ +40% (5 inner stage-2 convs ×2.25 + two half-unit transitions ×1.5 over ~17 conv units) → dt ≈ 22.4 + 0.40×13.3 ≈ **27.7ms** → ~112 epochs; deficit ≈ 27 ep × 0.014 ≈ **−0.38**. Expected params: baseline 4,286,026 + 1,105,920 conv + 768 BN = **5,392,714**.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py`: `STAGE_WIDTHS = (64, 192, 256)` via the same three-edit pattern as EXP-044 (constant + `ResNet.__init__(.., widths=...)` + print line); everything else byte-identical to baseline @ 1990397
- [x] CPU sanity (CUDA_VISIBLE_DEVICES="", `sys.path.insert(0, <project root>)`):
  - param count == **5,392,714** exactly (if mismatch, STOP and re-derive)
  - forward (4,3,32,32) → (4,10) finite
  - pad shortcuts: stage2 block1 pads 128 (192−64), stage3 block1 pads 64 (256−192), identity elsewhere
  - 2-step train smoke, finite loss

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Build `/tmp/exp045_composite.sh` from `/tmp/exp044_composite.sh` with GATE threshold **31ms** and contention threshold max(D0×1.25, 31); launch in background
- [x] Confirm run.log being written / step prints appearing (STARTUP guard tick 12 covers failure)

### Milestone 3: Run resolved — full completion OR pre-registered gate/abort branch
- [x] If GATE_KILL (D0 median > 31ms): record measured dt → branch (iii): 192 misprices, fast lattice = {64,128,256} exactly, verdict `invalid`, NO relaunch (the lattice has no further points — this closes the class)
- [ ] If CONTENTION_KILL/STARTUP_KILL: relaunch byte-identically once dual gates re-clear (max 2 relaunches, infra)
- [ ] If completed: rc=0 + summary block present

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition, then necessary conditions in order; results recorded in exp-log-045.md

## Code Changes
- **train.py** (only file): identical mechanics to EXP-044 — `STAGE_WIDTHS = (64, 192, 256)` constant (comment noting baseline (64,128,256)), `ResNet.__init__` widths-tuple signature, print line `64/192/256 asymmetric`. Rationale: the unique surviving within-lattice capacity configuration (all widths ≡ 0 mod 64, all ≤256); tests both 192's kernel status and — if fast — the capacity-at-converged-epochs question. `BasicBlock` pad shortcut handles 64→192 (pad 128) and 192→256 (pad 64) unchanged. Risk: 192 mispricing (EXP-034's 27.4ms fallback datum) — handled by the gate, not by code.

## Configuration Changes
- Stage widths: (64, 128, 256) → (64, 192, 256) (+40% FLOPs, +1.107M params → 5.39M; dense-law dt 27.7ms → ~112 epochs; brainstorm-045 Idea 1 — last lattice permutation with viable arithmetic)
- All training constants unchanged (recipe certified: EXP-007…036)

## Execution Environment
- Method: local, composite background script `/tmp/exp045_composite.sh` — copy of the validated exp044 script with two threshold edits:
  1. Dual launch gates (GPU-0 zero compute apps AND load < 60), poll 30s × 240 (infra-errors EXP-032/EXP-011)
  2. `rm -f run.log`; `uv run train.py > run.log 2>&1 &` (no tee)
  3. Watchdog 44 × 15s, windowed dt from pct deltas (≥200-step windows; never printed dt, never 50-step windows):
     - **GATE_KILL**: median of first 3 windows > **31ms** (off-rung; dense-law pass shows ~28, EXP-044-style mispricing lands ≥35; 31ms projects ~100 epochs = arithmetic implausible given the −0.38-at-112 baseline deficit)
     - **CONTENTION_KILL**: 4 consecutive windows > max(D0×1.25, 31ms)
     - **STARTUP_KILL**: no step prints by tick 12 (~180s)
     - **NaN guard**; **divergence** (eval < 15% after ep5); **WALL_CAP** at tick 44 (~660s)
  4. On exit: rc, summary greps, last-8 eval lines
- Resources: GPU 0 only (H20); VRAM ~2–3GB expected; CIFAR-10 cached in `data/`
- Estimated runtime: ~500s clean run (startup ~12–25s, 300s charged, ~112 evals ≈ 100–150s uncharged); GATE_KILL resolves in ~90s
- Log output: `<project root>/run.log` (source of truth) + composite stdout (watchdog lines)
- Tool skill: none (local)

## Abort Criteria
- GATE_KILL D0 > 31ms → pre-registered branch (iii): verdict `invalid`, metric NaN, capacity class closed on hardware grounds; do NOT relaunch or substitute another width this loop
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- NaN loss or eval < 15% after epoch 5 → research failure, no retry
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition** (guards false readings, not a goal condition): pristine profile — ≥200-step windows mean ≤ ~29ms, none > 34 (off-rung); num_epochs within 100–122 (projection ~112; below ~97 implies contention — cross-check pct deltas); printed params == 5,392,714; training_seconds == 300.0; eval lines ≤ num_epochs. Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → run the pre-registered replicate pair (two byte-identical gated runs; improvement only if mean ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log` (vs ~112 projection — also the 192-pricing datum)
- num_params: `grep "^num_params:" run.log` (expect 5,392,714)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185 ⇒ level-saturation branch; clearly below ⇒ capacity was binding)

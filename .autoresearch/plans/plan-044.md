# Plan EXP-044: Within-cliff asymmetric widening — stage widths 64/160/256 (dt-gated)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py`: replace the uniform `WIDTH_MULT` width derivation with explicit `STAGE_WIDTHS = (64, 160, 256)`; `ResNet.__init__` takes the widths tuple; print line updated to show stage widths; everything else byte-identical (blocks, init, transforms, loader, optimizer, schedule, warmup, timed loop, eval cadence)
- [x] CPU sanity (CUDA_VISIBLE_DEVICES="" python script with `sys.path.insert(0, <project root>)` — /tmp scripts lose the project root otherwise):
  - param count == **4,793,290** exactly (analytic: baseline 4,286,026 + 506,880 conv + 384 BN; if mismatch, STOP and re-derive before launch)
  - forward of a (4,3,32,32) batch returns (4,10) finite logits
  - stage-transition pad shortcuts: stage2 block1 pads 96 channels (160−64), stage3 block1 pads 96 (256−160) — assert `pad_channels` values
  - 2-step train smoke (loss finite and decreasing-or-flat)

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Build `/tmp/exp044_composite.sh` from the exp042 D0-median pattern (threshold and tick constants below), launch in background
- [x] Confirm `run.log` is being written and first step prints appear (watchdog STARTUP guard covers the failure case)

### Milestone 3: Run resolved — full completion OR pre-registered gate/abort branch
- [x] If GATE_KILL (D0 median > 28ms): record measured dt, follow branch (iii) — verdict `invalid`, new kernel-pricing datum, NO relaunch this loop
- [ ] If CONTENTION_KILL/STARTUP_KILL: relaunch byte-identically once gates re-clear (max 2 relaunches, infra not research failure)
- [ ] If completed: rc=0, summary block present in run.log

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition checked, then necessary conditions in order; results recorded in exp-log-044.md

## Code Changes
- **train.py** (only file; one logical change):
  - Constants: `WIDTH_MULT = 4` → `STAGE_WIDTHS = (64, 160, 256)` (keep a comment noting baseline was (64,128,256) = 4x uniform).
  - `ResNet.__init__(self, num_blocks, num_classes=10, widths=(16, 32, 64))`: `w1, w2, w3 = widths` replaces the `16*width_mult, ...` derivation. Construction call becomes `ResNet(NUM_BLOCKS, NUM_CLASSES, STAGE_WIDTHS)`.
  - Print line: `ResNet-20 (64/160/256 asymmetric) | params: ...`.
  - Rationale: widening stage 2 by 25% is the only capacity increase available that keeps every layer ≤ 256 channels (EXP-040 cliff) and stays ≥ ~120 epochs — the first non-starved, non-cliff level test of capacity. `BasicBlock` already handles arbitrary in/out widths via the pad shortcut; no other code is touched, so deferral/numerics/noise currencies are unchanged.
  - Risks: 160 is 32-aligned but not 64-aligned — kernel mispricing is possible; handled by the dt gate, not by code.

## Configuration Changes
- Stage widths: (64, 128, 256) → (64, 160, 256) (+18% FLOPs, +507k params; dense-law projection dt ≈ 24.8ms → ~125 epochs; chosen per brainstorm-044 Idea 1 — cheapest non-cliff capacity placement, RegNet w1<w2<w3 ordering)
- All training constants unchanged (recipe certified optimal: EXP-007…036)

## Execution Environment
- Method: local, composite background script `/tmp/exp044_composite.sh` (exp042 D0-median pattern):
  1. **Dual launch gates, poll 30s × 240**: GPU 0 zero compute apps (`nvidia-smi --query-compute-apps=pid --id=0`) AND host 1-min load < 60 (`/proc/loadavg`) — both required (infra-errors EXP-032: load contamination with GPU free)
  2. `rm -f run.log`, then `cd <project root> && uv run train.py > run.log 2>&1 &` (no tee, per goal Procedure)
  3. **Inline watchdog, 44 ticks × 15s**, windowed dt from pct-print deltas (`WIN ms = Δpct×3000/Δstep`, ≥200-step windows — never the printed dt, never 50-step fine windows [6ms rung quantization, EXP-037]):
     - **GATE_KILL**: median of first 3 windows > **28ms** (off-rung; projection 24.8ms — a clean pass shows ~25; the >28 region means the dense law failed) → kill, record measured dt
     - **CONTENTION_KILL**: 4 consecutive windows > max(D0×1.25, 28ms) after the gate passes → kill (relaunch when gates re-clear)
     - **STARTUP_KILL**: no step prints by tick 12 (~180s) → kill
     - **NaN guard**: any `loss: nan` → kill
     - **Divergence**: eval acc < 15% at any eval after epoch 5 → kill
     - **WALL_CAP_KILL**: process still alive at tick 44 (~660s) → kill (run >10 min = failure)
  4. On exit: echo rc, `grep` summary block, eval-line tails
- Resources: GPU 0 only (H20), ~2.5GB VRAM expected (baseline 1.4GB + width); CIFAR-10 cached in `data/`
- Estimated runtime: ~500s total run (startup ~12s [inductor cache may miss on new shapes — up to ~25s], training 300s charged, ~125 evals ≈ 110–160s uncharged); GATE_KILL branch resolves in ~90s
- Log output: `<project root>/run.log` is the single source of truth; composite script prints watchdog lines to its own stdout (background Bash output)
- Tool skill: none (local)

## Abort Criteria
- GATE_KILL: first-3-window median dt > 28ms (pre-registered branch (iii): verdict `invalid`, metric NaN, idea closed on hardware grounds — do NOT relaunch or retune this loop)
- CONTENTION_KILL / STARTUP_KILL: infra, not research — relaunch byte-identically once both gates clear (max 2 relaunches, then Outcome failed)
- NaN loss or eval < 15% after epoch 5: kill, Outcome failed (research failure, no retry)
- Wall clock: any run reaching ~660s wall is killed and treated as failure (goal: ≤ 600s total)

## Verification Protocol

### Verification Procedure
First-failure-stop. Before condition 1, an **integrity pre-condition** (not a goal condition — guards false readings): profile must be pristine — windowed dt from ≥200-step windows mean ≤ ~26ms with no window > 31ms (off-rung), num_epochs within 112–135 (projection ~125; below ~108 implies contention even if dt windows looked clean — cross-check pct deltas), printed params == 4,793,290, training_seconds == 300.0, eval lines ≤ num_epochs. If integrity fails due to contention → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp):
   - Command (from project root): `grep "^best_test_acc:" run.log` — timeout 10s; empty output ⇒ crash ⇒ read `tail -n 50 run.log`
   - Pass: value ≥ 96.81. **Replicate branch**: if value ∈ [96.70, 96.80] with pristine profile, run the pre-registered replicate pair (two more byte-identical gated runs; claim improvement only if mean of the runs ≥ 96.81) before rendering a verdict. Fail: < 96.70 (or replicate mean < 96.81).
2. **Run completes within budget**: composite script rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 — timeout 10s
3. **Validation at most once per epoch**: `grep -c "eval ep" run.log` ≤ `grep "^num_epochs:" run.log` value — timeout 10s

Cleanup per goal Procedure: delete `run.log` after the experiment concludes (analyze phase).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log` (throughput check vs ~125 projection)
- num_params: `grep "^num_params:" run.log` (expect 4,793,290)
- final_test_loss: `grep "^final_test_loss:" run.log` (mechanism read: family ~0.185 ⇒ level-saturation branch; clearly below ⇒ capacity was binding)

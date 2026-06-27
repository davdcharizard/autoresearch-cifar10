# Plan EXP-040: Uniform 5× width (80/160/320) behind the early dt gate
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-040.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked (CPU)
- [x] On branch `autoresearch/exp-040` (cut from `autoresearch/dev`), edit `train.py`: `WIDTH_MULT = 4` → `WIDTH_MULT = 5` (one constant; stage widths become 80/160/320)
- [x] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python`): construct `ResNet(3, 10, 5)`, assert `sum(p.numel())` == 6,693,850 and a forward on a (2,3,32,32) tensor returns shape (2,10)
- [x] `git diff --stat` shows train.py only (1 insertion / 1 deletion)

### Milestone 2: Gated launch — dt gate decides regime, then full run or fallback
- [x] Build `/tmp/exp040_composite.sh` from the exp039 pattern with the EXP-040 dt-gate variant (details under Abort Criteria): launch gates (GPU-0 zero compute apps AND load < 60) → background `uv run train.py > run.log 2>&1` → watchdog 44×15s with (a) GATE_KILL if the median of the first 3 valid windows > 36ms (projected < ~86 epochs), (b) dt-adaptive contention kill (4 consecutive windows > D0 × 1.25), (c) NaN / divergence / startup / wall-cap guards as in exp039
- [ ] If GATE_KILL on 5×: record the measured dt, change the widths to the pre-registered fallback 4.5× — replace `w1, w2, w3 = 16 * width_mult, ...` arithmetic by passing explicit widths via `WIDTH_MULT` kept int: instead set `NUM_BLOCKS/WIDTH_MULT` untouched and hardcode `w1, w2, w3 = 72, 144, 288` in `ResNet.__init__` (params 5,423,122 — verify by CPU walk) — and relaunch once under the same gate
- [ ] A run survives the gate and completes: rc=0, `best_test_acc` present, num_epochs consistent with measured dt (300/dt/97 ± 5%)

### Milestone 3: Verification and exp-log complete
- [ ] First-failure-stop verification executed (protocol below), results in `logs/exp-log-040.md § Verification Results`
- [ ] Diagnostics recorded regardless of verdict: measured dt and epoch count (regime determination), ep5/10/20 evals, last-15 plateau mean/spread, final_test_loss, peak VRAM
- [ ] run.log deleted after metric extraction (analyze housekeeping)

## Code Changes
- **train.py** (only editable file): `WIDTH_MULT = 4` → `5`. Stage widths 64/128/256 → 80/160/320 (all multiples of 16 — tensor-core friendly under channels_last/bf16); params 4,286,026 → 6,693,850 (+56%); everything else (schedule, optimizer, augmentation, BN, compile path) byte-identical. This tests whether the converged width-level curve (+2.07 from 1×→4×, EXP-001) continues past 4× when convergence is GUARANTEED by the dt gate — the question all three prior width failures (EXP-002/005/007, starvation at ≤55 epochs) never answered. Risks: (a) dt lands compute-bound → gate kills at ~90s, fallback 4.5× tried once; (b) wider compile may lengthen startup (~budgeted, startup is uncharged); (c) eval cost grows with width (~1.3 → ~2s/epoch) — wall arithmetic below shows ≥60s headroom.

## Configuration Changes
- WIDTH_MULT: 4 -> 5 (uniform; the gate-protected interior of the width curve — 6× measured compute-bound at 58ms/55ep, 4× measured launch-bound at 22.4ms ≈ 9 blocks × 2.5ms width-independent launch cost, so 5× is the first width where the regime is genuinely undetermined)
- Fallback (only on GATE_KILL of 5×): explicit stage widths 72/144/288 (≈4.5×, params 5,423,122), one relaunch under the same gate. If BOTH gate-kill: no full run exists — pre-registered verdict `invalid` (metric NaN), with the two measured dts as the key learning (width >4× is compute-bound on this stack; axis stays closed on dt grounds).

## Execution Environment
- Method: local, via `/tmp/exp040_composite.sh` with `run_in_background: true` (script owns gating, launch, dt gate, watchdog, summary)
- Resources: GPU 0 ONLY (never GPU 1; wait if busy), ~2.5GB VRAM expected (5×), host load < 60 at launch
- Estimated runtime: dt 26–32ms scenario → ~100–115 epochs; wall ≈ 20s startup + 300s charged + ~110 × ~1.9s eval ≈ 530s (≥60s headroom under the 600s cap). GATE_KILL scenario: ~90–120s
- Log output: `uv run train.py > run.log 2>&1` (no tee); watchdog prints window-dt per tick; run.log deleted after extraction
- Tool skill: none (local)

## Abort Criteria
- **GATE_KILL (experiment-specific, the core screen)**: median of the first 3 valid watchdog windows (ticks ~3–5, ≥400-step windows — far above pct-quantization granularity) > 36ms ⇒ projected epochs < ~86 (300/0.036/97 ≈ 86), below the convergence margin (smallest measured converged run: 83 epochs, EXP-008; bigger models need more, not less). Kill, record dt, proceed to the 4.5× fallback (once). This implements — more strictly than — the Failed-Approaches re-entry rule "measured compiled dt must project ≥70 epochs".
- CONTENTION_KILL (dt-adaptive): D0 = median of first 3 valid windows; kill on 4 consecutive windows > D0 × 1.25 (at D0=28ms → >35ms). On kill: confirm contamination (nvidia-smi, loadavg), relaunch byte-identically when gates clear — contaminated runs are never analyzed.
- STARTUP_KILL: no step line by tick 12 (~180s; wider model compiles slower — 2 ticks more than exp039's 10).
- NaN guard: any `loss: nan` → kill, crash.
- DIVERGENCE_KILL: any eval < 15% after epoch 5.
- WALL_CAP_KILL: still running at tick 44 (~660s) → kill, >600s failure per goal constraint.

## Verification Protocol

### Verification Procedure
First-failure-stop; baseline from `exp-index.sh baseline` at verification time (currently 96.71; bar = 96.81 = baseline + 0.1pp).

**Pre-condition (run integrity — gates merit judgment):**
- Profile: from run.log step lines, compute 200-step windows (every 4th 50-step print pair, quantization-safe per the EXP-037 protocol note); require 0 windows > D0 × 1.25 (D0 from the watchdog's first 3 windows) and window mean within ±1ms of D0. Require num_epochs within ±5% of 300/(D0/1000)/97. If contaminated → rerun, do not judge.
- Integrity: `num_params: 6,693,850` (5×; or 5,423,122 if the fallback ran), `training_seconds: 300.0`, eval-line count == num_epochs.
- Timeout: greps on a finished run.log; missing summary lines ⇒ crashed run (`tail -n 50 run.log`).

**Condition 1 — best_test_acc ≥ 96.81**: `grep "^best_test_acc:" run.log`, numeric compare. Pass → Condition 2; fail → STOP, verdict `no-improvement` (remaining conditions incidental).

**Condition 2 — completes within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600.0.

**Condition 3 — validation ≤ once/epoch**: `grep -c "eval ep" run.log` ≤ num_epochs.

**Diagnostics (recorded regardless of verdict):**
- Measured dt + epoch count → which regime 5× landed in (launch- vs compute-bound) — this datum extends the per-block dt law whatever the verdict
- ep5/10/20 evals vs 4× family (~64/~75/~79): transit speed of the wider model under identical heat
- Last-15 plateau mean/spread + `final_test_loss` vs family (~96.5/±0.15, ~0.185): converged level and basin quality — THE hypothesis read
- peak_vram_mb (soft constraint awareness)

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~2400–2600 (5×)
- num_epochs: `grep "^num_epochs:" run.log` — regime-dependent, expect 95–120 if gate passed
- num_params: `grep "^num_params:" run.log` — 6,693,850 (5×) / 5,423,122 (fallback)

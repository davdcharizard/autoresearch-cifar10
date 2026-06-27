# Plan EXP-066: Kernel-size corner — 5x5 stem (launchable) + 5x5 stage-3 (probe-only) behind the internal-control GPU probe
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md

## Pre-run inequality arithmetic (pre-registered)

Baseline (exp-index header): 96.71 @ 1990397; family mean ≈ 96.57, σ ≈ 0.16; bar = 96.81.
Family signatures: dt 22.0–22.8ms, 137–140 epochs, steps [13,100, 13,600], params 4,286,026
(baseline) / **4,289,098 (stem-5x5: + (75−27)×64 = +3,072)**, VRAM ≈ 1,613MB.

- **Stage-3 5x5 (T)**: stage 3 = 6 convs ≈ 1/3 of net FLOPs; 5x5 = 25/9 ≈ 2.78× on those →
  net FLOPs ≈ 1.59×. Even on the FAST path the dense law (13.3ms/FLOPs-unit, EXP-034) predicts
  dt ≈ 30.3ms → ~101 epochs → starvation deficit ≈ −0.53 → required true gain ≥ ~0.8 vs a
  ~0 free-structural prior. **T is probe-only (cost cartography). It is NEVER launched,
  regardless of its probe reading.** Its purpose is the kernel-size pricing datum (fast path
  vs slow tier) at a second, well-separated FLOPs point.
- **Stem 5x5 (S)**: FLOPs delta ≈ +0.1% (stem is 3 input channels). Launch inequality:
  P_norm_S ≤ 22.9ms (toll ≤ +0.5ms ≈ −3 epochs ≈ −0.04pp, negligible against the band).
  If P_norm_S > 22.9 → **NO-LAUNCH**: the kernel-size corner closes on cost at zero charged
  seconds (verdict invalid/NaN, EXP-040/042/044/045/063/064 precedent).
- Probe normalization (EXP-064 internal-control pattern): time baseline net B in the same
  session; P_norm = 22.4 × P / B. B must read 21.5–23.5ms for the session to be valid
  (re-gate and retry otherwise).

## Milestones

### Milestone 1: Sanity check (CPU/GPU-light, no budget cost)
- [x] `/tmp/exp066_sanity.py` (run with `PYTHONPATH=. uv run python /tmp/exp066_sanity.py` from project root): build stem-5x5 and stage3-5x5 variants by subclassing/patching the train.py ResNet; assert (a) output shape (N,10), (b) param counts: S = 4,289,098, T = baseline + 6×(25−9)/9-scaled stage-3 conv params (compute exact in script and print), (c) forward+backward run, (d) 5 SGD steps on a 64-image batch at lr 0.01 reduce loss (last < first), (e) spatial dims preserved (padding=2 for 5x5). — T exact: 10,053,194.
- [x] All assertions pass (exit 0). SANITY_OK.

### Milestone 2: GPU probe with internal control (uncharged, ~4 min GPU)
- [x] Gates: GPU 0 zero compute apps AND host load < 40 (`awk '{print $1}' /proc/loadavg`), poll 30s until both pass. — passed first check (0 apps, load 7.53).
- [x] `/tmp/exp066_gpu_probe.py` (from exp064 template): for each of B (baseline), S (stem-5x5), T (stage3-5x5): `torch.compile` default mode, channels_last, bf16 autocast, 3-iter warmup (fwd+bwd, no step), then time 40 steps of the full charged step (H2D-equivalent tensors pre-staged on GPU is fine — comparative timing only; identical procedure for all three nets), report median-of-windows dt.
- [x] Record B, P_S, P_T, P_norm_S = 22.4×P_S/B, P_norm_T = 22.4×P_T/B in the exp-log. — B=22.18, S=22.44, T=30.72; P_norm_S=22.66, P_norm_T=31.02.
- [x] Session validity: B ∈ [21.5, 23.5]; else re-gate (load may have risen) and rerun probe once; if still invalid, wait and retry. — VALID (22.18).
- [x] **Branch (pre-registered)**: P_norm_S ≤ 22.9 → proceed to Milestone 3. P_norm_S > 22.9 → NO-LAUNCH: skip Milestones 3–4, record both pricing data, verdict invalid/NaN, proceed to verification-of-closure and analysis. — **22.66 ≤ 22.9: LAUNCH branch fired.** (T datum: 31.02 ≈ dense-law 30.3 — 5x5 is fast-path, FLOPs-priced.)

### Milestone 3 (conditional on M2 launch branch): Implement and run the stem-5x5 full experiment
- [x] `train.py`: single line — `ResNet.__init__`: `self.conv1 = nn.Conv2d(3, w1, 5, stride=1, padding=2, bias=False)` (was kernel 3, padding 1). No other changes of any kind. — diff confirmed 1 line.
- [x] Composite launcher `/tmp/exp066_composite.sh` from `/tmp/exp061_composite.sh` via sed (header rename only): dual launch gates (GPU 0 apps==0 AND load<60, poll 30s×240) → `rm -f run.log` → background `uv run train.py > run.log 2>&1` → watchdog 44×15s ticks computing win=(Δpct×3000/Δstep); GATE_KILL D0>26ms; CONTENTION_KILL 4 consecutive >max(26, D0×1.25); STARTUP_KILL no step prints by tick 12; NaN guard; divergence guard (eval acc >10 then <20 → kill); WALL_CAP 600s.
- [x] Launch via Bash run_in_background + until-grep watcher + TaskOutput(block=true). — background task bypsic286; completion notification pattern.
- [x] Run completes with pristine telemetry (no kills) and `best_test_acc:` present in run.log. — rc=0, zero watchdog events.

### Milestone 4 (conditional): Verification and decision
- [x] Integrity pre-condition (gates Condition 1): num_steps ∈ [13,100, 13,600]; num_params = 4,289,098; dt windows family-band; epochs 134–141; no contamination signature. — PASS (13,266 steps; exact params; 22.0–23.3ms; 137 ep).
- [x] Extract metrics, compare to bar 96.81 per the goal Procedure. — best 96.14 < 96.81.
- [x] **Pre-registered decision branches**: (i) best ≥ 96.81 → run ONE byte-identical replicate (EXP-052 protocol — low-prior candidate near the bar); decision on the PAIR MEAN ≥ 96.81. (ii) best ∈ [96.41, 96.73] family band → no-improvement (kernel-size corner closed as absorbed null). (iii) best < 96.41 → no-improvement, structural-negative reading. (iv) integrity fail → rerun byte-identically once (contamination), never analyze a contaminated run. — **Branch (iii) fired: 96.14 = mean − 2.7σ, real structural negative.**
- [x] Delete run.log after metric extraction.

## Code Changes
- **train.py** (ONLY if M2 launch branch): `ResNet.__init__` stem conv kernel 3→5, padding 1→2. Tests whether enlarged input receptive field (the one structural change with no direct null in the record) moves the plateau; +3,072 params, FLOPs +~0.1%. Risk: 5x5 kernel may fall off the cudnn/inductor fast path even at 64 output channels — exactly what the probe prices before any budget is spent.
- **/tmp scripts** (not in repo, not committed): exp066_sanity.py, exp066_gpu_probe.py, exp066_composite.sh.

## Configuration Changes
- None. All hyperparameters, transforms, schedule, optimizer, and loop are byte-identical to baseline. (Kaiming init covers the new fan-in automatically.)

## Execution Environment
- Method: local, GPU 0 only. Probe and run both behind dual gates (GPU 0 zero compute apps; host load < 40 for probe, < 60 for run — infra-errors EXP-032/059).
- Resources: 1× H20 (GPU 0), ~1.7GB VRAM.
- Estimated runtime: sanity ~1 min; probe ~4 min GPU; full run (if launched) ~8 min wall (300s charged + startup + evals); worst case incl. replicate ~20 min.
- Log output: full run → `run.log` in project root (per goal Procedure, no tee); probe → stdout captured by Bash; composite telemetry → `/tmp/exp066_composite_run1.log`.
- Tool skill: none (local).

## Abort Criteria
- Probe session: B outside [21.5, 23.5] after one re-gated retry → treat as host contamination, wait for load < 40 and retry (do not interpret an invalid session).
- Full run (watchdog-automated): GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive windows > max(26, D0×1.25); STARTUP_KILL no step prints by tick 12 (~180s); NaN in loss; divergence (eval acc reached >10 then falls <20); WALL_CAP 600s total.
- Manual: any run.log traceback → kill, classify per execute-skill failure rules.

## Verification Protocol

### Verification Procedure
Follows goals/maximize-cifar10-test-accuracy.md § Procedure exactly.

1. Confirm GPU 0 free (`nvidia-smi`) — handled by composite gates.
2. Run: `uv run train.py > run.log 2>&1` (inside composite; no tee). Timeout: WALL_CAP 600s enforced by watchdog.
3. Extract: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`; empty → crashed → `tail -n 50 run.log`.
4. Baseline: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline ".autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv"` → 96.71; bar = 96.81.
5. Necessary conditions: (a) best_test_acc ≥ 96.81 — subject to the integrity pre-condition and the pre-registered replicate-pair-mean protocol for branch (i); (b) total_seconds ≤ 600 (`grep "^total_seconds:" run.log`); (c) validation at most once per epoch — structurally guaranteed (one `evaluator.evaluate` call per epoch loop iteration; confirm no extra eval calls in the diff).
6. NO-LAUNCH branch: conditions are vacuous — verdict is invalid/NaN with zero charged seconds; the "result" is the pair of pricing data (P_norm_S, P_norm_T), recorded in the exp-log § Verification Results as the pre-registered closure outcome.
7. `rm -f run.log` after extraction (and any replicate's log).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_params: `grep "^num_params:" run.log` (must read 4,289,098 for S)
- Probe pricing data (always collected, both branches): B, P_S, P_T, P_norm_S, P_norm_T — the kernel-size corner's cost map.

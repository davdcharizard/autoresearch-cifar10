# Plan EXP-039: BN running-stat momentum 0.1 → 0.25 (freshness side of the dose-response)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-039.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked (CPU)
- [x] On branch `autoresearch/exp-039` (cut from `autoresearch/dev`), edit `train.py`: add `BN_MOMENTUM = 0.25` to the constants block and pass `momentum=BN_MOMENTUM` at all three `nn.BatchNorm2d` construction sites (BasicBlock bn1/bn2, ResNet stem bn1)
- [x] CPU sanity (system python3 lacks torchvision — use `CUDA_VISIBLE_DEVICES="" uv run python`): construct `ResNet(3, 10, 4)`, assert all 19 BatchNorm2d modules report `momentum == 0.25` and `sum(p.numel())` == 4,286,026
- [x] `git diff --stat` shows train.py only (4 insertions / 3 deletions, mirroring EXP-038's diff shape)

### Milestone 2: Gated launch, clean run to completion
- [x] Adapt the composite launcher (baseline-threshold variant, `/tmp/exp036_composite.sh` pattern) to `/tmp/exp039_composite.sh`: dual launch gates (GPU 0 zero compute apps AND 1-min load < 60, poll 30s × 240) → `rm -f run.log` → background `uv run train.py > run.log 2>&1` → 44×15s tick watchdog (window ms = Δpct×3000/Δstep; CONTENTION_KILL on 4 consecutive windows > 27ms; STARTUP_KILL if no step line by tick 10; NaN guard; DIVERGENCE_KILL if eval < 15% after ep5; WALL_CAP_KILL at tick 44) → wait → rc + summary greps + eval tails
- [x] Run completes: rc=0, `best_test_acc` line present, num_epochs 139, dt signature 22.31ms

### Milestone 3: Verification and exp-log complete
- [x] First-failure-stop verification executed (protocol below), results recorded in `logs/exp-log-039.md § Verification Results` — Condition 1 FAILED on merits (96.64 < 96.81)
- [x] Diagnostic suite recorded regardless of verdict: ep5/10/20 evals (hot-phase lag/variance signature), last-15 plateau mean and spread, final_test_loss
- [ ] run.log deleted after metrics are extracted into the exp-log (done during analyze housekeeping)

## Code Changes
- **train.py** (only editable file): add `BN_MOMENTUM = 0.25` to the hyperparameter constants block (after `LABEL_SMOOTHING`), and change the three BatchNorm2d constructions to `nn.BatchNorm2d(..., momentum=BN_MOMENTUM)` — `BasicBlock.bn1` (line ~52), `BasicBlock.bn2` (line ~56), `ResNet.bn1` (line ~78). This shortens the running-stat EMA horizon from ~10 batches to ~4, testing whether the lag mechanism EXP-038 measured from below (m=0.02 → −0.30 via stale constants) still costs accuracy at the default m=0.1. Training path is byte-identical — weights, gradients, schedule, batch noise, kernel numerics all untouched; only the stat-buffer EMA coefficient changes. Risks: (a) higher estimator variance at a 4-batch horizon — arithmetically small at batch 512 (≈2,048 samples per channel estimate) and directly measured by plateau scatter; (b) none in any closed currency (zero dt, zero params, zero heat, zero numerics change). Momentum is set at construction only — no runtime mutation, no compile-guard risk (EXP-038 confirmed identical signatures).

## Configuration Changes
- BN running-stat momentum: 0.1 (PyTorch default, implicit) -> 0.25 (explicit `BN_MOMENTUM`) — completes the dose-response bracket around the default: {0.02: −0.30 (EXP-038), 0.1: incumbent, 0.25: this run}. Chosen point gives ~2.5× less lag while keeping per-estimate sample count high (~4 batches × 512); 0.2 would be too timid to separate from noise if the effect is small, 0.5+ enters the genuinely-noisy single-batch-dominated regime.

## Execution Environment
- Method: local, via composite gated launcher `/tmp/exp039_composite.sh` run with `run_in_background: true`; the script owns gating, launch, watchdog, and post-run summary
- Resources: GPU 0 ONLY (hard constraint — never GPU 1; wait for GPU 0 if busy), ~1.6GB VRAM, ~10 of 180 cores; host 1-min load must be < 60 at launch (infra-errors EXP-032)
- Estimated runtime: ~480–500s total (300s charged + ~9–18s startup depending on compile-cache state + ~139 × ~1.3s evals + stalls); hard failure above 600s
- Log output: `uv run train.py > run.log 2>&1` (no tee, per goal procedure); watchdog ticks print window-dt and last eval line to the composite script's stdout; run.log deleted after metric extraction
- Tool skill: none (local execution)

## Abort Criteria
- STARTUP_KILL: no `step` line in run.log by tick 10 (~150s) — compile or loader hang
- CONTENTION_KILL: 4 consecutive 15s-tick windows with window-avg dt > 27ms (baseline true dt 22.3–22.4ms quantizes to 18.0/24.0 rungs; 27 sits OFF the rungs per the EXP-037 protocol note — only genuine ≥30ms windows trip it). On kill: confirm contamination via nvidia-smi/loadavg, then relaunch byte-identically once gates clear (infra-errors EXP-011/EXP-032; contaminated runs are never analyzed)
- NaN guard: any `loss: nan` in run.log — kill immediately, verdict crash (not expected: training path identical to baseline)
- DIVERGENCE_KILL: any eval < 15% after epoch 5 — fresher constants should not depress evals (EXP-038's depression came from lag; m=0.25 reduces lag). Note for monitoring, not abort: if ep5 eval lands well BELOW family (~64), that is the variance-dominance signature — record it, let the run finish
- WALL_CAP_KILL: still running at tick 44 (~660s) — kill, treat as >600s failure per goal constraint

## Verification Protocol

### Verification Procedure
First-failure-stop, conditions in order; evaluate after the composite script reports completion. Baseline at verification time: `exp-index.sh baseline` on `experiment-indices/maximize-cifar10-test-accuracy.tsv` → currently 96.71 (bar = 96.81 = baseline + 0.1pp).

**Pre-condition (run integrity — gates whether Condition 1 is judged on merits):**
- Profile: parse all step lines from run.log via the coarse window method (pct deltas across watchdog-scale ≥200-step windows, avoiding the 6ms quantization rungs of 50-step windows — EXP-037 protocol note); require mean window dt ≈ 22.0–23.0ms and 0 windows > 27ms. Also require num_epochs in 135–143. If contaminated → rerun, do not judge.
- Integrity sub-checks: `num_params: 4,286,026` (capacity unchanged), `training_seconds: 300.0` (timer semantics untouched), eval-line count == num_epochs (one eval per epoch exactly).
- Timeout: verification greps run on a finished run.log; treat any missing summary line as a crashed run (`tail -n 50 run.log` for traceback).

**Condition 1 — best_test_acc ≥ 96.81** (baseline + 0.1pp):
- Command (project root): `grep "^best_test_acc:" run.log` — numeric compare against 96.81.
- Pass → continue to Condition 2. Fail → STOP, verdict `no-improvement` (conditions 2–3 noted as incidental only).

**Condition 2 — run completes within budget**:
- Composite script's recorded rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600.0.

**Condition 3 — validation at most once per epoch**:
- `grep -c "eval ep" run.log` ≤ `grep "^num_epochs:" run.log` value.

**Diagnostics (recorded regardless of verdict, for the dose-response read):**
- Hot phase: evals at ep 5/10/20 vs family (~64/~75/~79) — lag reduction should show as at-or-above family; below-family = variance cost
- Plateau: mean and spread of last-15 evals vs family (~96.5, ±0.15); spread is the direct variance-cost diagnostic
- `grep "^final_test_loss:" run.log` vs family ~0.185

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect 1613.0 (unchanged)
- num_epochs: `grep "^num_epochs:" run.log` — expect 139 ± 4
- num_params: `grep "^num_params:" run.log` — expect 4,286,026 (unchanged)

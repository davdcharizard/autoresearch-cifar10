# Plan EXP-067: σ-tightening baseline replicate pair (zero-diff, n=3 → n=5, drift re-anchor)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md

## Pre-registered protocol (EXP-027/EXP-052 lineage)

- **Zero code diff**: train.py is NOT modified. Verify `git diff --quiet -- train.py` before
  each launch.
- **Metric recorded** = mean(R1, R2). **Verdict pre-registered: no-improvement** regardless of
  individual draws — a zero-diff replicate cannot constitute an improvement even if one draw
  exceeds the bar (EXP-027 precedent; harvesting a lucky draw would be variance mining /
  seed-class reward hacking).
- Prior baseline-config draws (n=3): {96.71, 96.59, 96.40} → mean 96.567, σ ≈ 0.16.
- **Decision bands (pre-registered)**:
  - Each draw expected in mean ± 2σ = [96.25, 96.89] with family signatures (dt 22.0–22.8ms,
    epochs 134–141, steps [13,100, 13,600], params 4,286,026, VRAM ≈ 1,613MB).
  - Both in-band → pool to n=5; report pooled mean and σ̂ (expected σ̂ ∈ [0.10, 0.22]); bands,
    bar arithmetic (mean + 1.5σ), and effect-size screen re-certified (update goal-learnings σ
    entry with n=5 numbers).
  - Any draw out-of-band WITH clean integrity → drift/σ-underestimate DETECTION: record in
    goal-learnings (Protocol Findings revision), verdict still no-improvement; flag that all
    standing bands need recalibration next loop.
  - Any run failing integrity (contamination signature: steps below family band, slow windows,
    foreign load) → rerun that replicate byte-identically ONCE; never pool a contaminated read
    (infra-errors EXP-011/032/058).

## Milestones

### Milestone 1: Pre-launch checks
- [ ] On branch autoresearch/exp-067; `git diff --quiet -- train.py` passes (zero diff vs autoresearch/dev HEAD).
- [ ] Composite launcher `/tmp/exp067_composite.sh` created from `/tmp/exp061_composite.sh` via sed (header rename only); same gates/watchdog as EXP-066 (GATE_KILL D0>26ms, CONTENTION_KILL 4×>max(26, D0×1.25), STARTUP_KILL tick 12, NaN/divergence guards, WALL_CAP).

### Milestone 2: Replicate Run 1
- [ ] Launch via composite (gates: GPU 0 apps==0 AND load<60) in background; output → run.log, telemetry → /tmp/exp067_composite_run1.log.
- [ ] Pristine completion (no watchdog kills, rc=0); extract best_test_acc, num_steps, num_epochs, num_params, total_seconds; integrity check against family bands.
- [ ] `mv run.log` metrics extracted, then `rm -f run.log` before Run 2.

### Milestone 3: Replicate Run 2
- [ ] Same composite, fresh invocation; telemetry → /tmp/exp067_composite_run2.log.
- [ ] Pristine completion + integrity check as Run 1; delete run.log after extraction.

### Milestone 4: Pooling and verification
- [ ] Compute mean(R1, R2) (recorded metric) and pooled n=5 sample σ̂ over {96.71, 96.59, 96.40, R1, R2}.
- [ ] Evaluate pre-registered bands (in-band re-certification vs out-of-band detection branch).
- [ ] Record everything in exp-log § Verification Results.

## Code Changes
- **None.** train.py byte-identical to autoresearch/dev HEAD (1990397 recipe). The "experiment"
  is two more draws from the baseline run distribution.
- **/tmp scripts** (not in repo): exp067_composite.sh only.

## Configuration Changes
- None.

## Execution Environment
- Method: local, GPU 0 only, two sequential gated composite runs (the second waits behind the
  same gates; no concurrency).
- Resources: 1× H20 (GPU 0), ~1.7GB VRAM.
- Estimated runtime: 2 × ~8 min wall + gate waits; worst case with one contamination rerun ~30 min.
- Log output: each run → `run.log` (no tee, per goal Procedure), deleted after extraction;
  composite telemetry persists in /tmp/exp067_composite_run{1,2}.log.
- Tool skill: none (local).

## Abort Criteria
- Watchdog-automated per run: GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive windows >
  max(26, D0×1.25); STARTUP_KILL no step prints by tick 12; NaN loss; divergence (eval >10 then
  <20); WALL_CAP 600s.
- Manual: traceback in run.log → classify per execute-skill rules. A second contamination on
  the SAME replicate after its one allowed rerun → record partial result (pool what is clean;
  if only one clean draw exists, record metric = that draw's pair-with-prior-mean is NOT used —
  instead record mean of clean new draws collected and note reduced pooling in the log).

## Verification Protocol

### Verification Procedure
Follows goals/maximize-cifar10-test-accuracy.md § Procedure for each run (GPU 0 free check via
composite gates; `uv run train.py > run.log 2>&1`; extract via
`grep "^best_test_acc:\|^peak_vram_mb:" run.log`; delete run.log after extraction; 600s cap
watchdog-enforced).

1. **Condition 1 — best_test_acc ≥ bar 96.81** (baseline 96.71 via exp-index + 0.1): evaluated
   on the RECORDED metric = mean(R1, R2). Pre-registered FAIL is the expected outcome (the
   pair mean of a zero-diff config estimates 96.57; even a lucky pair cannot pass the
   intervention-class requirement — there is no intervention). Verdict: no-improvement.
2. **Condition 2 — each run completes ≤ 600s total**: `grep "^total_seconds:" run.log` per run.
3. **Condition 3 — validation once per epoch**: structurally guaranteed (zero diff).
4. Integrity per run (gates pooling, not the verdict): steps ∈ [13,100, 13,600], params
   4,286,026, pristine watchdog windows, epochs 134–141.

### Informational Metrics (Optional)
- R1, R2 individual best_test_acc; pooled n=5 mean and σ̂ (the experiment's actual product)
- peak_vram_mb, num_epochs, num_steps per run (ledger re-anchor)
- Drift verdict: in-band re-certification vs out-of-band detection

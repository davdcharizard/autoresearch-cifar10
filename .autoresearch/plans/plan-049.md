# Plan EXP-049: PEAK_LR 0.4 → 0.3 — the heat-down bracket (single-constant probe)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16. Heat bracket today: peak 0.6 → −0.57 (EXP-010); below 0.4 unmeasured (exp-report-010 § Unexplored Avenues, queued and never run).

Projection: signatures byte-identical to baseline — params **4,286,026**, dt 22.3–22.7ms, ~139 epochs, ~13,4xx steps (EXP-010 demonstrated pure-LR isolation: identical throughput at peak 0.6). The ONLY changed quantity is the LR at every progress point (×0.75).

## Milestones

### Milestone 1: Code change implemented and passing static checks
- [x] `train.py` L23: `PEAK_LR = 0.4  # linear scaling: 0.1 x (512/128)` → `PEAK_LR = 0.3  # heat-down bracket (EXP-049): 0.75x the EXP-000 linear-scaled peak`
- [x] Verify diff is exactly one line: `git diff --stat` shows 1 file, ±1 line ✓ (1 file, +1/−1)
- [x] Static check: AST parses; extracted constants PEAK_LR=0.3/WARMUP_FRAC=0.15; lr_at(0.15)=0.3, lr_at(0.075)=0.15, lr_at(0.575)=0.15, lr_at(1.0)≈0 — all pass

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Reuse `/tmp/exp046_composite.sh` AS-IS — verified on disk (4023 bytes, executable), used verbatim
- [x] GATE_DECISION D0=22.2ms, projected_epochs=139, contention_thresh=27.8ms — GATES_CLEAR poll 1 (apps=0, load=5), pid 1664016 launched 23:15:20

### Milestone 3: Run resolved — full completion OR pre-registered branch
- [x] Early trajectory: ep1 35.70; mid-schedule slightly below family as expected (ep7 63.06); converged through the anneal — no abort signals
- [x] No kills — gates clear poll 1, slow_streak 0 throughout
- [x] Completed: RC=0, full summary block present (best 96.52, 139 ep, 13,456 steps, 487.7s total)

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition PASS; Condition 1 FAIL (96.52 < 96.81, below replicate band) — first-failure-stop; results in exp-log-049.md

## Code Changes
- **train.py** (only file, one constant): `PEAK_LR = 0.4` → `0.3`. Why this tests the hypothesis: time-keyed one-cycle scales LR multiplicatively at every progress point, so this is a pure 0.75× integrated-heat probe at unchanged noise sources (batch/momentum), schedule shape, arithmetic, and signatures — the exact mirror of EXP-010's 1.5× up-probe, sampling the unmeasured shallow side of the optimum. Risks: none structural; the known failure shape is a converged shallow-side deficit (clean no-improvement).

## Configuration Changes
- `PEAK_LR`: 0.4 → 0.3 (×0.75). Rationale: brackets the heat axis from below with a step comparable in spirit to EXP-010's ×1.5 up-step but conservative (down-side is the shallow side per one-cycle asymmetry); 0.4 was linearly scaled for an unaugmented 1x net in EXP-000 and never probed downward after 4x widening + heavy aug shifted the landscape. Warmup (0.15), momentum (0.9), WD, batch, LS unchanged.

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` reused verbatim (validated 3× this session): dual launch gates (GPU-0 zero compute apps AND load < 60, poll 30s×240) → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → watchdog 44×15s (GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN; divergence eval < 15% after ep5; WALL_CAP tick 44)
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~470–505s clean (startup ~12–25s, 300s charged, ~139 uncharged evals)
- Log output: `<project root>/run.log` + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- NaN loss or eval < 15% after epoch 5 → research failure, no retry (not expected at LOWER LR)
- GATE_KILL / CONTENTION_KILL / STARTUP_KILL → infra by definition for a pure-LR change: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27 (off-rung); num_epochs within 136–142 (family ~139; pure-LR isolation per EXP-010); num_steps within ~1% of the 13,428–13,515 family ledger (EXP-048 protocol finding); printed params == 4,286,026; training_seconds == 300.0; eval lines ≤ num_epochs. Numerics judged by trajectory + plateau + test_loss (single-eval bands unreliable — EXP-010/048). Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → pre-registered replicate pair (two byte-identical gated runs; improvement only if mean ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185; above ⇒ undertrained shallow-side signature, below ⇒ smoother convergence)
- num_epochs / num_steps: ledger cross-check (expect family values exactly)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- Plateau shape (last-8 evals): converged-flat expected; still-climbing-at-cutoff would mirror EXP-010's inverse (heat too low to finish) — branch (iii) mechanism evidence

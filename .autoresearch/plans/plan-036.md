# Plan EXP-036: LABEL_SMOOTHING 0.1 → 0.2 — the last unmeasured recipe constant, at the in-domain anchor value
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md

## Milestones

### Milestone 1: One-constant change implemented and sanity-checked
- [x] On branch `autoresearch/exp-036` (cut from `autoresearch/dev` @ 1990397), edit `train.py`: `LABEL_SMOOTHING = 0.1` → `LABEL_SMOOTHING = 0.2  # in-domain anchor dose (cifar10-fast, airbench); 0.1 was never dosed (EXP-000 bundle)`.
- [x] Sanity: AST parse OK; `git diff` shows exactly 1 changed line; the constant feeds all three `F.cross_entropy(..., label_smoothing=LABEL_SMOOTHING)` sites (timed step, warmup, and nothing else — eval uses `evaluator.evaluate`, untouched).

### Milestone 2: Run 1 launched with gates and baseline watchdog
- [x] Launch gates: zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h. (GATES_CLEAR poll 1: apps=0 load=4; LAUNCHED pid=1520172 18:46:23)
- [x] Composite watchdog (15s ticks, 44): contention 4 consecutive windows >27ms (baseline thresholds — signatures must be byte-identical); STARTUP_KILL tick 10; NaN/inf; divergence eval <15% after ep5; wall cap 600s.
- [ ] Expected signatures: dt ≈ 22.4ms windows, ~139 epochs, total ~475–495s. **Monitoring note**: the printed train-loss trace will sit ~0.2–0.3 ABOVE the baseline family at equal progress — LS=0.2 raises the CE floor (entropy of the smoothed target). This is arithmetic, not divergence; the NaN/divergence guards are unaffected.

### Milestone 3: Completion and readout
- [x] Full run: rc=0, total 479.9s ≤600s, epochs 139, eval_lines 139 = num_epochs, params 4,286,026. (best 96.58 — below bar)
- [x] No fallback runs: single clean run decides; 0/0.05 doses are NEW experiments contingent on this read's sign (per brainstorm bracketing logic).

## Code Changes
- **train.py** (only file): exactly one constant. Why this tests the hypothesis: the ONLY change is the loss-target smoothing dose — margin/logit-norm pressure on a boundary-limited model (EXP-032) — at byte-identical execution signatures, so any plateau shift is a pure LEVEL read of the last unmeasured constant, aligned with both in-domain anchor recipes (LS 0.2).
- Risks/edge cases: (a) loss-floor shift (cosmetic, noted above); (b) over-smoothing compressing logit gaps → a negative read, which is the falsification arm; (c) none in any closed currency — dt/heat/noise/VRAM/epochs all unchanged.

## Configuration Changes
- LABEL_SMOOTHING: 0.1 → 0.2. Anchors: cifar10-fast (Page) and CIFAR-10 airbench (Jordan) both use 0.2 on this dataset with wide nets and budget-matched one-cycle schedules (knowledge README References); Müller et al. NeurIPS 2019 establishes task-dependent dose-response. NOT tuned beyond the anchor value; 0.05/0 reserved as bracketing follow-ups keyed to this read's sign.
- All other constants unchanged.

## Execution Environment
- Method: local composite background Bash (dual launch gates → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` → 44×15s watchdog → wait → rc + summary greps + eval tails), branch `autoresearch/exp-036`, GPU 0 only.
- Resources: VRAM ~1613MB (baseline-identical); 8 loader workers.
- Estimated runtime: ~475–495s total (baseline signature). ~105s margin under the cap.
- Log output: `run.log` (no tee); post-hoc awk profile authoritative; delete run.log after analysis.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Contention**: 4 consecutive windows >27ms → kill, contaminated, rerun once after both gates re-clear.
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).
- NOTE: an elevated train-loss TRACE is expected (LS floor) and is NOT an abort signal; eval accuracy is the health metric.

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = **96.81**. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>27)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>27: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤24ms AND num_epochs within ±4 of 139. Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check: `grep "^num_params:" run.log` = 4,286,026; training_seconds = 300.0; eval_lines = num_epochs; on a bar-pass confirm the plateau (last ~15 evals) sits above the baseline band, not a single-eval spike.
   - Timeout: greps instant; missing run.log = infrastructure failure.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: eval_lines ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **Plateau level & final_test_loss** vs baseline family (~96.6 / ~0.185): NOTE — test_loss is measured by Eval() against HARD labels, but the model's logit distribution under LS 0.2 is more compressed; a higher test_loss with equal/higher accuracy is expected and NOT a quality signal here. Accuracy is the only honest cross-dose comparator.
- **dt / epochs / startup / VRAM**: must be baseline-identical (22.4ms / 139±4 / ~10s / 1613MB) — any deviation flags an implementation error, not a finding.
- **Plateau onset epoch** (first eval ≥ 96.0) vs baseline family (~ep 120) — margin pressure could shift convergence shape.

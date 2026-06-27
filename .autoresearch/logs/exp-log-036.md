# Experiment Log: EXP-036 — LABEL_SMOOTHING 0.1 → 0.2 (last unmeasured recipe constant, in-domain anchor dose)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-036.md
- **Plan**: plans/plan-036.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-036 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-constant change per plan Milestone 1: on branch `autoresearch/exp-036`, edited `train.py` line 27 `LABEL_SMOOTHING = 0.1` → `0.2` (with comment citing the in-domain anchors). Sanity passed: AST parse OK, `git diff` shows exactly 1 changed line. The constant feeds exactly two `F.cross_entropy(..., label_smoothing=LABEL_SMOOTHING)` call sites in the file (timed training step L230, compile-warmup L192) — both training-side; eval goes through `evaluator.evaluate(base_model, device)` and is untouched. Execution signatures (dt, epochs, VRAM, params) are expected byte-identical to baseline; this is a pure plateau-LEVEL probe.

### Surprises & Discoveries
- None at implementation time — the edit is the smallest intervention of any experiment in the series. Note for verification: the plan text says "all three" CE sites but the file contains two (timed step + warmup); the third CE mention in earlier plans referred to eval, which in the current baseline uses `evaluator.evaluate` and never touches LABEL_SMOOTHING. No functional discrepancy.

### Decisions
- Comment kept on the constant line so the diff is self-documenting at commit time (anchors + audit-gap rationale), matching the exact string pre-registered in the plan.

## Run Log

### Run 1
- **Description**: Full composite gated run of the LS=0.2 recipe on GPU 0: dual launch gates (zero GPU-0 compute apps AND 1-min load <60, 30s polls up to 2h) → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background → 44×15s tick watchdog (contention 4 consecutive windows >27ms — baseline thresholds since signatures must be baseline-identical; STARTUP_KILL at tick 10 if no step lines; NaN/inf guard; divergence guard eval <15% after epoch 5; wall cap 600s) → wait → rc + summary greps + eval tails. Expected: dt ≈22.4ms, ~139 epochs, total ~475–495s, best_test_acc read against bar 96.81. The train-loss trace is expected ~0.2–0.3 ABOVE the baseline family (LS=0.2 raises the CE floor) — arithmetic, not divergence.
- **Job ID / PID**: background task be1t9wycs (composite script /tmp/exp036_composite.sh; train PID printed in its output)
- **Log file**: run.log (project root; deleted after analysis per goal constraints)
- **WandB**: n/a
- **Status**: completed (rc=0, watchdog never triggered)
- **Started**: 2026-06-10 18:46:23 (GATES_CLEAR on poll 1: apps=0, load=4)
- **Ended**: 2026-06-10 ~18:54:23 (PROC_EXITED at watchdog tick 33; total_seconds 479.9)
- **Observations**: Signatures byte-identical to baseline exactly as predicted: all 31 watchdog windows 21.7–22.7ms (slow_streak never >0), startup 9.2s, VRAM 1613.0MB, 139 epochs / 13,439 steps. Train run was clean end-to-end — no contention, no NaN, no divergence. test_loss plateau ~0.284–0.286 vs baseline family ~0.185: expected arithmetic (Eval uses hard labels; LS=0.2 compresses logits → higher CE at equal accuracy), pre-flagged in plan as NOT a quality signal. Plateau onset (first eval ≥96.0) at ep 127 vs family ~120 — slightly later, same shape. Last-8 evals 96.45–96.58, tight plateau centered ≈96.52.
- **Key Metrics**: best_test_acc 96.58 | final 96.58 | final_test_loss 0.2860 | 139 epochs | dt mean 22.3ms (267 windows, 0 slow>27) | params 4,286,026 | training_seconds 300.0 | total 479.9s

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
First-failure-stop protocol per plan-036 (bar = baseline 96.71 + 0.1 = 96.81).

1. **best_test_acc ≥ 96.81** — **FAIL**. `grep "^best_test_acc:" run.log` → **96.58%**. Pre-condition (clean-run profile) PASSED first: awk over 267 step-line windows → mean 22.3ms, 0 windows >27ms (require ≤2), num_epochs 139 (within 139±4) — run is uncontaminated, the read is honest. Integrity sub-checks all pass: num_params 4,286,026 ✓; training_seconds 300.0 ✓; eval_lines 139 = num_epochs 139 ✓. 96.58 < 96.81 → condition fails on its merits.
2. **Completes within budget** — skipped (aborted after prior failure). [Incidental: rc=0, total_seconds 479.9 ≤ 600.]
3. **Validation ≤ once/epoch** — skipped (aborted after prior failure). [Incidental: 139 = 139.]

**Verdict basis**: valid clean run, necessary condition 1 failed → no-improvement. Result 96.58 sits essentially AT the baseline-recipe mean (≈96.57, σ≈0.16 per EXP-027): LS=0.2 produced zero measurable level shift vs LS=0.1.

### Informational Metrics
- Plateau level last-8 evals 96.45–96.58 (center ≈96.52) vs baseline family ~96.6 — within noise, no shift.
- final_test_loss 0.2860 vs family ~0.185 — expected hard-label CE arithmetic under compressed logits (pre-flagged in plan; not a quality signal).
- dt 22.3ms / 139 epochs / startup 9.2s / VRAM 1613.0MB — baseline-identical, confirming implementation purity.
- Plateau onset (first eval ≥96.0): ep 127 vs family ~120 — marginally later, same converged shape.

## Human Notes
(autopilot — none)

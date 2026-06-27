# Plan EXP-050: Additive logit margin on the true class (MARGIN = 0.75, training-loss-only)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16. First probe of the loss-geometry class (never measured in 50 experiments); target mechanism: the decision-boundary-limited ceiling (EXP-011/032 insight).

Projection: signatures byte-identical to baseline — params **4,286,026**, dt 22.2–22.7ms (the added op is a fused elementwise subtract on a 512×10 tensor, < 0.01ms), ~139 epochs, ~13.4–13.5k steps. The printed TRAIN loss will read systematically HIGHER than family (the margin inflates the loss value by construction) — this is cosmetic and expected; numerics are judged on TEST evals only.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py`: MARGIN = 0.75 constant added; timed-loop and warmup losses subtract the true-class margin (identical form, graph identity)
- [x] Diff check: 1 file, +8/−2 — constant + two loss sites only
- [x] CPU sanity: (a) params 4,286,026; (b) m=0 identity; argmin(loss vs gap) shifts 4.512→5.262 (+0.750 exactly); (c) gradient signs at gap=opt+0.09: plain +0.007 (down) vs margin −0.071 (up); (d) smoke monotone 7.74→0.96. NOTE: original (b) "margin loss > plain on confident logits" was a wrong invariant (LS interaction) — replaced with argmin-shift; recorded in exp-log Surprises

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Reused `/tmp/exp046_composite.sh` AS-IS
- [x] GATE_DECISION D0=22.7ms ≤ 23 — family dt; GATES_CLEAR poll 1 (apps=0, load=6), pid 1737561 launched 03:19:56

### Milestone 3: Run resolved — full completion OR pre-registered branch
- [x] Trajectory normal; no abort signals; plateau converged-flat (depressed — branch iii shape)
- [x] No kills — gates clear poll 1, slow_streak 0 throughout
- [x] Completed: RC=0, full summary block (best 96.19, 139 ep, 13,431 steps, 485.6s total)

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition PASS; Condition 1 FAIL (96.19 < 96.81) — first-failure-stop; results in exp-log-050.md

## Code Changes
- **train.py** (only file): (1) new constant `MARGIN = 0.75`; (2) timed-loop loss line subtracts `MARGIN * F.one_hot(targets, NUM_CLASSES).to(outputs.dtype)` from `outputs` inside the existing autocast block; (3) warmup loss line gets the identical form with `warm_y` so the compiled graph matches the timed loop. Why this tests the hypothesis: the subtraction makes every sample appear m less confident to the loss, so gradient pressure persists until the true-class logit gap exceeds the LS optimum + m — converged decision boundaries sit further from training points; eval (Eval.evaluate on eager base_model) is untouched, so any gain is realized boundary placement, not measurement change. Risks/edge cases: `F.one_hot` requires int64 targets (they are); `.to(outputs.dtype)` keeps bf16 graph purity; loss display inflation is cosmetic; destroyed-label samples under TA+RE also get pushed — the pre-registered branch (iii) failure shape.

## Configuration Changes
- `MARGIN`: new, 0.75. Rationale: ≈ 17% of the LS-converged logit-gap optimum log(0.91/0.01) ≈ 4.51 (Müller et al. 2019 arithmetic, brainstorm-050) — large enough to move boundaries measurably, small enough not to fight LS (m ≪ gap). Single dose knob; branches (ii)/(iii) calibrate any follow-up dose. All other constants unchanged (PEAK_LR 0.4, WARMUP 0.15, LS 0.1, WD 5e-4, batch 512).

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` reused verbatim (validated 4× including EXP-049): dual launch gates (GPU-0 zero compute apps AND load < 60, poll 30s×240) → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → watchdog 44×15s (GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN; divergence eval < 15% after ep5; WALL_CAP tick 44)
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~470–505s clean
- Log output: `<project root>/run.log` + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- NaN loss or eval < 15% after epoch 5 → research failure, no retry
- GATE_KILL / CONTENTION_KILL / STARTUP_KILL → infra (elementwise subtract cannot alter dt): relaunch byte-identically when gates clear (max 2, then Outcome failed)
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27; num_epochs 136–142; num_steps within ~1% of family ledger 13,428–13,515; printed params == 4,286,026; training_seconds == 300.0; eval lines ≤ num_epochs. Numerics by trajectory + plateau criterion (TEST evals only; single-eval bands unreliable — EXP-010/048). NOTE: final_test_loss MAY legitimately shift from the family ~0.185 band this time — the margin changes converged logit scale — so test_loss is informational, NOT an integrity gate for this experiment. Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → replicate pair (two byte-identical gated runs; improvement only if mean ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185; a HIGHER value with equal/higher accuracy is consistent with the margin mechanism — larger logit gaps raise CE on miscalibrated samples; record, don't gate)
- num_epochs / num_steps: ledger cross-check (expect family values — confirms the subtract is throughput-free)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- Plateau shape (last-8 evals): converged-flat expected; level vs the EXP-027 mean is the dose-response datum for the loss-geometry class

# Plan EXP-052: Replicate-pair resolution of the anti-aliased shortcut (EXP-046 re-applied; n=2, MEAN decision)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16; σ_mean(n=2) ≈ 0.113. EXP-046 prior read: 96.65 (+0.5σ) at pristine signatures. Decision statistic: **MEAN of two fresh byte-identical runs** — the max is never used (max would be scatter farming; the mean at the same bar is STRICTER than the standard single-run protocol: 1.6% vs 6.7% false-positive under H0).

Projection (measured in EXP-046): params **4,286,026** (zero-param change), dt 22.0–22.7ms, ~139 epochs, family steps. Failed-Approaches retry justification (per checkpoint): the EXP-046 entry (count 1, Low importance) recorded a correct n=1 verdict; this plan changes the MEASUREMENT (n=2, mean), not the approach — a resolution follow-up on the only positive-direction datum on the board, pre-registered terminal on every branch.

## Milestones

### Milestone 1: Code change re-applied and passing CPU sanity
- [x] `train.py` `BasicBlock.forward`: replace `shortcut = shortcut[:, :, :: self.stride, :: self.stride]` with `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)` (channel zero-pad line unchanged; nothing else touched) — byte-equivalent to the EXP-046 diff (plan-046 M1)
- [x] Diff check: `git diff --stat` shows 1 file, logic confined to BasicBlock.forward (1 file changed, +2/−1, hunk only in BasicBlock.forward)
- [x] CPU sanity `/tmp/exp052_sanity.py` (rebuild of the validated exp046 pattern): params == 4,286,026; forward (4,3,32,32) → (4,10) finite; constant-input semantic check (new == old on constants, differs on random); stride-1 blocks unaffected (need_pad True only at layer2[0]/layer3[0] for stride reasons — assert padding sites); 2-step train smoke decreasing — ALL PASS (smoke 2.571 → 1.716)

### Milestone 2: Run A — gated launch, completion, pristine check
- [x] Launch `/tmp/exp046_composite.sh` AS-IS (gate 26ms; the change measured D0=22.5ms in EXP-046)
- [x] GATE_DECISION D0 ≤ 23ms (22.5ms); run to completion (rc=0, summary block); best_A = 96.84
- [x] Pristine check per integrity pre-condition below — PASS (windows 22.0–23.2ms; 138 ep; 13,349 steps; 300.0s; run.log preserved to /tmp/exp052_runA.log)

### Milestone 3: Run B — byte-identical second run
- [x] Re-launch the SAME composite with the SAME working tree (asserted `git diff --stat` unchanged: train.py +2/−1 only)
- [x] Same gates/completion/pristine checks — D0 = 22.0ms, PASS (windows 21.7–23.2ms; 138 ep; 13,363 steps); best_B = 96.56
- [x] MEAN = (96.84 + 96.56)/2 = **96.70** — the decision statistic

### Milestone 4: Verification executed (first-failure-stop on the MEAN) and exp-log updated
- [x] Integrity pre-condition PASS both runs; Condition 1 MEAN 96.70 < 96.81 → FAIL → branch (ii) weak-positive-closed; results recorded in exp-log-052.md

## Code Changes
- **train.py** (only file, one line of logic in `BasicBlock.forward`): identical to EXP-046 — the pad shortcut's strided slice `[::2,::2]` (which discards 75% of identity signal and aliases the rest) → `F.avg_pool2d(shortcut, self.stride)` guarded by `if self.stride != 1`. Affects exactly two forward sites (layer2[0], layer3[0]); zero parameters; deterministic, mode-independent op (no train/eval asymmetry). Why this tests the hypothesis: if the BlurPool-class effect partially survives absorption (+0.1–0.2 true), two draws resolve it against the 96.81 bar at 1.6% false-positive; the change itself was already measured signature-clean and is NOT re-derived here.

## Configuration Changes
- None. Pure function-quality probe; every constant baseline (the EXP-046 design, unchanged).

## Execution Environment
- Method: local, `/tmp/exp046_composite.sh` verbatim, run TWICE sequentially (Run B starts only after Run A's verdict-relevant data is extracted and run.log A is preserved to /tmp/exp052_runA.log before deletion — the composite itself does `rm -f run.log` at launch, so copy A's log first)
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached
- Estimated runtime: ~16–18 min total for two clean runs (each ~470–505s + gate polls)
- Log output: run.log per run (Run A copied to /tmp/exp052_runA.log before Run B launch) + composite stdout per task
- Tool skill: none (local)

## Abort Criteria
- Per run: NaN, eval < 15% after ep5 → research failure, no retry
- GATE_KILL / CONTENTION_KILL / STARTUP_KILL → infra: relaunch that run byte-identically when gates clear (max 2 per run, then Outcome failed)
- Wall ≥ ~660s per run → kill, failure
- If Run A completes pristine but Run B cannot be obtained within retries → record Run A only, verdict no-improvement UNLESS best_A ≥ 96.81 alone would have passed the STANDARD protocol — in that exact case fall back to the standard single-run condition (pre-registered here to remove ambiguity; the mean protocol is a stricter overlay, not a way to discard a bar-clearing run)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition (each run independently)**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27; num_epochs 136–142; num_steps within ~1% of 13,428–13,515; params 4,286,026; training_seconds 300.0; evals ≤ epochs; trajectory family-shaped. A contention-tainted run is rerun (infra), never averaged.

1. **MEAN(best_A, best_B) ≥ 96.81** (pre-registered pair decision; baseline 96.71 + 0.1pp): extract each via `grep "^best_test_acc:" <log>` (timeout 10s; empty ⇒ crash path `tail -n 50`). Pass: mean ≥ 96.81 → improvement (commit the shortcut change; TSV metric = the mean). Fail: mean < 96.81 (branches: (ii) mean ∈ [96.61, 96.80] weak-positive-closed; (iii) mean ≤ 96.60 confirmed-null-closed). The max of the pair is NOT a decision input.
2. **Within budget (each run)**: composite rc == 0 AND `grep "^total_seconds:"` ≤ 600 (timeout 10s)
3. **Eval cadence (each run)**: `grep -c "eval ep"` ≤ num_epochs (timeout 10s)

Cleanup per goal Procedure: delete run.log and /tmp/exp052_runA.log at loop end.

### Informational Metrics (Optional)
- Per-run best/final_test_loss/epochs/steps (family expected; EXP-046 read test_loss 0.185 family)
- Spread |best_A − best_B| (σ check: expect ~0.1–0.25; a spread > 0.5 flags an integrity question)
- Pooled view: the three draws (96.65, A, B) vs EXP-027 mean — recorded in the report for the permanent-closure statement

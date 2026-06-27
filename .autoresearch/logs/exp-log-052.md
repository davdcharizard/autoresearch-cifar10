# Experiment Log EXP-052: Replicate-pair resolution of the anti-aliased shortcut (n=2, MEAN decision)

## Execution

- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-052
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Re-applied EXP-046's exact one-logic-line change to `train.py` `BasicBlock.forward`: the pad shortcut's strided slice `shortcut[:, :, ::self.stride, ::self.stride]` → `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)` (channel zero-pad line unchanged). Diff check confirms 1 file changed (+2/−1), hunk confined to BasicBlock.forward. CPU sanity `/tmp/exp052_sanity.py` all pass: (a) params 4,286,026 exactly (zero-param change); (b) forward (4,3,32,32) → (4,10) finite; (c) semantic check — avg_pool2d(s=2) equals the old strided slice on constant inputs and differs on random inputs at matching shapes; (d) need_pad sites exactly [layer2[0], layer3[0]] both stride-2, so stride-1 blocks are untouched and the new `if self.stride != 1` guard is exercised only where intended; (e) 2-step train smoke decreasing (2.571 → 1.716). M1 complete. Per plan M2/M3: two byte-identical gated composite runs (`/tmp/exp046_composite.sh` verbatim), decision statistic = MEAN(best_A, best_B) ≥ 96.81; Run A's run.log preserved to /tmp/exp052_runA.log before Run B launches (composite deletes run.log at start).

### Surprises & Discoveries
- None at implementation time — the diff and sanity pattern were both validated in EXP-046 and reproduced without deviation.

### Decisions
- Decision statistic pre-registered as the MEAN of the two runs (never the max) — stricter than the standard single-run protocol (1.6% vs 6.7% false-positive under H0 at the 96.81 bar). Branches pre-registered in plan-052: (i) mean ≥ 96.81 improvement; (ii) mean ∈ [96.61, 96.80] weak-positive-closed; (iii) mean ≤ 96.60 confirmed-null-closed at n=3 total draws; (iv) gate/contention kills → infra relaunch (max 2 per run). Fallback if Run B unobtainable after retries: no-improvement unless best_A ≥ 96.81 alone (standard-protocol fallback, pre-registered).

## Run Log

### Run 1 (Run A)

**Description**: First of two byte-identical runs of the anti-aliased-shortcut variant (EXP-046 diff re-applied). Launched via `/tmp/exp046_composite.sh`: dual launch gates (zero GPU-0 compute apps AND 1-min load < 60, poll 30s × 240), then background `uv run train.py > run.log 2>&1` with a 44×15s watchdog (NaN / divergence <15% after ep5 / WALL_CAP ~660s / startup checks). Expected: D0 ≈ 22.5ms (EXP-046 measured), ~139 epochs, family steps ledger 13,428–13,515, best_test_acc one draw from the variant's distribution. run.log will be copied to /tmp/exp052_runA.log on completion, before Run B.

**Metadata**:
- Job ID: background task bwmu7gloe (composite pid 1797140)
- Log file: run.log → preserved at /tmp/exp052_runA.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 03:56:28
- Ended: 2026-06-11 ~04:04:40

**Observations**: PRISTINE. GATES_CLEAR at poll 1 (apps=0, load=11). GATE_DECISION D0=22.5ms (identical to EXP-046's measured D0), projected 137 ep. All watch windows 22.0–23.2ms (mean ≤23.5 satisfied, none >27, slow_streak 0 throughout). Summary: **best_test_acc 96.84%**, final 96.79, final_test_loss 0.1790 (family), training_seconds 300.0, total 493.2 ≤ 600, startup 11.9, epochs 138 (∈136–142), steps 13,349 (within ~1% of family band 13,428–13,515: ≥13,294), params 4,286,026, evals 138 ≤ 138 epochs. Tail plateau converged-flat: last 4 evals 96.84/96.83/96.82/96.79. best_A = 96.84 — above the 96.81 bar on its own; decision still awaits MEAN with Run B (mean clears iff best_B ≥ 96.78).

### Run 2 (Run B)

**Description**: Second byte-identical run, same composite, same working tree (asserted via `git diff --stat` unchanged between runs). MEAN(best_A, best_B) is the pre-registered decision statistic.

**Metadata**:
- Job ID: background task b27f5s2b8 (composite pid 1811341)
- Log file: run.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 04:05:23
- Ended: 2026-06-11 ~04:13:50

**Observations**: PRISTINE. Working tree asserted unchanged before launch (`git diff --stat` identical: train.py +2/−1 only). GATES_CLEAR at poll 1 (apps=0, load=23). GATE_DECISION D0=22.0ms. All watch windows 21.7–23.2ms, slow_streak 0 throughout. Summary: **best_test_acc 96.56%**, final 96.54, final_test_loss 0.1826 (family), training_seconds 300.0, total 503.8 ≤ 600, epochs 138 (∈136–142), steps 13,363 (within ~1% band), params 4,286,026, evals 138 ≤ 138 epochs. Tail plateau converged-flat (96.56/96.55/96.50/96.54). **MEAN = (96.84 + 96.56)/2 = 96.70 < 96.81** — pre-registered branch (ii): weak-positive sub-bar. Spread |A−B| = 0.28 (within noise expectation; below the 0.5 integrity flag).

## Experimental Adjustments

- None — both runs executed byte-identically on the first launch attempt; no retries, no parameter changes.

## Errors & Dead Ends

- None.

## Verification Results

### Conditions Checked

**Integrity pre-condition (per run, both PASS)**:
- Run A: windows 22.0–23.2ms (≤23.5 mean, none >27) ✓; epochs 138 ∈ [136,142] ✓; steps 13,349 within ~1% of 13,428–13,515 ✓; params 4,286,026 ✓; training_seconds 300.0 ✓; evals 138 ≤ 138 epochs ✓; trajectory family-shaped, converged-flat tail ✓. Source: /tmp/exp052_runA.log, composite task bwmu7gloe output.
- Run B: windows 21.7–23.2ms ✓; epochs 138 ✓; steps 13,363 ✓; params 4,286,026 ✓; training_seconds 300.0 ✓; evals 138 ≤ 138 ✓; family-shaped ✓. Source: run.log, composite task b27f5s2b8 output.

**Condition 1 — MEAN(best_A, best_B) ≥ 96.81**: best_A = 96.84 (grep `^best_test_acc:` /tmp/exp052_runA.log), best_B = 96.56 (grep run.log). MEAN = 96.70 < 96.81 → **FAIL** (first-failure-stop). Pre-registered branch (ii): mean ∈ [96.61, 96.80] → weak-positive-closed. The max of the pair (96.84) is NOT a decision input per plan-052; the single-run fallback applies only if Run B were unobtainable — it was obtained pristine.

**Condition 2 — within budget (each run)**: rc=0 both; total_seconds 493.2 / 503.8 ≤ 600 → PASS (informational; condition 1 already failed).

**Condition 3 — eval cadence**: 138 evals ≤ 138 epochs both runs → PASS (informational).

**Verdict basis**: no-improvement (valid result, pre-registered mean decision below bar).

## Human Notes

(autopilot — none)

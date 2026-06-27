# Experiment Log EXP-053: Cross-axis compound — anti-aliased shortcut + de-overhead prefetch (n=2, MEAN decision)

## Execution

- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-053
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Re-applied both certified-free diffs in one train.py on branch autoresearch/exp-053. (1) Shortcut: `BasicBlock.forward` strided slice → `if self.stride != 1: F.avg_pool2d(shortcut, self.stride)` (EXP-046/052 diff, byte-equivalent). (2) De-overhead pair (EXP-048 diff): module-level `collate_channels_last` (default_collate + contiguous channels_last) wired via `collate_fn=`; module-level `CUDAPrefetcher` (side stream, wait_stream/record_stream, CPU passthrough fallback); loop iterates `CUDAPrefetcher(train_loader, device)` with the two in-step `.to()` lines deleted and `t0` now immediately preceding the `progress` computation (timed region otherwise byte-identical). Diff: 1 file, +52/−7, hunks confined to the four planned sites. Merged CPU sanity `/tmp/exp053_sanity.py` ALL PASS: params 4,286,026 exact; forward finite; shortcut semantics (constant-equal/random-differ) + pad sites [layer2[0], layer3[0]]; collate value-identity (`torch.equal`) + channels_last contiguity; prefetcher sequence-identity 7/7 batches over two passes; 2-epoch DataLoader smoke decreasing (2.767 → 2.105). AST parse OK. M1 complete.

### Surprises & Discoveries
- None — both diffs reproduced from plan-046/048 without deviation; sanity patterns merged cleanly.

### Decisions
- Decision statistic pre-registered as MEAN of two byte-identical runs ≥ 96.81 (max never used), per the EXP-052-validated protocol. Branches: (i) mean ≥ 96.81 → improvement, commit both diffs; (ii) mean ∈ [96.61, 96.80] → weak-positive, compound-of-frees closed at resolution limit; (iii) mean ≤ 96.60 → sub-additive/null, closed with negative datum; (iv) infra relaunch (max 2/run). Fallback if Run B unobtainable: no-improvement unless best_A ≥ 96.81 alone.
- Numerics judged by the EXP-048 trajectory criterion (rejoin family by ~ep7, family plateau + test_loss); single ep1 reads informational except the < 30% defect tripwire.

## Run Log

### Run 1 (Run A)

**Description**: First of two byte-identical runs of the compound variant. Launched via `/tmp/exp046_composite.sh` (dual gates: zero GPU-0 compute apps AND load < 60; watchdog 44×15s with GATE_KILL > 26ms, contention, NaN, divergence, WALL_CAP). Expected: D0 ≈ 22.0–22.5ms, 136–143 epochs, steps ≈ 13,350–13,650 (prefetch should reproduce the +87-step saving over the 046-family ledger), family trajectory. run.log copied to /tmp/exp053_runA.log on completion before Run B.

**Metadata**:
- Job ID: background task bpfez12fu (composite pid 1846689)
- Log file: run.log → preserved at /tmp/exp053_runA.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 04:26:46
- Ended: 2026-06-11 ~04:35:10

**Observations**: PRISTINE. GATES_CLEAR poll 1 (apps=0, load=13). GATE_DECISION D0=22.3ms. All watch windows 21.7–22.7ms, slow_streak 0. ep1 = 37.66 — inside the 36–41 family band, prefetcher-defect tripwire passed; trajectory family-shaped throughout. Summary: **best_test_acc 96.61**, final 96.60, final_test_loss 0.1818 (family), training_seconds 300.0, total 506.3 ≤ 600, epochs 139 (∈136–143), steps 13,428 (∈13,290–13,650; NOTE: exactly the 046-family figure — the +87-step prefetch saving did NOT visibly reproduce this draw, within per-step scatter), params 4,286,026, evals 139 ≤ 139. Tail converged-flat (96.61/96.58/96.61/96.54/96.60). best_A = 96.61: mean ≥ 96.81 now requires best_B ≥ 97.01 (+2.75σ) — Run B proceeds per pre-registration and decides the closure branch (ii) vs (iii).

### Run 2 (Run B)

**Description**: Second byte-identical run, same composite, same working tree (asserted via `git diff --stat` unchanged). MEAN(best_A, best_B) is the decision statistic.

**Metadata**:
- Job ID: background task b0hjv5hef (composite pid 1863524)
- Log file: run.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 04:36:05
- Ended: 2026-06-11 ~04:44:20

**Observations**: PRISTINE. Working tree asserted unchanged before launch (train.py +52/−7 only). GATES_CLEAR poll 1 (apps=0, load=12). GATE_DECISION D0=22.5ms. Windows 21.7–22.7ms, slow_streak 0. Summary: **best_test_acc 96.28**, final 96.25, final_test_loss 0.1915 (slightly above family ~0.185 but within scatter), 300.0s, total 495.2 ≤ 600, epochs 139, steps 13,434, params 4,286,026, evals 139 ≤ 139. Tail converged-flat. **MEAN = (96.61 + 96.28)/2 = 96.445 ≤ 96.60** — pre-registered branch (iii): sub-additive/null. Spread |A−B| = 0.33 (~1.5σ of a pair difference; below the 0.5 integrity flag). Steps ledger: 13,428/13,434 — the +87-step prefetch saving did NOT reproduce in either run (both at the 046-family figure).

## Experimental Adjustments

- None — both runs executed byte-identically on first launch; no retries.

## Errors & Dead Ends

- None.

## Verification Results

### Conditions Checked

**Integrity pre-condition (per run, both PASS)**:
- Run A: windows 21.7–22.7ms ✓; epochs 139 ∈ [136,143] ✓; steps 13,428 ∈ [13,290–13,650] ✓; params 4,286,026 ✓; 300.0s ✓; evals 139 ≤ 139 ✓; ep1 37.66 ∈ [36,41] + family trajectory + family test_loss 0.1818 ✓. Source: /tmp/exp053_runA.log, task bpfez12fu.
- Run B: windows 21.7–22.7ms ✓; epochs 139 ✓; steps 13,434 ✓; params ✓; 300.0s ✓; evals 139 ≤ 139 ✓; family trajectory, test_loss 0.1915 within scatter ✓. Source: run.log, task b0hjv5hef.

**Condition 1 — MEAN(best_A, best_B) ≥ 96.81**: best_A = 96.61 (/tmp/exp053_runA.log), best_B = 96.28 (run.log). MEAN = 96.445 < 96.81 → **FAIL** (first-failure-stop). Pre-registered branch (iii): mean ≤ 96.60 → sub-additive/null; compound-of-frees closed with a negative datum. Max never used.

**Condition 2 — within budget**: rc=0 both; 506.3 / 495.2 ≤ 600 → PASS (informational).

**Condition 3 — eval cadence**: 139 ≤ 139 both runs → PASS (informational).

**Verdict basis**: no-improvement (valid pristine pair, pre-registered mean decision far below bar).

## Human Notes

(autopilot — none)

# EXP-067: σ-tightening baseline replicate pair (zero-diff, n=3 → n=5, drift re-anchor)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md
- **Plan**: plans/plan-067.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-067
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Zero-code-change experiment per plan: train.py verified byte-identical to autoresearch/dev HEAD
(`git diff --quiet -- train.py` → ZERO_DIFF_OK) on branch autoresearch/exp-067. Composite
launcher /tmp/exp067_composite.sh created from the exp061 template (header rename only); two
sequential gated runs collect fresh baseline draws R1, R2. Recorded metric = mean(R1, R2);
verdict pre-registered no-improvement (EXP-027 protocol). Product = pooled n=5 σ̂ over
{96.71, 96.59, 96.40, R1, R2} plus a level-layer drift check (band [96.25, 96.89]).

### Surprises & Discoveries
- None yet.

### Decisions
- Pre-registered: even a single draw ≥ bar does NOT change the verdict — zero-diff draws are
  distribution samples, not interventions; harvesting one would be variance mining (EXP-052:
  the max of a pair is never a decision input).

## Experimental Adjustments

## Run Log

### Run 1 (replicate R1)

Metadata:
- **Job ID**: background task buekqcq7l (composite /tmp/exp067_composite.sh)
- **Log file(s)**: run.log (project root); composite telemetry /tmp/exp067_composite_run1.log
- **WandB**: N/A
- **Status**: completed (rc=0, pristine — no watchdog events)
- **Started**: 2026-06-11
- **Ended**: 2026-06-11

Description:
- First of two byte-identical baseline draws. Standard composite gates/watchdog. Expected:
  family signatures (dt 22.0–22.8ms, 137–140 ep, steps 13,100–13,600, params 4,286,026) and
  best_test_acc in [96.25, 96.89].

Observations:
- Pristine: D0 = 22.5ms, projected 137 ep, all post-gate windows 22.0–22.7ms, slow_streak
  never > 0, rc=0 (source: /tmp/exp067_composite_run1.log).
- **R1 = 96.53 — IN-BAND** [96.25, 96.89]; integrity PASS: 13,455 steps (family), 139 ep,
  params 4,286,026, total 492.5s, test_loss 0.1867 (family ~0.185), long flat plateau
  (source: /tmp/exp067_composite_run1.log SUMMARY + last 8 evals).

Key Metrics:
- best_test_acc: 96.53% @ ep137 of 139; num_steps 13,455; VRAM 1613.0
  (source: /tmp/exp067_composite_run1.log SUMMARY)

### Run 2 (replicate R2)

Metadata:
- **Job ID**: background task blml9vdwv (composite /tmp/exp067_composite.sh)
- **Log file(s)**: run.log (project root); composite telemetry /tmp/exp067_composite_run2.log
- **WandB**: N/A
- **Status**: completed (rc=0, pristine — no watchdog events)
- **Started**: 2026-06-11
- **Ended**: 2026-06-11

Description:
- Second byte-identical baseline draw, same composite, fresh invocation after run.log deletion.
  Same expectations as Run 1.

Observations:
- Pristine: D0 = 22.7ms, projected 136 ep, zero watchdog events, rc=0
  (source: /tmp/exp067_composite_run2.log).
- **R2 = 96.44 — IN-BAND** [96.25, 96.89]; integrity PASS: 13,461 steps, 139 ep, params
  4,286,026, total 486.3s, test_loss 0.1892 (family), flat plateau
  (source: /tmp/exp067_composite_run2.log SUMMARY + last 8 evals).

Key Metrics:
- best_test_acc: 96.44% @ ep136/139; num_steps 13,461; VRAM 1613.0
  (source: /tmp/exp067_composite_run2.log SUMMARY)

## Verification Results

### Conditions Checked

- **Integrity pre-condition (per run, gates pooling)**: PASS both runs — steps 13,455 / 13,461
  ∈ [13,100, 13,600]; params 4,286,026 both; D0 22.5 / 22.7ms with zero slow streaks; epochs
  139 both; family test_loss 0.1867 / 0.1892. Both draws POOLABLE
  (source: /tmp/exp067_composite_run{1,2}.log).
- **Condition 1 — recorded metric (mean(R1,R2)) ≥ bar 96.81**: FAIL as pre-registered —
  mean(96.53, 96.44) = 96.485 < 96.81. Verdict: no-improvement (zero-diff replicate pair;
  improvement was impossible by design).
- **Condition 2 — each run ≤ 600s total**: not evaluated for verdict (aborted after first
  failure); informationally PASS — 492.5s / 486.3s, rc=0 both.
- **Condition 3 — validation once per epoch**: not evaluated for verdict; informationally PASS
  (zero code diff — structurally guaranteed).

### Informational Metrics

- R1 = 96.53, R2 = 96.44 — BOTH IN-BAND [96.25, 96.89]: **no drift detected**, level ledger
  re-anchored months into the program.
- **Pooled n=5 statistics (the experiment's product)**: draws {96.71, 96.59, 96.40, 96.53,
  96.44} → mean 96.534, sample σ̂ = 0.123 (was ~0.16 at n=3; hypothesis band [0.10, 0.22]
  confirmed). Derived: mean ± 2σ̂ = [96.29, 96.78]; the standing bar 96.81 (baseline 96.71 +
  0.1) now sits at mean + 2.24σ̂ — slightly MORE selective than the prior mean + 1.5σ reading;
  effect-size screen for future candidates: true effect ≥ +0.28 ≈ 2.2σ̂ needed to clear the bar
  at n=1.
- peak_vram_mb 1613.0 both; num_epochs 139 both; num_steps 13,455 / 13,461 (ledger re-anchor:
  the 046-family step band reproduces exactly).

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

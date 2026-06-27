# Plan EXP-069: Data-composition corner closure at zero charged cost (label-noise curation, subset selection, duplicate handling)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md

## Pre-registered protocol (EXP-063/064/068 lineage: inequality-gated NO LAUNCH)

- **Zero code diff**: train.py is NOT modified. Verify `git diff --quiet -- train.py`.
- **Zero charged GPU seconds**: no launch, no probe — every bound composes the brainstorm's
  searched evidence and in-ledger measured anchors.
- **Recorded metric = NaN, verdict pre-registered = invalid** (NO-LAUNCH precedent),
  regardless of branch outcomes.
- **Escape hatch (falsification branch)**: any sub-class whose honest candidate-favoring
  bound clears bar 96.81 is promoted to the EXP-070 brainstorm lead (not launched this
  loop). An inequality that cannot be closed with honest anchors MUST fire the hatch — it
  must not be forced closed.
- **Decision arithmetic (standing, EXP-067)**: pooled mean 96.534, σ̂ 0.123, bar 96.81 =
  mean + 2.24σ̂; required true effect ≥ +0.28.
- **Anchor discipline**: every quantity cites a measured ledger source or a brainstorm-069
  § Web Search URL; uncertain quantities take the candidate-FAVORING end.

## Milestones

### Milestone 1: Setup and integrity
- [x] On branch autoresearch/exp-069 (cut from autoresearch/dev); `git diff --quiet -- train.py` passes; no run.log exists.
- [x] exp-log-069.md created from template; Outcome pending.

### Milestone 2: Sub-class (i) — label-noise curation (drop/relabel ~0.5–1% natural train errors)
- [x] Work the bound: absorption screen (published +0.9pp is 20–40% ADDED-noise regime, arXiv 1911.00068; natural rate ~0.54% confirmed test-set / same pipeline train-set, arXiv 2103.14749 — no ≥0.3pp natural-rate evidence); mechanism redundancy (LS 0.1 measured flat-on-[0.1,0.2] EXP-036 + heavy TA+RE supply the robustness cleaning would add; per-sample loss interventions measured negative BOTH directions EXP-050/051); cost-landing screen (honest error identification needs trained-model predictions = charged ranking passes; an imported precomputed error list rides on externally trained models — pretrained-knowledge class — and even granting it, the effect ceiling stays sub-screen); effect-size arithmetic (≤500 of 50,000 labels ⇒ ≤0.1pp class ≪ +0.28).
- [x] Record worked bound + explicit branch decision (CLOSED or ESCAPE-HATCH) in exp-log § Run Log.

### Milestone 3: Sub-class (ii) — subset selection / importance sampling / dataset pruning under the fixed budget
- [x] Work the bound: published budget-matched NULL (arXiv 2110.14283, surfaced brainstorm-066); per-epoch example-count reduction is pressure-DOWN on the bracketed regularization/pressure axis (EXP-015 WD-down −0.30; EXP-033 light-tail freeze; pressure must be constant-on at certified level, EXP-025/033/065 four-quadrant law); selection-weighting is the EXP-051 anti-curriculum mechanism (confidence-keyed suppression −7.8σ).
- [x] Record worked bound + branch decision in exp-log § Run Log.

### Milestone 4: Sub-class (iii) — near-duplicate handling (train↔test near-duplicates)
- [x] Work the bound: Eval pins the official test distribution (prepare.py read-only), and the official train set is the matched training distribution; removing train near-duplicates strictly removes information about test-adjacent modes (can only lower best_test_acc); adding/duplicating examples re-weights the empirical distribution = composition-side pressure change on a bracketed axis. Note the integrity boundary: train-on-test or test-informed selection would be leakage (banned), so the only legal moves in this sub-class are removals/re-weightings, both bounded ≤ mean.
- [x] Record worked bound + branch decision in exp-log § Run Log.

### Milestone 5: Ledger completion check + verification
- [x] Record the enumeration-completion statement: with composition closed, every member of the goal-learnings residual space ("data composition/order, objective shaping, architecture") carries a closure citation (EXP-041; EXP-050/051+009/036; lattice/depth/head/kernel/attention/reparam family closures; this loop).
- [x] Confirm zero charged seconds (no run.log) and zero diff maintained; evaluate pre-registered branches; record in exp-log § Verification Results; Outcome → completed.

## Code Changes
- **None.** train.py byte-identical to autoresearch/dev HEAD (1990397) throughout.
- **No /tmp scripts.**

## Configuration Changes
- None.

## Execution Environment
- Method: documentation-only loop; zero GPU. All execution is working three bounded
  arguments + one enumeration check against cited anchors, recorded in the exp-log.
- Resources: repo + .autoresearch ledger only.
- Estimated runtime: ~10–15 minutes of artifact work.
- Log output strategy: exp-log-069.md § Run Log carries one entry per sub-class (i/ii/iii)
  plus the enumeration check; no run.log is ever created (absence checked in Milestone 5).
- Tool skill: none.

## Abort Criteria
- No run to monitor. Loop-level abort: if any cited anchor fails to support its quantity on
  re-read, re-derive with the corrected anchor and record in § Experimental Adjustments; if
  an honest bound then clears the bar, fire the escape hatch rather than forcing closure.

## Verification Protocol

### Verification Procedure
Follows goals/maximize-cifar10-test-accuracy.md § Procedure where applicable; resolves as in
the EXP-063/064/068 NO-LAUNCH precedent:

1. **Condition 1 — best_test_acc ≥ bar 96.81** (baseline 96.71 via `exp-index.sh baseline`
   + 0.1): NOT SATISFIABLE — no run, no metric. Recorded metric = NaN; verdict **invalid**
   per precedent.
2. **Condition 2 — run ≤ 600s total**: vacuous; substitute check — `ls run.log` must fail
   at loop end.
3. **Condition 3 — validation at most once per epoch**: vacuous; structurally guaranteed by
   zero diff.
4. **Loop-specific integrity**: (a) `git diff --quiet -- train.py` at loop end; (b) every
   bound quantity carries an EXP citation or brainstorm-069 URL; (c) all sub-class entries
   state explicit branch decisions; (d) the enumeration-completion statement lists a closure
   citation for every residual-space member.

### Informational Metrics (Optional)
- The three worked bounds (numeric where applicable, vs bar 96.81).
- Escape-hatch status per sub-class (expected: none fired).
- Charged GPU seconds (expected 0); probe seconds (expected 0); code diff (expected zero).

# Plan EXP-070: Gap-completion closure loop — infra/throughput angle + data-addition bound (zero charged cost)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md

## Pre-registered protocol (EXP-063/064/068/069 lineage: inequality-gated NO LAUNCH)

- **Zero code diff**: train.py is NOT modified. Verify `git diff --quiet -- train.py`.
- **Zero charged GPU seconds**: no launch, no probe this loop — both bounds compose searched
  primary sources (brainstorm-070 § Web Search) with measured laws; neither depends on an
  unmeasured dt. (If the infra bound were to SURVIVE, the follow-up is an EXP-064-style
  internal-control probe in the NEXT loop, not here.)
- **Recorded metric = NaN, verdict pre-registered = invalid** (NO-LAUNCH precedent).
- **Escape hatch**: any bound that survives honest candidate-favoring arithmetic is promoted
  to the EXP-071 brainstorm lead; never force a closure.
- **Decision arithmetic (standing, EXP-067)**: pooled mean 96.534, σ̂ 0.123, bar 96.81 =
  mean + 2.24σ̂; required true effect ≥ +0.28. Conversion law: +0.019/epoch (EXP-006),
  ~139 epochs at dt 22.4ms ⇒ clearing the bar via throughput alone needs ≥ +15 epochs
  ≈ −2.4ms/step ≈ 11% numerics-IDENTICAL speedup.
- **Anchor discipline**: every quantity cites a measured ledger source or a brainstorm-070
  § Web Search URL; uncertain quantities take the candidate-FAVORING end.

## Milestones

### Milestone 1: Setup and integrity
- [x] On branch autoresearch/exp-070 (cut from autoresearch/dev); `git diff --quiet -- train.py` passes; no run.log exists.
- [x] exp-log-070.md created from template; Outcome pending.

### Milestone 2: Bound A — infra/throughput angle (the one legal throughput escape)
- [x] Work the bound with cited anchors: (1) availability — the searched stack (PyTorch 2.9 release notes/blog, GH v2.9.0, issue tracker) offers NO new numerics-preserving CUDA conv path; 2.9's conv work is AArch64-scoped; the known 2.9.0 conv regression is 3D+AMP (irrelevant — 2D convs, family dt 22.4ms ledger-stable); (2) the knob class that exists (cuDNN algo selection, e.g. torch.backends.cudnn.benchmark_limit) produces faster-but-DIFFERENT arithmetic — the measured EXP-021 trap (max-autotune/cudagraphs/fused: +10 epochs yet −0.20, replicated), and EXP-054 brackets precision-side numerics flat; (3) required-gain inequality — bar-clearing via conversion needs ≥11% numerics-identical speedup vs a measured 99.3%-kernel floor (EXP-048) at an already-cudnn.benchmark'd, compile-floored config; algo-benchmarking marginals are an order short.
- [x] Record worked bound + explicit branch decision (CLOSED or ESCAPE-HATCH) in exp-log § Run Log.

### Milestone 3: Bound B — data addition (synthetic/generated training examples)
- [x] Work the bound with cited anchors, by funding path: (1) in-budget generation — generator training/inference prices ON the charged 300s (cost-landing fail; any compute diverted from the model's own steps inherits the EXP-043 dilution arithmetic: −0.9 per halving vs +0.3–0.5 class gains); (2) imported generated/external data — rides on externally trained models (pretrained-knowledge boundary, same as EXP-069's imported-list variant) and exceeds the task's data scope (TASK.md frames the task on the provided CIFAR-10 data; isolation rules exist to prevent exactly this class of leakage); (3) augmentation-like in-budget synthesis (mixing/erasing/policy variants of existing examples) — the measured-closed pressure dose/type axis (EXP-009 stacking −0.46; EXP-060 type-substitution null; four-quadrant pressure law).
- [x] Record worked bound + branch decision in exp-log § Run Log.

### Milestone 4: Backlog-empty check + verification
- [x] Record the backlog statement: with A and B closed, no nominated-but-unbounded class remains in the ledger (EXP-068 closed the radical classes, EXP-069 the composition corner + enumeration, this loop the two residual nominations); next scheduled inflow = the 2026-06-17 periodic sweep.
- [x] Confirm zero charged seconds (no run.log) and zero diff; evaluate pre-registered branches; record in exp-log § Verification Results; Outcome → completed.

## Code Changes
- **None.** train.py byte-identical to autoresearch/dev HEAD (1990397) throughout.
- **No /tmp scripts.**

## Configuration Changes
- None.

## Execution Environment
- Method: documentation-only loop; zero GPU. Two bounded arguments + one backlog check
  against cited anchors, recorded in the exp-log.
- Resources: repo + .autoresearch ledger only.
- Estimated runtime: ~10 minutes of artifact work.
- Log output strategy: exp-log-070.md § Run Log carries one entry per bound (A, B) plus the
  backlog check; no run.log is ever created (absence checked in Milestone 4).
- Tool skill: none.

## Abort Criteria
- No run to monitor. Loop-level abort: if a cited anchor fails on re-read, re-derive with
  the corrected anchor (record in § Experimental Adjustments); if an honest bound then
  clears the bar, fire the escape hatch rather than forcing closure.

## Verification Protocol

### Verification Procedure
Follows goals/maximize-cifar10-test-accuracy.md § Procedure where applicable; resolves per
the NO-LAUNCH precedent:

1. **Condition 1 — best_test_acc ≥ bar 96.81** (baseline 96.71 via `exp-index.sh baseline`
   + 0.1): NOT SATISFIABLE — no run, no metric. Recorded metric = NaN; verdict **invalid**.
2. **Condition 2 — run ≤ 600s total**: vacuous; substitute check — `ls run.log` must fail
   at loop end.
3. **Condition 3 — validation at most once per epoch**: vacuous; structurally guaranteed by
   zero diff.
4. **Loop-specific integrity**: (a) `git diff --quiet -- train.py` at loop end; (b) every
   bound quantity carries an EXP citation or brainstorm-070 URL; (c) both bound entries
   state explicit branch decisions; (d) the backlog statement enumerates the closure chain
   (EXP-068 → 069 → 070) with the next scheduled inflow date.

### Informational Metrics (Optional)
- The two worked bounds vs bar 96.81; the required-vs-available speedup gap for Bound A.
- Escape-hatch status per bound (expected: none fired).
- Charged GPU seconds (expected 0); probe seconds (expected 0); code diff (expected zero).

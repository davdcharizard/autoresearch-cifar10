# Brainstorm EXP-071
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No external search this loop — the periodic double-screened sweep is scheduled for
  ~2026-06-17 (goal-learnings § Protocol Findings, idle-loop policy) and includes the new
  infra modality. Running queries 6 days early is the exact spend the policy interval
  exists to avoid (priced empty in brainstorm-070 candidate 2).
- knowledge/README.md re-read: 15 paper entries + 7 reference entries; the screens and
  laws the next sweep must apply are currently SCATTERED across goal-learnings entries,
  exp-reports (064/069/070), and paper notes — there is no single pre-registered sweep
  protocol artifact. That gap is addressable with zero external input.

## Experimental History Review

- 74 index rows. Baseline 96.71 @ 1990397; pooled mean 96.534, σ̂ 0.123, bar 96.81 =
  mean + 2.24σ̂, required true effect ≥ +0.28 (EXP-067 standing arithmetic). Last
  improvement EXP-006. External transfer 0-for-21.
- **State after EXP-070**: explicit-bound backlog EMPTY (closure chain EXP-068→069→070);
  steady-state idle-loop policy in force: sweep on schedule (~2026-06-17), close on
  nomination (none pending), replicate only when a near-bar decision pends (none does),
  anchor meta-audit only when anchors are stale (re-read clean three times this week —
  not yet), never manufacture GPU runs to look busy.
- **What this loop can honestly do** (exp-report-070 § Next Steps): "the smallest useful
  ledger work available." Two genuine gaps qualify:
  (a) the standing decision arithmetic's replicate pool (n=5) has never been census-audited
  against the full TSV — every closure bound cites "≥ +0.28 ≈ 2.2σ̂", making the pool's
  completeness the single most load-bearing unverified assumption in the ledger;
  (b) the 2026-06-17 sweep's screen battery exists only as scattered entries — no
  pre-registered protocol document a sweep loop can execute without re-deriving it.

## Candidate Ideas

### 1. Pre-sweep consolidation loop: replicate-pool census + pre-registered sweep protocol (zero charged cost)
**Summary**: Two small ledger entries in the NO-LAUNCH lineage. **Entry A — replicate
census**: enumerate every TSV row that is an honest baseline-config replicate (zero-diff
or family-null at byte-identical signatures), verify EXP-067's pooled n=5 is the complete
eligible pool (no eligible run omitted, no ineligible run included), and recompute
σ̂/required-effect only if the pool changes. **Entry B — sweep protocol pre-registration**:
write `knowledge/references/sweep-protocol-2026-06-17.md`, a single executable checklist
for the scheduled sweep: modality list (ML-technique classes + the new infra modality),
the full screen battery with citations (absorption law; cost-landing screen EXP-064;
noise-rate regime check EXP-069; numerics-equivalence law EXP-021; required-gain
throughput inequality 11% EXP-070; deferral law; internal-control probe gate EXP-064/066),
and the candidate-record format. Verdict pre-registered invalid/NaN; zero code diff,
zero GPU.

**Reasoning**: This is the policy's "smallest useful ledger work" branch executed
honestly. Entry A audits the number every bound in the closure ledger leans on — if the
census finds the pool wrong in either direction, σ̂ and the +0.28 threshold move, and
every near-bar margin must be re-checked (explicit escape hatch). Entry B converts
scattered protocol findings into the artifact the next scheduled real event will execute,
making the sweep cheaper and screen-complete. Neither manufactures work: both products
are consumed by already-scheduled future loops.

**Sources**: experiment-indices TSV (census substrate); exp-report-067 (pool definition,
σ̂ arithmetic); exp-reports 064/069/070 + goal-learnings § Protocol Findings (the screens
to consolidate); knowledge/README.md (entry placement).

**Estimated Effort**: low (zero GPU; ledger + artifact work).

**Risk Assessment**: No metric movement possible (no launch); verdict pre-registered
invalid/NaN per NO-LAUNCH precedent. Failure modes: census mistakes (mitigated by
citing each row's exp-report for eligibility) and protocol drift (mitigated by citing
each screen to its measured source). Escape hatch: if the census changes σ̂ such that any
recorded bound comes within reach of the bar, that bound promotes to the EXP-072 lead.

### 2. Full periodic lit sweep, early
**Summary**: Run the complete double-screened sweep now instead of ~2026-06-17.

**Reasoning**: Six days inside the policy interval; the last full sweep plus the
EXP-070 infra-scoped search leave expected yield ≈ 0. Priced empty by the governing
policy (same rejection as brainstorm-070 candidate 2).

**Sources**: goal-learnings idle-loop policy entry; brainstorm-070 § Candidate 2.

**Estimated Effort**: low-medium.

**Risk Assessment**: Near-certain empty; erodes the policy's discipline. Rejected on
timing.

### 3. Load-bearing anchor meta-audit
**Summary**: Adversarially re-verify the five most-cited closure anchors against their
primary reports.

**Reasoning**: The policy's own trigger condition is unmet — all anchors were re-read
clean during EXP-068/069/070's anchor-discipline checks within the past two days, and no
recorded bound sits within 0.18 of the bar, so even a moderate anchor revision flips no
verdict. Candidate 1's Entry A covers the one anchor (σ̂ pool) that has never had a
census-style check, which is the genuinely unaudited piece.

**Sources**: exp-logs 068/069/070 § Experimental Adjustments (anchors re-read clean).

**Estimated Effort**: low-medium.

**Risk Assessment**: Redundant with day-fresh checks; rejected until anchors are stale
or a near-bar bound appears (policy condition).

## Idea Evaluation

**Evidence strength**: Candidate 1 targets the two documented gaps the ledger itself
names: the σ̂ pool is cited by every closure bound yet never census-verified, and
exp-report-070 § Next Steps explicitly requires the sweep to apply a screen battery that
currently exists only as scattered entries. Candidates 2 and 3 are both rejected by the
standing policy's own trigger conditions (interval not elapsed; anchors not stale).

**Mechanism clarity**: Entry A is pure arithmetic over recorded rows with per-row
eligibility citations; Entry B is consolidation with per-screen citations. Both products
have named consumers (any future near-bar decision; the 2026-06-17 sweep).

**Expected impact**: None can move the metric this loop (measured ceiling; no launch).
Candidate 1 either confirms the decision arithmetic (hardening every standing bound) or
corrects it (escape hatch — the highest-value possible outcome of idle work).

**Risk profile / Feasibility**: Candidate 1 is zero-GPU, zero-code, fails safest (a
census discrepancy = found work, not a wasted run), trivially feasible.

## Chosen Idea
**Selected**: Pre-sweep consolidation loop: replicate-pool census + pre-registered sweep
protocol

**Why this idea**:
It is the policy-compliant smallest useful ledger work: it audits the single most
load-bearing unverified quantity in the closure ledger (the n=5 replicate pool behind
σ̂ = 0.123 and the +0.28 threshold) and produces the protocol artifact the scheduled
2026-06-17 sweep will execute. Both outputs are consumed by already-scheduled future
work — nothing is manufactured.

**Hypothesis**:
(a) The census confirms EXP-067's n=5 pool is exactly the set of eligible baseline-config
replicates in the TSV, leaving σ̂ = 0.123 and required effect ≥ +0.28 standing unchanged
(falsification branch: any pool change triggers a σ̂/threshold recompute and a re-check of
all recorded bound margins, with any newly-near-bar bound promoted to the EXP-072 lead);
(b) the pre-registered sweep protocol enumerates every measured screen with citations such
that the 2026-06-17 sweep can run without re-derivation. Verdict pre-registered
invalid/NaN (NO-LAUNCH lineage; no metric movement claimed).

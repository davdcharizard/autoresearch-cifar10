# Plan EXP-068: Radical-class inequality closures at zero charged cost (self-distillation, resolution-up, optimizer-family) + near-miss pool audit
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md

## Pre-registered protocol (EXP-063/EXP-064 lineage: inequality-gated NO LAUNCH)

- **Zero code diff**: train.py is NOT modified at any point. Verify `git diff --quiet -- train.py`.
- **Zero charged GPU seconds**: no `uv run train.py` launch occurs unless an inequality
  SURVIVES (escape hatch below). No GPU probe is needed — every bound composes already-measured
  ledger anchors; none depends on an unmeasured dt.
- **Recorded metric = NaN, verdict pre-registered = invalid** (EXP-063/064 precedent for
  inequality-gated NO LAUNCH experiments), UNLESS the escape hatch fires.
- **Escape hatch (falsification branch)**: if any class's worked bound lands ≥ bar 96.81
  under conservative (candidate-favoring) arithmetic, that class is NOT launched this loop;
  it is recorded as the surviving candidate and promoted to the EXP-069 brainstorm lead.
  The verdict for THIS loop remains invalid/NaN either way (no run, no metric).
- **Decision arithmetic (standing, EXP-067)**: baseline pooled mean 96.534, σ̂ = 0.123,
  bar 96.81 = mean + 2.24σ̂; a launchable candidate needs a worked bound that clears 96.81.
- **Anchor discipline**: every quantity in every inequality must cite a measured ledger
  source (EXP number + report/log path). No free parameters; where a quantity is uncertain,
  take the candidate-FAVORING end so a failed inequality is robust.

## Milestones

### Milestone 1: Setup and integrity
- [x] On branch autoresearch/exp-068 (cut from autoresearch/dev); `git diff --quiet -- train.py` passes (zero diff).
- [x] exp-log-068.md created from template with protocol header; Outcome pending.

### Milestone 2: Inequality A — sequential self-distillation (teacher→student inside 300s)
- [x] Work the bound with cited anchors: budget split T+S = 300s charged; per-phase epoch count from family dt 22.4ms (EXP-067 ledger); half-budget starvation toll from EXP-043 (measured −0.9 at ~70 ep) cross-checked against the EXP-002/007 starvation slope; KD credit set at the candidate-favoring published maximum (+1.0pp, fixed-epoch literature) and double-screened (absorption: no heavy-aug budget-matched KD evidence; cost-landing: KD's second forward prices ON the charged step — add its dt toll or note it as a further unmodeled cost in the candidate's favor being waived).
- [x] Also bound the asymmetric split (e.g. 100s/200s) to show the bound is split-robust: teacher quality degrades faster than student steps recover (EXP-002/007 slope), so no split clears.
- [x] Record the worked bound and branch decision (closed vs escape hatch) in exp-log § Run Log.

### Milestone 3: Inequality B — resolution-up training (bilinear 32→40px)
- [x] Work the bound with cited anchors: (i) information term — deterministic upsampling of 32px source adds ZERO information by inspection; (ii) cost term — FLOPs ×(40/32)² = ×1.56 is a measured-law dt floor (square-kernel dense path is FLOPs-priced, EXP-066; off-power-of-2 spatial tiers can only make it worse, EXP-044/045 analog) → ≥ ~35ms → ≤ ~90 epochs → starvation toll ≥ ~0.6 by the EXP-002/007 slope and the EXP-006 conversion law; (iii) mismatch term — Eval is PINNED at 32px (prepare.py read-only), so training BN statistics at 40px spatial distribution is a train/eval constants mismatch with measured negative sign (EXP-029 mechanism; FixRes-class alignment transients measured net-negative, EXP-025/065).
- [x] Record the worked bound and branch decision in exp-log § Run Log.

### Milestone 4: Inequality C — optimizer-family swap (AdamW / Lion / cautious-SGD)
- [x] Work the closure by screen + subsumption with cited anchors: absorption screen fails (no published ≥0.3pp gain over tuned SGD for ResNet/CIFAR under heavy-aug budget-matched comparison; published direction is adverse for adaptive methods on small CNNs); subsumption — EXP-028 measured the strongest 2024–25 CNN-speedrun optimizer (Muon, airbench-anchored) at baseline mean with a worse basin (test_loss 0.193 vs 0.185) and a dt toll; EXP-062 measured the schedule-free branch at −1.84; weaker family members (AdamW/Lion/cautious) inherit the closure a fortiori plus per-param state costs (VRAM/step overhead) Muon did not pay.
- [x] Record the closure and branch decision in exp-log § Run Log.

### Milestone 5: Near-miss combination-pool audit
- [x] Record the audit with citations: pool = {EXP-046/052 anti-aliased shortcut, pooled +0.11 at n=3, closed permanently as unresolvable; EXP-048 prefetch +87 steps, non-reproducing per EXP-053}; the only compound already measured (EXP-053: 96.445, no additivity); therefore no unmeasured near-miss pair exists. Directive prong "combine previous near-misses" discharged.

### Milestone 6: Verification and bookkeeping
- [x] Confirm zero charged seconds spent (no run.log exists; `ls run.log` fails) and zero diff maintained.
- [x] Evaluate the pre-registered branches (all closed vs escape hatch fired) and record in exp-log § Verification Results.
- [x] Outcome → completed.

## Code Changes
- **None.** train.py byte-identical to autoresearch/dev HEAD (1990397 recipe) throughout.
- **No /tmp scripts** — no probe, no composite launcher.

## Configuration Changes
- None.

## Execution Environment
- Method: documentation-only loop, local CPU text work; zero GPU use (no charged seconds,
  no probes). All "execution" is working four bounded arguments against ledger anchors and
  recording them in the exp-log.
- Resources: none beyond the repo and .autoresearch ledger.
- Estimated runtime: ~10–20 minutes of artifact work.
- Log output strategy: exp-log-068.md § Run Log carries one entry per inequality (A, B, C)
  plus the pool audit — each with the worked arithmetic, anchor citations, and branch
  decision. No run.log is ever created (and that absence is itself a Milestone-6 check).
- Tool skill: none.

## Abort Criteria
- Not applicable in the usual sense (no run to monitor). Loop-level abort: if any anchor
  citation turns out NOT to support the quantity used (e.g., the EXP-043 toll re-reads
  differently from its report), STOP using that inequality, re-derive with the corrected
  anchor, and record the correction in exp-log § Experimental Adjustments before deciding
  the branch. An inequality that cannot be closed with honest anchors fires the escape
  hatch — it must NOT be forced closed.

## Verification Protocol

### Verification Procedure
Follows goals/maximize-cifar10-test-accuracy.md § Procedure where applicable; this loop has
no run, so the conditions resolve as in the EXP-063/064 precedent:

1. **Condition 1 — best_test_acc ≥ bar 96.81** (baseline 96.71 via `exp-index.sh baseline`
   + 0.1): NOT SATISFIABLE — no run is launched, no metric exists. Recorded metric = NaN.
   Per EXP-063/064 precedent the verdict is **invalid** (inequality-gated NO LAUNCH; the
   experiment's product is the closure ledger, not a metric draw).
2. **Condition 2 — run completes ≤ 600s total**: vacuous (no run). Check instead: no
   run.log exists in the project root at loop end (`ls run.log` → no such file).
3. **Condition 3 — validation at most once per epoch**: vacuous (no run); structurally
   guaranteed by zero diff.
4. **Integrity checks specific to this loop**: (a) `git diff --quiet -- train.py` at loop
   end; (b) every inequality quantity carries an EXP-citation; (c) each of the four
   Run-Log entries states its branch decision explicitly (CLOSED or ESCAPE-HATCH).

### Informational Metrics (Optional)
- The three worked bounds (numeric, vs bar 96.81) — the experiment's actual product.
- Escape-hatch status per class (expected: none fired).
- Charged GPU seconds spent (expected: 0) and probe seconds (expected: 0).

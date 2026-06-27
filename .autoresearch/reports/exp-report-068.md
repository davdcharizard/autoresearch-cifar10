# Report EXP-068: Radical-class inequality closures at zero charged cost + near-miss pool audit
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md
- **Plan**: plans/plan-068.md
- **Log**: logs/exp-log-068.md

## Goal
Maximize best_test_acc (%) within the fixed 300s charged budget. Baseline 96.71 @ 1990397;
bar 96.81 (= pooled mean 96.534 + 2.24σ̂ per EXP-067). This loop was a CLOSURE experiment in
the EXP-063/064 lineage: with the measured-ceiling audit complete and the steady-state
protocol in force, the prescribed move for a no-surviving-candidate loop is zero-charged-cost
closures over re-measuring closed interiors.

## Idea & Hypothesis
Close the three remaining "radical" program-level classes — the ones a fresh reader would
flag as untried and the autopilot think-harder directive points at — by pre-run inequalities
composed entirely of measured ledger anchors, plus a citable exhaustion record for the
near-miss combination pool. Hypothesis: all three bounds land strictly below the bar under
candidate-FAVORING arithmetic, so no launch occurs; any surviving bound instead fires an
escape hatch promoting that class to the EXP-069 lead.

## Approach
Zero code diff (verified twice), zero GPU seconds (no run, no probe — no bound depended on an
unmeasured dt), no /tmp scripts. Four worked entries in the exp-log: (A) sequential
self-distillation, (B) resolution-up training, (C) optimizer-family swap, (D) near-miss pool
audit. Verdict pre-registered invalid/NaN per the EXP-063/064 NO-LAUNCH precedent. This loop
also completed the in-scope-file audit (pyproject.toml, README.md, TASK.md re-read: no
unexploited resources) recorded in brainstorm-068.

## Execution
No runs, no retries, no errors. All four closures worked first-pass with anchors re-read as
cited; the plan's abort criterion (an anchor failing to support its quantity) never fired.

## Results
- **Primary metric**: NaN (no launch; baseline: 96.71, delta: N/A)
- **Observations**:
  - **Inequality A (self-distillation)**: any T+S split of 300s prices both phases on the
    measured starvation ladder (EXP-043: ~70-epoch members ≈ 95.6; EXP-002/005/007 slope);
    granting the published-MAXIMUM fixed-epoch KD credit (+1.0pp) and waiving the teacher's
    charged-step inference toll, the best split bounds at ≈ 96.6 < 96.81. CLOSED.
  - **Inequality B (resolution-up)**: gain ≤ 0 by inspection — bilinear 32→40px adds zero
    information and Eval is pinned at 32px (prepare.py), making 40px BN statistics a
    train/eval constants mismatch with measured negative sign (EXP-029); the FLOPs-floor
    cost term (×1.5625, citable via EXP-066's square-kernel FLOPs-pricing datum at 1.59×)
    only deepens the failure (≥ ~0.6 starvation toll). CLOSED without needing the cost term.
  - **Inequality C (optimizer family)**: absorption screen fails (no heavy-aug
    budget-matched evidence; published direction adverse) and the family's measured BEST
    member already read ≤ mean (Muon, EXP-028: plateau at mean in a worse basin; EXP-062
    schedule-free −1.84; EXP-023/024 noise bracket); AdamW/Lion/cautious inherit a fortiori
    plus pointwise-pass tolls and a forced retune off the certified optimum. CLOSED.
  - **Pool audit (D)**: the only positive-reading components in 69 rows are EXP-046/052
    (+0.11, closed permanently) and EXP-048 (non-reproducing); their compound is already
    measured (EXP-053, no additivity). Pool EXHAUSTED — the "combine near-misses" directive
    prong is discharged by enumeration.
- **Analysis**: The measured-ceiling reading strengthens: the three classes most plausibly
  nominated as "untried radical moves" are now cited closures rather than open questions,
  and none came within 0.2 of the bar even with every uncertain quantity resolved in the
  candidate's favor. The loop also demonstrates the steady-state protocol's economics: three
  family-level closures + one audit for zero GPU seconds, versus ~8 minutes per dominated
  run had they been measured directly. EXP-066's probe datum (square kernels FLOPs-priced at
  1.59×) turned out to be exactly the citation Inequality B needed — probe data compounds.
- **Key Learning**: All three remaining radical classes (sequential KD, resolution-up,
  optimizer-family) close by measured-anchor arithmetic alone; the closure ledger now covers
  every class a fresh reader would nominate, at zero charged cost.

## Verification
- **Conditions**: Condition 1 not satisfiable (no run; metric NaN) — verdict invalid as
  pre-registered; Conditions 2–3 vacuous with substitute checks PASS (no run.log; zero
  diff). Loop-specific integrity PASS: zero diff maintained, every quantity EXP-cited, all
  four entries carry explicit branch decisions (4× CLOSED, 0× ESCAPE-HATCH).
- **Review Notes**: trustworthy — nothing was measured, so the only integrity surface is
  citation fidelity, and every anchor was re-read from its report/log this session.
- **Verdict**: invalid
- **Verdict Basis**: inequality-gated NO LAUNCH (EXP-063/064 precedent) — no metric exists
  by design; the experiment's product is three class closures and a pool-exhaustion record.

## Unexplored Avenues
- Within-class variants are closed by the same bounds, not just the representatives:
  KD with online/byte-shared teachers re-lands on ensemble dilution (EXP-043) or charged-step
  tolls; mixed-resolution batches inherit B's mismatch term plus EXP-059's dynamic-shape tax;
  optimizer hybrids (SGD+adaptive head) re-enter the EXP-057/058 bracketed corner.
- A measured falsification of any bound would require spending exactly the dominated run the
  inequality exists to avoid — only worth it if a future publication supplies heavy-aug
  budget-matched evidence contradicting an anchor.

## Next Steps
1. Steady-state continues: next loop should run the periodic double-screened lit sweep ONLY
   if a publication-scale interval has passed (last sweep 2026-06-10/11 in brainstorm-066);
   otherwise prefer another zero-charged-cost closure or instrument improvement. Confidence high.
2. The closure ledger is now arguably complete at the class level — a candidate inflow can
   only come from (a) new external publications passing both screens, or (b) a revision of a
   standing anchor. Treat anchor-revision claims with the same pre-run-inequality discipline.
   Confidence high.
3. If no zero-cost closure target remains next loop, the highest-information GPU spend
   reverts to instrument work (e.g., n=7 σ pooling) only when a near-bar decision actually
   pends — otherwise document-level work (ledger consolidation) is the honest move.
   Confidence medium.

## Exit Action Results

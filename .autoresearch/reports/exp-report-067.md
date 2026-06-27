# Report EXP-067: σ-tightening baseline replicate pair (zero-diff, n=3 → n=5)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md
- **Plan**: plans/plan-067.md
- **Log**: logs/exp-log-067.md

## Goal
Maximize best_test_acc (%) within the fixed 300s charged budget. Baseline 96.71 @ 1990397;
bar 96.81. This loop was an INSTRUMENT experiment: with the ceiling audit complete (EXP-066)
and no positive-prior construction left, the highest-information spend was the decision
instrument — run-level σ̂ rested on n=3 draws while every verdict band derives from it.

## Idea & Hypothesis
Two byte-identical baseline runs (EXP-027 protocol), metric = mean(R1, R2), verdict
pre-registered no-improvement (a zero-diff draw cannot be an improvement; harvesting a lucky
one would be variance mining). Hypothesis: both draws in mean ± 2σ = [96.25, 96.89] with family
signatures; pooled n=5 σ̂ ∈ [0.10, 0.22]; out-of-band draws would instead detect drift.

## Approach
Zero code diff (verified `git diff --quiet -- train.py`). Composite launcher reused from the
exp061 template; two sequential gated runs with the standard watchdog. No deviations.

## Execution
Both runs pristine on the first attempt: gates cleared immediately, D0 22.5 / 22.7ms, zero
slow streaks, rc=0, totals 492.5s / 486.3s. No retries, no errors.

## Results
- **Primary metric**: 96.485 = mean(96.53, 96.44) (baseline: 96.71, delta: −0.225, −0.23%)
- **Observations**:
  - R1 = 96.53, R2 = 96.44 — both inside the pre-registered band [96.25, 96.89], both with
    exact family signatures (13,455 / 13,461 steps — the 046-family step band reproduces to
    0.05%; 139 epochs both; test_loss 0.1867 / 0.1892).
  - **Pooled n=5 statistics (the product)**: {96.71, 96.59, 96.40, 96.53, 96.44} → mean
    96.534, sample σ̂ = 0.123 (vs ~0.16 at n=3). Hypothesis confirmed on every branch.
  - Derived decision quantities: mean ± 2σ̂ = [96.29, 96.78]; the standing bar 96.81 sits at
    mean + 2.24σ̂ (more selective than the prior +1.5σ reading); a future candidate needs a
    true effect ≥ +0.28 ≈ 2.2σ̂ to clear the bar on a single draw.
- **Analysis**: No environment drift months into the program — the LEVEL ledger now matches
  the signature ledger (EXP-066 re-validated dt/steps; this pair re-validates the plateau
  distribution itself). The recorded 96.71 baseline is re-confirmed as the distribution top:
  it remains the maximum of all five draws (+1.4σ̂ above the pooled mean). The slightly tighter
  σ̂ mildly RAISES the effective bar in σ units, further supporting the measured-ceiling
  reading: of 50 intervention experiments since EXP-006, none produced a draw above 96.84,
  and the honest replicate protocol has never been threatened.
- **Key Learning**: The baseline run distribution is stationary (no drift) and tighter than
  previously estimated — σ̂ = 0.123 at n=5 — which makes the standing bar a mean + 2.2σ̂ event
  and sharpens every future verdict at zero ongoing cost.

## Verification
- **Conditions**: Integrity PASS both runs (poolable); Condition 1 FAIL as pre-registered
  (96.485 < 96.81); Conditions 2–3 informationally pass.
- **Review Notes**: results confirmed trustworthy — pristine telemetry, exact family ledgers,
  pre-registered protocol followed exactly.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure by design — zero-diff replicate pair; the experiment's
  value is the instrument update, honestly recorded at the pair mean.

## Unexplored Avenues
- Further σ pooling (n=7+): diminishing — σ-of-σ̂ at n=5 is already adequate for ±0.3-class
  screens; only worth it if a future near-bar decision actually hinges on the third decimal.
- Per-epoch plateau-shape statistics (eval-trace variance models): would refine the
  max-statistic understanding but changes no decision rule. Not worth a loop.

## Next Steps
1. Steady-state protocol for the measured-ceiling regime: future loops should spend GPU time
   only on (a) candidates passing BOTH standing screens plus the pre-run inequality, probed
   before launch, or (b) periodic (publication-interval, not per-loop) double-screened lit
   sweeps for new 2026 techniques. Confidence high that this is the correct economics.
2. Update the σ entry in goal-learnings with the n=5 numbers (done this loop) and use
   mean 96.534 / σ̂ 0.123 / bar = mean+2.2σ̂ as the standing decision arithmetic. Confidence high.
3. If a future loop has no surviving candidate, prefer zero-charged-cost closures (probe-gated
   inequalities, by-inspection audits) over re-measuring closed interiors. Confidence high.

## Exit Action Results

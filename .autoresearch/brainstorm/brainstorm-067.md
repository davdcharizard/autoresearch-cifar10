# Brainstorm EXP-067
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external sources this loop. The double-screened sweep (absorption screen: heavy-aug
  budget-matched evidence; cost-landing screen: price off the charged step) ran ~24h ago in
  brainstorm-066 and surfaced nothing surviving both screens (airbench element map: all
  closed/banned; augmentation multiplicity → noise law; budgeted importance sampling →
  published null supporting the ceiling). Re-sweeping one loop later has no expected yield;
  next periodic sweep should wait for new 2026 publications (exp-report-066 Next Steps #2,
  confidence low).

## Experimental History Review

- 68 experiments. Baseline 96.71 @ 1990397 (distribution top); family mean ≈ 96.57, σ ≈ 0.16
  (n=3 baseline-config draws: 96.71, 96.59, 96.40 — EXP-027); bar 96.81 = mean + 1.5σ. Last
  improvement EXP-006. External transfer 0-for-21.
- **State after EXP-066**: the adversarial ceiling audit is COMPLETE. Recipe constants
  (bracketed, incl. Normalize-std closed by inspection), structural classes (incl. the final
  kernel-size corner — 5x5 stem measured −2.7σ at zero cost), funding currencies, pressure
  profiles, schedule/noise/loss/init/optimizer axes: all measured-closed or law-closed. The
  measured-ceiling hypothesis survived its designated falsification attempt; the record
  contains NO untested construction with measured evidence in its favor.
- Decision-instrument state: run-level σ̂ for the BASELINE config rests on n=3 (EXP-027's pair
  plus the standing 96.71); the broader family band pools ~15 mean-band nulls but across
  DIFFERENT configs. EXP-052 validated the replicate-pair MEAN protocol (single near-bar draws
  are ~weekly-expected under H0). The family signature ledger (dt, steps, epochs) was
  re-validated incidentally by EXP-066's pristine run.
- Untried gaps: none with positive priors. Remaining law-closed residuals (GhostBN, gradient
  clipping, TA dose-down, warmup shape) are interpolations against bracketed axes.

## Candidate Ideas

### 1. σ-tightening baseline replicate pair (n=3 → n=5, drift re-anchor)
**Summary**: Two byte-identical runs of the unmodified baseline train.py (zero diff), behind
the standard composite gates/watchdog, pre-registered as no-improvement with metric =
mean(R1, R2) — the EXP-027 protocol exactly. Pool the two fresh draws with the existing n=3 to
re-estimate run-level σ at n=5, and check for plateau drift (driver/library/thermal) months
into the program: both draws should land in mean ± 2σ = [96.25, 96.89] with family signatures.

**Reasoning**: At a measured ceiling, the loop's remaining lever on FUTURE decisions is
decision quality. Every future candidate verdict leans on σ ≈ 0.16 measured once (EXP-027,
n=2 fresh draws). n=5 roughly halves the σ-of-σ̂, hardens the bar arithmetic (mean + 1.5σ), and
a drift check protects the entire ledger against silent environment shift — EXP-066's family-
exact signatures suggest no drift, but its LEVEL was an intervention read, not a baseline draw.
Ranked first in exp-report-066 Next Steps (medium confidence).

**Sources**: EXP-027 (protocol + prior σ), EXP-052 (pair-mean decision protocol),
goal-learnings § Protocol Findings (σ entry), exp-report-066 § Next Steps.

**Estimated Effort**: low (zero code change; 2 × ~8 min wall; composite template reuse).

**Risk Assessment**: No improvement possible by design (pre-registered). Worst case: a draw
lands outside [96.25, 96.89] — that is itself the most valuable outcome (drift or σ
underestimate detected → every standing band gets revised). Contamination risk handled by
gates/watchdog/step-ledger as usual.

### 2. Fresh literature re-sweep under the double screen
**Summary**: Re-run the screened lit sweep for 2025–2026 heavy-aug budget-matched techniques.

**Reasoning**: Periodic re-checks are the only inflow channel for new candidates.

**Sources**: brainstorm-066 § Web Search (the sweep that returned empty 24h ago).

**Estimated Effort**: low.

**Risk Assessment**: Near-certain empty yield one day after the last sweep; burns a loop
producing no TSV-recordable measurement. Rejected on timing — revisit after a publication-scale
interval.

### 3. Law-closed residual probe (GhostBN / gradient clipping / TA-Wide→TA dose)
**Summary**: Measure one of the remaining never-directly-dosed residuals.

**Reasoning**: Only as map-completion; all three sit ON bracketed axes rather than at unpriced
corners: GhostBN = BN-stat noise on the peaked pressure axis + load-bearing BN constants
(EXP-029/038/039); clipping = stability headroom the recipe cannot spend (EXP-018 lesson, heat
quality-certified EXP-010/049); TA dose-down = pressure-down (EXP-015/033 monotone side).

**Sources**: goal-learnings § Failed Approaches (cited closures).

**Estimated Effort**: low-medium.

**Risk Assessment**: These are interpolations of measured-closed brackets — the EXP-049-class
"do not probe a flat/peaked optimum's interior" rule applies. A run here re-measures known
laws; strictly less informative than Candidate 1. Rejected.

## Idea Evaluation

**Evidence strength**: Candidate 1 follows two validated protocols (EXP-027 replicate
methodology, EXP-052 pair-mean decisions) and addresses a quantified weakness (σ̂ at n=3
governs every verdict band in the program). Candidate 2's evidence says it will return empty
(same sweep, 24h stale). Candidate 3 contradicts standing closure logic — bracketing closes
interiors.

**Mechanism clarity**: Candidate 1's mechanism is exact and arithmetic: pooled n=5 tightens
σ-of-σ̂ by ~√2 and either re-certifies or revises the [96.25, 96.89] decision band and the
mean + 1.5σ bar; the drift check re-anchors the LEVEL ledger the way EXP-066 re-anchored the
signature ledger. Candidates 2–3 have no metric-relevant mechanism this loop.

**Expected impact**: None of the three can clear the bar (measured ceiling). Candidate 1
maximizes the value of every FUTURE read; it is the only candidate whose output compounds.

**Risk profile**: Candidate 1 fails safest — even its "failure" branch (out-of-band draw) is a
high-value detection. **Feasibility**: trivial.

## Chosen Idea
**Selected**: σ-tightening baseline replicate pair (n=3 → n=5, drift re-anchor)

**Why this idea**:
With the ceiling audit complete and no positive-prior construction left, the highest-information
spend is the decision instrument itself: σ̂ rests on n=3 while every verdict band, the bar
(mean + 1.5σ), and the effect-size screen (≥ +0.3) derive from it. Two pristine baseline draws
re-estimate σ at n=5, test for environment drift at the LEVEL (not just signature) layer, and
follow protocols already validated twice in this program. Ranked first by exp-report-066.

**Hypothesis**:
Both replicate draws land inside mean ± 2σ = [96.25, 96.89] with family signatures (dt
22.0–22.8ms, 137–140 epochs, steps 13,100–13,600, params 4,286,026), the pooled n=5 σ̂ stays in
[0.10, 0.22] (no drift, bands re-certified), and the pre-registered verdict is no-improvement
with metric = mean(R1, R2). A draw outside the band falsifies band stability and triggers a
ledger revision instead.

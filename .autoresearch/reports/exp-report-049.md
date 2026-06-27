# Report EXP-049: PEAK_LR 0.4 → 0.3 — the heat-down bracket
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Log**: logs/exp-log-049.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: **96.71** @ 1990397; bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis

After EXP-048 closed the throughput seam, the single genuinely unmeasured bracket on any recipe constant was integrated LR heat DOWNWARD: the heat axis was bracketed only from above (peak 0.6 → −0.57 in EXP-010; warmup 0.08 → −0.22 in EXP-014; linear anneal → −0.50 in EXP-016), and exp-report-010 § Unexplored Avenues had explicitly queued "PEAK_LR 0.3 (downward probe)" 38 loops earlier without it ever running. Chosen idea: a one-line change PEAK_LR 0.4 → 0.3 — a pure 0.75× integrated-heat probe at byte-identical signatures (EXP-010 demonstrated pure-LR isolation). Hypothesis: best ≥ 96.81 if 0.4 sits above the optimum (the EXP-000 linear scaling predates 4x widening + heavy aug); pre-registered branches (ii) mean band 96.42–96.72 → heat axis closed flat-below, (iii) < 96.42 → shallow-side loss with 0.4 at an interior optimum.

## Approach

Exactly one constant changed in train.py L23 (`PEAK_LR = 0.4` → `0.3`, comment updated); diff verified 1 file / ±1 line. Static checks: AST parse, extracted constants, lr_at spot checks (peak 0.3 at warmup end, warmup midpoint 0.15, cosine midpoint 0.15, anneal → ~0). No separate CPU sanity script — zero new code surface (recorded in exp-log Decisions). Launch via /tmp/exp046_composite.sh verbatim (dual gates, D0 gate 26ms, watchdog). No plan deviations.

## Execution

Single pristine run: gates clear at poll 1 (apps=0, load=5); GATE_DECISION D0=22.2ms, projected 139 epochs; 31 post-gate windows all 21.7–22.8ms, slow_streak never above 0; RC=0; 139 epochs / 13,456 steps / 300.0s charged / 487.7s total. No retries, no kills, no errors.

## Results

- **Primary metric**: best_test_acc 96.52 (baseline: 96.71, delta: −0.19, −0.20%)
- **Observations**: 96.52 = mean − 0.3σ. The plateau is converged-FLAT over the last 8 evals (96.25–96.52) at family test_loss (0.1855–0.1882, final 0.1882) — explicitly NOT the still-climbing heat-starved signature that would have indicated branch (iii)'s mechanism. Mid-schedule evals tracked slightly below family (ep7 63.06 vs ~65) and the anneal closed the gap, exactly the EXP-010-inverse shape predicted. Step ledger: 13,456 ∈ [13,428, 13,515] — pure-LR isolation confirmed to the step.
- **Analysis**: Pre-registered branch (ii) — a mean-band null. 0.75× heat converges to the same basin quality as 1.0× heat: the LR optimum is flat over at least [0.3, 0.4] and falls off by 0.6 (−0.57, EXP-010). The asymmetric-optimum reading: 0.4 sits on a measured plateau whose down-side extends at least to 0.3 and whose up-side breaks between 0.4 and 0.6. The interesting micro-signal — the run lost ~2pp mid-schedule and fully repaid it through the anneal, landing at family test_loss — confirms the one-cycle anneal, not the peak, is what sets final basin quality in this regime (consistent with EXP-016's tail-shape sensitivity). With this read, EVERY recipe constant is now certified bracketed-both-directions or measured-flat: the recipe-constant audit that 43 loops of negatives have been building is complete. No knob on the existing recipe has unmeasured headroom.
- **Key Learning**: PEAK_LR 0.3 (0.75× heat) converged flat at mean−0.3σ with family test_loss and step count — the LR optimum is flat over [0.3, 0.4]; heat axis closed both directions; the recipe-constant audit is complete.

## Verification

- **Conditions**: Integrity pre-condition PASSED (windows 21.7–22.8ms; 139 epochs; 13,456 steps in family ledger; params 4,286,026 exact; 300.0s charged; 139 evals ≤ 139 epochs; trajectory-criterion numerics normal). Condition 1 (best ≥ 96.81) FAILED: 96.52, also below the replicate band [96.70, 96.80] so no replicate pair triggered. Conditions 2–3 skipped per first-failure-stop (informationally both pass: 487.7s ≤ 600; 139 ≤ 139).
- **Review Notes**: Results trustworthy — pristine profile, family signatures to the step, plateau at family test_loss; the change is one constant with no path to charging or eval semantics. No suspicion of false failure: the read is mid-band, not anomalous.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid mean-band result below the bar (pre-registered branch ii).

## Unexplored Avenues

- **PEAK_LR 0.5 (the remaining gap in the up-bracket)**: the up-side breaks somewhere in (0.4, 0.6); 0.5 would localize the cliff. But both neighbors are measured (0.4 mean-level, 0.6 −0.57) and the down-side is now flat — there is no mechanism by which 0.5 beats a flat-optimum 0.4; only cartographic value. Not worth a run.
- **WARMUP_FRAC 0.25 (brainstorm-049 runner-up)**: same heat-down intent, confounded with anneal time-support; dominated by this experiment's clean answer. Closed by implication — the heat variable itself is now measured flat-below.
- **LR-floor tail**: documented rejection (scatter farming = reward hacking, brainstorm-049 Idea 3). Never propose.

## Next Steps

1. **The audit is complete — face the compound/radical frontier honestly** (high confidence): every recipe constant is bracketed or flat, every structural class measured-closed, throughput exhausted. Per the standing directive, the next brainstorm must construct candidates from the only unfalsified regions: compound interventions of individually-certified components (weighting the EXP-009 precedent that stacking certified regularizers lost), or a qualitatively different function class / training signal that passes ALL measured laws simultaneously.
2. **State the detection arithmetic in every brainstorm** (high confidence): bar = mean + 1.5σ; a single-draw improvement claim requires a true effect ≥ +0.3 — candidate mechanisms should be sized against this before spending a run.
3. **Protocol stack unchanged** (high confidence): composite gates, step ledger, trajectory-criterion numerics all resolved this run exactly per pre-registration — seven consecutive pristine runs.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->

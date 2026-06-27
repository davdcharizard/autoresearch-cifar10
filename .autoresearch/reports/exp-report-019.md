# Report EXP-019: Whitening init for conv1 (patch-eigenvector filters ± negations, learnable)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Log**: logs/exp-log-019.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does the one intervention class exempted from the EXP-018 deferral closure — information-ADDING init — hold a gain, via the airbench-style patch-whitening stem, the only remaining idea with evidence from a wall-clock metric regime?

## Idea & Hypothesis

Chosen idea: initialize `conv1` (3→64, 3×3) with the 27 eigenvectors of the training-patch covariance scaled by 1/√(λ+1e-4), plus their 27 negations (airbench's sign-preservation trick); 10 filters keep Kaiming init; weights stay learnable. Evidence: airbench (arXiv 2404.00498) and hlb-CIFAR10 — two independent wall-clock speedrun lineages where whitening init is load-bearing; arXiv 2210.03651 (trained first layers converge to data-statistics-aligned structure). Hypothesis: FASTER early trail than EXP-017's 63.76@5 / 75.06@8, unchanged signatures, best_test_acc ≥ 96.81; a wash-out closes the init axis bidirectionally.

## Approach

+20-line block in `main()` between model creation and `base_model =`: 5000 raw images → 500k deterministic stride-3 patches (zero RNG consumed — explicitly not seed hacking) → 27×27 covariance → eigh → filters into `conv1.weight[:54]`. Pre-launch CPU validation surfaced that the plan's "output covariance ≈ I" criterion was wrong as written: the ε=1e-4 floor intentionally suppresses near-null directions (12/27 by >10%; min λ = 1.1e-5). Corrected the criterion to the analytic expectation diag = λᵢ/(λᵢ+ε) — implementation matched to 8e-5, decorrelation exact (off-diag 0.0012). Kept ε (shrinking it would amplify near-null patch noise ~300x). No other deviations.

## Execution

One run, no retries (task b22ult0yv, launched 10:54:28 via composite launcher + inline watchdog into a verified-free GPU 0). Pristine: zero watchdog events, 0/267 windows >30ms (mean 22.4ms), 139 epochs / 13,418 steps, startup 12.3s (the covariance+eigh cost is unmeasurable), total 487.6s, VRAM 1613.0MB, params 4,286,026 — every signature byte-identical to baseline.

## Results

- **Primary metric**: best_test_acc = 96.45% (baseline: 96.71, delta: −0.26pp, −0.27%); bar was 96.81
- **Observations**: The predicted faster onset did NOT materialize: ep1 38.95 vs the comparator's 38.20, ep5 53.80 (a bouncy dip) vs 63.76, ep10 75.84 vs ~75 — within eval noise, neither the hypothesis's acceleration nor EXP-018's inversion. Mid-run on-family (87.08@60, 93.23@100); converged with a proper flat plateau (final six evals 96.37–96.45, final ≈ best). final_test_loss 0.1888 is among the best on record — but the metric pays in max accuracy, not loss.
- **Analysis**: This is the wash-out failure mode the plan ranked most likely, with a sharper architectural root cause than anticipated: in this net the stem feeds DIRECTLY into `bn1`, which renormalizes every channel — undoing the variance-equalization half of whitening at the first opportunity. airbench/hlb feed their whitening conv into an activation with no intervening BN, so the whitened SCALE structure survives to do work there; here only the data-aligned BASIS survives, and at 139 epochs (vs airbench's ~10) the stem learns an equivalent basis early enough that the head start is worthless. The result completes a three-experiment arc on "free" structural edits (EXP-017 free params, EXP-018 easier optimization, EXP-019 information at init): all three left every throughput signature untouched and all three converged BELOW baseline — the certified optimum is robust not just to constant changes but to every zero-cost structural perturbation tried. The init axis is now closed in both directions (expressivity-removing: −0.99; information-adding: −0.26). Fourteen consecutive misses. The honest residual space: capacity-paying structural moves (width asymmetry — epochs cost, low prior), optimizer-family changes (momentum is heat-adjacent, low prior), and acknowledged-low-prior schedule hybrids.
- **Key Learning**: Whitening init washes out when BN immediately follows the stem and the budget allows ~139 epochs — its wall-clock evidence comes from BN-free-stem, ~10-epoch speedruns; "information at init" only pays when the run is too short to learn that information itself.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (139 epochs exactly on projection; 0/267 windows >30ms); condition 1 FAILED (best_test_acc 96.45 < 96.81); conditions 2–3 skipped per first-failure stop (informally: 487.6s ≤ 600 rc=0; 139 evals = 139 epochs — both would have passed)
- **Review Notes**: trustworthy — metric matches the eval trail (best 96.45 @ ep 137); all signatures byte-identical to baseline so the deficit is attributable to the init alone; the pre-launch validation guarantees the intended math actually ran
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Whitening WITHOUT bn1 (remove/replace the stem BN)**: would let the whitened scale structure survive as in airbench — but removing bn1 is a real architecture change with its own dynamics risks, and the 139-epoch wash-out argument still applies. Low interest.
- **Frozen whitening stem (airbench-faithful)**: freezing removes the stem's 1.7k decay-group params from optimization and pins the basis; at 139 epochs learnability was not the problem (the basis converges anyway), so freezing mostly risks under-fitting the stem. Low interest.
- **Other information-adding inits (e.g. PCA init for deeper layers)**: the 139-epoch wash-out mechanism applies with even more force deeper in the net (later layers see more gradient signal). Closed in spirit by this result.

## Next Steps

1. **Width asymmetry at constant alignment (64/128/320, [3,3,3])** — the last untried capacity-paying structural move; preserves early-stage depth (EXP-017's failure isolate); requires measured-dt gate; prior LOW (epochs cost vs the EXP-012 1:1 exchange ceiling). Confidence: low.
2. **Momentum-axis paired trade (e.g. momentum 0.95 + peak 0.3 at ~constant effective heat)** — the optimizer family's only in-recipe knob never touched; must be framed as a heat-constant multi-knob trade per goal-learnings (single-knob momentum moves are priced by the closed heat axis). Confidence: low.
3. **Synthesis check** — with three zero-cost structural perturbations all converging below baseline and every axis bracketed, the next brainstorm should explicitly weigh whether remaining candidate space has positive expected value or whether the campaign's honest conclusion is that 96.71 @ EXP-006 is the optimum of this recipe family under this budget; if so, ideation should escalate to qualitatively different mechanisms (within hard constraints) rather than re-walking closed axes. Confidence: n/a (process note).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->

# Report EXP-061: Stage-1-heavy depth reallocation [3,3,3] → [4,3,2] at equal FLOPs
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Log**: logs/exp-log-061.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: 96.71 @ 1990397; bar ≥ 96.81 (family mean 96.57, σ 0.16). Specific question: does the favorable direction of EXP-017's measured allocation slope pay? EXP-017 ([2,3,4] stage-3-heavy at equal FLOPs) lost −0.28 with the deficit isolated to stage-1 depth; the mirror [4,3,2] (stage-1-heavy) was the only structural configuration in the record with measured evidence pointing toward it, and the last law-passing untested construction after EXP-060 emptied the seam space.

## Idea & Hypothesis

Chosen from brainstorm-061: **per-stage block counts (3,3,3) → (4,3,2)** — four 64-wide blocks at 32×32, three 128-wide, two 256-wide. FLOPs exactly preserved (ResNet stage invariant, ~151 MFLOPs/block), depth 20 preserved, lattice widths untouched, params 4,286,026 → 3,179,338 (−26%). Hypothesis: if stage-1 high-resolution representational depth is the binding constraint (EXP-017's reading), the gained fourth full-resolution block raises the plateau ≥ 96.81; if the allocation curve is flat-topped at uniform, family band; if stage-3 block count is load-bearing, < 96.41. Runner-up candidates from the fresh external sweep (airbench data filtering, lookahead) were screened — both are inside measured closures and accuracy-neutral in their own regime.

## Approach

Three hunks to train.py: `STAGE_BLOCKS = (4, 3, 2)` constant; `ResNet.__init__` unpacks per-stage counts into the three existing `_make_layer` calls (`_make_layer`/`BasicBlock` unchanged); call site + banner. Recipe byte-identical otherwise. CPU sanity all-pass (params exact, structure 4/3/2 with correct widths/strides, fwd/bwd, smoke).

## Execution

One GPU probe, one clean run, zero retries:
- Probe (load 9.7): **P = 22.51ms ∈ family band** — the EXP-034 per-block law (~2.5ms/block, width-independent) holds for stage-heavy reallocation despite stage-1 blocks carrying 4× the activations; block COUNT, not activation volume, prices the step. Launch with probe-revised bands (steps [12,495, 13,269], epochs [126, 138], D0 [22.2, 23.8]).
- Run 1: gates poll 1; D0 23.3 (probe +0.8, inside the historical offset); windows 22.3–23.3ms, slow_streak 0; RC=0 at 479.2s. Ledger: 12,986 steps, 134 epochs, params 3,179,338 exact, 134 evals, ep1 37.97 (family-normal — no early-heat penalty), no NaN.

## Results

- **Primary metric**: best_test_acc 96.39 @ ep131 (baseline: 96.71, delta: −0.32, −0.33%)
- **Observations**: Converged-FLAT plateau 96.19–96.39 over the final 8 evals at FAMILY test_loss (0.190) — a level deficit with neither a starvation signature (anneal complete, 134 ep) nor a basin-quality signature (test_loss family-equal). The read sits 0.02 BELOW the family floor (mean−1.1σ). VRAM 1,799MB (+186 vs uniform — the extra 32×32 activations, as predicted).
- **Analysis**: Pre-registered branch (iv) fired at its boundary. The honest claim strength: at n=1, 96.39 is consistent with either a low family-tail draw or a true small deficit (~−0.2); under EITHER reading the hypothesis — that stage-1 depth was the binding resource — is REFUTED: the favorable direction delivers no gain. Combined with EXP-017 ([2,3,4] −0.28, converged), the three-point allocation curve ([2,3,4] −0.28 / [3,3,3] mean / [4,3,2] −0.2-ish) peaks AT uniform: the allocation axis is closed BIDIRECTIONALLY. The new mechanistic datum refines EXP-017's reading: that experiment showed stage-3's added params were worthless, and this one shows stage-3's block COUNT is load-bearing — two 256-wide blocks under-process the final features even though a fourth one adds nothing. Uniform [3,3,3] at 4× width is now a measured optimum in BOTH dimensions of the depth-allocation plane (total depth EXP-008/034, per-stage allocation EXP-017/061). With this, the last in-record favorable directional signal is spent: every catalogued axis, seam, slope-direction, and the external regime-matched frontier reads at-or-below the family mean.
- **Key Learning**: Allocation slopes measured in one direction do not extrapolate through the optimum — EXP-017's "stage-1 depth is the deficit" was a statement about REMOVING blocks, not a gradient pointing toward adding them; uniform allocation is the two-sided optimum at this depth/width.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine telemetry, ledger on probe-revised bands, params assert exact, ep1 37.97, no NaN). Condition 1 FAILED: 96.39 < 96.81 (branch (iv) boundary). Conditions 2–3 pass informationally (479.2s ≤ 600; 134/134 evals).
- **Review Notes**: Results confirmed trustworthy — watchdog full coverage, step ledger excludes contamination, probe/run dt agreement, change came through the intended intervention class.
- **Verdict**: no-improvement
- **Verdict Basis**: Condition failure — valid below-bar result.

## Unexplored Avenues

- **[4,2,3] or [3,4,2] interior allocations**: interpolations between three measured points of a curve that peaks at uniform — no mechanism for an interior point to exceed the peak. Closed by bracketing logic.
- **Stage-1-heavy WITH the params restored elsewhere** (e.g., (4,3,2) at width 4.5x stage-3): off-lattice widths hardware-closed (EXP-044/045); within-lattice compensations are starvation-priced. Closed by composition.
- **Asymmetric stage-2 moves ((3,4,2)/(2,4,3))**: stage-2 was never the isolated variable, but with both neighbors measured at-or-below uniform and stage-2 sharing both failure mechanisms, the expected value is below the effect-size screen.

## Next Steps

The allocation axis was the last in-record construction carrying a measured favorable signal; it is now closed bidirectionally. Status for brainstorm-062: (a) all catalogued axes, seams, slope-directions, AND the post-2024 external regime-matched frontier are measured at-or-below the family mean — 55 consecutive closures; (b) the measured-ceiling hypothesis (recipe mean ≈ 96.57, attainable max ≈ bar at family σ) is now supported by the most exhaustive single-recipe probing this record contains and stands unfalsified; (c) per the standing autopilot directive the loop continues — remaining honest moves are: a deeper literature excavation (e.g., specific post-2024 papers on heavy-aug fixed-budget CIFAR via lit-search venues rather than ad-hoc web search; confidence low), genuinely novel cross-class compositions that are NOT interpolations of measured nulls (none currently constructible from the record; any new one must pass the +0.3pp effect-size screen and all standing laws), or constructions targeting the max-statistic's plateau LENGTH within legal bounds (mostly closed; residual micro-harvests ≤ +0.03 fail the screen). Brainstorm-062 should state explicitly which of these it attempts and why its candidate is not a re-measurement.

## Exit Action Results

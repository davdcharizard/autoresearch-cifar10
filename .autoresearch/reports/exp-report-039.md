# Experiment Report: EXP-039 — BN running-stat momentum 0.1 → 0.25 (freshness side of the dose-response)

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.64%** (baseline 96.71, bar 96.81, delta −0.07; within the baseline band 96.4–96.7 — noise-band mean, no detectable level effect)
- **Branch**: autoresearch/exp-039 (discarded)
- **Artifacts**: brainstorm/brainstorm-039.md · plans/plan-039.md · logs/exp-log-039.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: EXP-038 inverted its own hypothesis and left the BN-momentum dial bracketed from one side only: m=0.02 lost −0.30 through constants LAG, with the mechanism visible at every phase (the cosine tail drifts weights to the final eval). If lag is the dominant constants-error term at m=0.1 too, freshening to m=0.25 (~4-batch horizon) buys ~2.5× less lag at arithmetically negligible variance cost (≈2,048 samples per channel estimate). The unique candidate with in-regime evidence: a slope measured on this exact recipe in the immediately preceding experiment.

**Hypothesis**: Fresher constants raise plateau evals (bar-pass if residual lag at 0.1 costs ≥ +0.25); falsified by a plateau within the baseline band → m=0.1 is the optimum from both sides and the dial closes with a complete dose-response curve.

## Approach
train.py only (4 insertions / 3 deletions — byte-mirror of EXP-038's diff with 0.02 → 0.25): `BN_MOMENTUM = 0.25` constant passed to all three `nn.BatchNorm2d` construction sites; CPU module walk verified 19/19 BN layers at 0.25, params 4,286,026 unchanged. Training path byte-identical — only the stat-buffer EMA coefficient differs.

## Execution
Single pristine run (gates clear at poll 1: GPU-0 apps=0, load=3.0; launched 20:03:11; rc=0; total 490.5s; no watchdog trigger). Signatures exactly baseline: watchdog windows 21.7–22.7ms across 30 ticks; coarse profile 267 windows mean 22.31ms, 0 >27ms (200-step quantization-safe windows max 22.5ms); 139 epochs / 13,446 steps; VRAM 1613.0MB; startup 18.1s. No retries, no adjustments, no errors.

## Results
- **best 96.64 (ep133), final 96.54, final_test_loss 0.1859 — the hypothesis's falsification branch, cleanly.** The plateau is in the baseline band; the bar-pass scenario (residual lag ≥ +0.25 at m=0.1) is refuted.
- **No lag remains at m=0.1.** Hot phase family-equal at every waypoint (ep5 62.23 vs ~64, ep10 75.52 vs ~75, ep20 82.01 vs ~79) — where EXP-038's m=0.02 showed dramatic depression (ep5 35.30), m=0.25 shows zero gain. The 10-batch default horizon already tracks the drifting weights closely enough that 2.5× fresher buys nothing measurable, even during the hot phase where per-batch drift is maximal.
- **The variance cost is real and visible, exactly where arithmetic predicted it would be smallest.** Last-15 plateau: mean 96.449 (noise-band vs family ~96.5) but spread 0.49 ≈ 3× family (~0.15). A ~4-batch EMA produces visibly noisier eval draws; the max-statistic harvests the distribution top anyway (best 96.64 ≈ family best-of-run), which is why the BEST lands family-equal while individual evals scatter. The two error terms have crossed: at m=0.02 lag dominates (−0.30 mean, 4× scatter); at m=0.25 variance dominates (mean ~flat, 3× scatter); m=0.1 sits at the crossover with both small.
- **Dose-response curve complete**: {0.02: −0.30 lag-dominated · 0.1: optimum · 0.25: ~0.0 variance-dominated}. The PyTorch default is the measured optimum of the BN-constants estimator, now from BOTH sides — the strongest closure form the program has (same as WD EXP-015 and gradient-noise EXP-023/024).
- **Trajectory fit**: 34th consecutive miss, but a cheap and conclusive one — single run, zero deficit currency spent, and it converts EXP-038's one-sided inversion into a fully bracketed law.

## Verification
First-failure-stop per plan-039. Pre-condition: profile pristine (mean 22.31ms, 0 slow >27 on both 50-step and 200-step windows — no quantization ambiguity), 139 epochs ✓. Integrity: params 4,286,026 ✓, training_seconds 300.0 ✓, eval_lines 139 = num_epochs ✓. **Condition 1 FAILED on merits: 96.64 < 96.81.** Conditions 2–3 skipped per protocol (incidental: rc=0, 490.5s ≤ 600; 139 = 139). No false-failure risk: clean profile, full epoch count, family signatures, and the diagnostic suite (hot phase, plateau scatter, test_loss) forms a coherent variance-dominance story. Verdict: **no-improvement**.

## Unexplored Avenues
- **Interior doses (m=0.15, 0.2)**: closed by bracketing logic — both endpoints of the {0.1, 0.25} interval read ≤ 0 mean effect, and the interior of a flat-to-negative interval cannot clear a +0.24 bar.
- **Phase-varying momentum (fresh early, smooth late)**: the data refute both halves — freshness buys nothing even at the hot phase (this run), smoothness damages even at the plateau (EXP-038, the tail still drifts). No phase exists where either deviation from 0.1 wins. Also a runtime-mutation compile-guard risk.
- The BN-constants engineering space (distribution: EXP-029; smoothness: EXP-038; freshness: this run) is now measured at every corner — nothing remains on this axis.

## Next Steps
1. **Record the BN-momentum dial as closed BOTH sides** and promote the EXP-038 Failed Approaches entry to count 2 with the completed curve (high confidence).
2. **The in-regime-evidence screen has now consumed its only available candidate** — the frontier returns to fully closed with no measured-slope leads; next brainstorms must source candidates from the directive's escalation path: re-read the papers/files for unexamined angles, recombine near-misses, or radical-but-law-compliant architecture, while honestly pricing every candidate against the four laws + absorption (medium confidence in the framing).
3. **If pursuing the radical-architecture branch**: any reshape must be free in whole-block dt (2.5ms/block), early heat, numerics, and noise simultaneously — micro-attachment-scale additions (~0.2ms, EXP-037 pricing) are the only structurally cheap class, and EXP-037 measured that class's best-evidenced member at zero (low confidence anything in this class clears +0.3).

## Key Learning
The BN running-stat estimator's two error terms cross exactly at the framework default: m=0.02 is lag-dominated (−0.30 mean, 4× scatter — EXP-038), m=0.25 is variance-dominated (mean flat, 3× scatter, zero hot-phase gain — this run), so m=0.1 is the measured two-sided optimum. A 10-batch EMA already tracks even hot-phase weight drift; there was no residual lag to harvest. The dose-response method turned an inverted hypothesis into a complete law in two cheap runs — but it also exhausted the program's last in-regime measured slope.

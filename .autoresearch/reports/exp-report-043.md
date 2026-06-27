# Experiment Report: EXP-043 — Full-alternation two-member ensemble (2 × 4x ResNet-20, per-step alternation, logit-mean inference)

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.07%** (baseline 96.71, bar 96.81, delta −0.64; ~0.5 BELOW the baseline band 96.4–96.7)
- **Branch**: autoresearch/exp-043 (discarded)
- **Artifacts**: brainstorm/brainstorm-043.md · plans/plan-043.md · logs/exp-log-043.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: The dense-kernel implementation of function-space (multi-mode) averaging — the last unmeasured averaging mechanism, pre-registered by exp-report-042 after the grouped implementation was hardware-closed. Two fully-independent 4x ResNet-20 members (independent Kaiming draws, disjoint batch streams) alternate full-batch steps so each runs the EXACT certified baseline recipe at half the step count; inference is the logit mean via a `MeanEnsemble` module satisfying `Eval.evaluate()`'s contract directly. Eval thinning (every 3rd epoch below 60% progress) absorbed the 2× eval cost.

**Hypothesis**: The 2-member decorrelation gain (+0.5–0.8 in literature) survives member starvation (−0.3 to −0.5 estimated) and lifts the plateau above 96.81. Pre-registered branches: (ii) sub-bar with clearly-below-family test_loss = mechanism real, starvation-limited; (iii) in-band with family-equal loss = no usable diversity.

## Approach
train.py only (72 insertions / 31 deletions): `ResNet` untouched; `MeanEnsemble` wrapper; two members + two selective-WD optimizers; dual compile warmup (second hit the inductor cache — startup 11.5s); per-STEP alternation (`step % 2`) keeps both members equally fresh at every eval (epoch-boundary law); time-keyed LR applied to the active optimizer so both anneals complete. Five CPU sanities passed pre-launch (init diversity, alternation isolation incl. BN stats, eval contract, params 8,572,052, thinning predicate).

## Execution
Single pristine run (gates clear poll 1; launched 21:24:52; rc=0; total 470.3s; no watchdog trigger). Signatures: 200-step windows mean 22.34ms, max 22.5, 0 > 27ms — the dense-kernel/zero-dt-cost claim confirmed exactly; 139 epochs / 13,426 steps (~6,713 per member); 85 evals (thinning exact); VRAM 1655MB. No retries, no adjustments, no errors.

## Results
- **best 96.07 (ep137), final 96.06, final_test_loss 0.1961 — a THIRD outcome shape, below both pre-registered branches.** The plateau (last-15 mean 95.915, spread 0.48) sits ~0.6 below the single-model family mean with 3× scatter, and test_loss is slightly WORSE than family (~0.185), not better.
- **The decomposition is nonetheless mechanism-positive**: single 4x members at ~6,713 steps price at ~95.5–95.7 on the starvation ladder (EXP-002/005/007 interpolation), so the ensemble's 95.9–96.07 plateau implies a REAL function-space decorrelation gain of roughly +0.3–0.5 over its own members — squarely the literature range for 2 members. The gain exists; it is simply 2–3× smaller than the cost of buying it: halving each member's steps costs ~−0.9 against the 96.57 single-model mean.
- **The averaging axis is now bracketed end-to-end.** Fork-at-zero / full diversity (this run): 96.07. Fork-at-one / zero diversity (canonical SWA, EXP-032): 96.60 = mean, zero gain; EMA (EXP-011): 96.46. Every interior point (mid-fork ensembles, brainstorm-043 Idea B) trades member quality against a diversity fraction and interpolates to ≲96.6 — sub-bar by ≥0.2. More members only worsen starvation. With EXP-042's kernel closure of in-one-pass implementations, ensemble multiplicity under this fixed budget is CLOSED as a class.
- **Trajectory fit**: 37th non-improvement at a metric value. The elevated plateau scatter (0.48) is the starved-member signature (members still moving per remaining step at anneal end), echoing the max-statistic law: the harvested plateau wants converged, settled members — exactly what budget-splitting denies.

## Verification
First-failure-stop per plan-043. Pre-condition PASS (profile pristine: 66 quantization-safe windows mean 22.34ms, 0 > 27; 139 epochs; params 8,572,052; training_seconds 300.0; 85 evals as designed). **Condition 1 FAILED on merits: 96.07 < 96.81.** Conditions 2–3 skipped per protocol (incidental: rc=0, 470.3s ≤ 600; 85 ≤ 139). No false-failure risk: clean profile, exact signatures, and the decomposition (gain visible over priced members, loss not below family) forms a coherent mechanism story. Verdict: **no-improvement**.

## Unexplored Avenues
- **Mid-fork ensemble (Idea B)**: now closed by interpolation rather than untried — both endpoints of the diversity-starvation curve are measured and the interior cannot exceed ~96.6. Only worth revisiting if a future improvement moves the single-model mean DOWN-bar (it won't — bar moves with baseline).
- **Heterogeneous ensembles (different architectures per member)**: more diversity per member-pair, but each member still pays the half-step toll and the second architecture would be a worse-than-certified net; dominated.
- **The MeanEnsemble eval-contract pattern and the eval-thinning predicate** are validated reusable engineering (this run actually finished FASTER than family, 470.3s, despite 2× eval cost).

## Next Steps
1. **Record the ensemble/multiplicity class as measured-closed** (full bracket: EXP-011/032/042/043) and the strategic law — under a fixed budget, diversity must be bought with steps and the 2-member gain (+0.3–0.5) is strictly smaller than the half-step starvation cost (−0.9) (high confidence).
2. **The program returns to novel mechanism construction** with one more law in hand; the only flagged-untried single-model gap remains within-cliff width asymmetry (64/160/256, low prior) — next brainstorm should weigh it against deeper recombinations of validated parts (medium confidence in framing).
3. **Protocol carry-over**: pre-registered replicate pair for any future promising mid-band read (carried from exp-report-040/041/042; medium confidence).

## Key Learning
Function-space averaging — the last open half of the averaging dichotomy — is real but unaffordable under a fixed budget: two independent members at half steps each showed the literature-sized decorrelation gain (+0.3–0.5 over their starvation-priced level) yet landed at 96.07 because halving steps costs ~−0.9; with SWA/EMA as the zero-diversity endpoint (zero gain at zero cost), the entire diversity-starvation curve is bracketed sub-bar and the ensemble/multiplicity class joins weight averaging as measured-closed. The plateau the max-statistic harvests wants one fully-trained model, not a committee of half-trained ones.

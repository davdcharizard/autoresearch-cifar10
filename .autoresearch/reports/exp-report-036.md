# Experiment Report: EXP-036 — LABEL_SMOOTHING 0.1 → 0.2 (last unmeasured recipe constant, in-domain anchor dose)

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.58%** (baseline 96.71, bar 96.81, delta −0.13)
- **Branch**: autoresearch/exp-036 (discarded)
- **Artifacts**: brainstorm/brainstorm-036.md · plans/plan-036.md · logs/exp-log-036.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) of a ResNet-20-family model within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397 (EXP-006 recipe); improvement requires ≥ +0.1pp (bar 96.81). σ context (EXP-027): baseline-recipe mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: An audit of the "recipe space closed" law found a factual gap — the dose-probe series (EXP-009/010/012/013/014/015/023/024) covered mixup, peak LR, batch, padding mode, warmup, WD, and momentum, but LABEL_SMOOTHING entered bundled at EXP-000 and was never dosed in 36 experiments. Both in-domain anchors (cifar10-fast, airbench) use 0.2 on CIFAR-10 wide nets under budget-matched one-cycle — 2× our value. LS dose sets the target logit margin (max-prob target 1−ε), the most direct loss-side lever on the EXP-032 decision-boundary-limited diagnosis, at exactly zero cost in every closed currency (dt/heat/noise/VRAM/epochs).

**Hypothesis**: LS=0.2 raises the converged plateau LEVEL at byte-identical execution signatures, predicting best_test_acc ≥ 96.81. Falsified by a clean converged plateau within/below the baseline band (96.4–96.7) → brackets 0.1 as at-or-above optimal from one side; a clearly negative read (<96.4) closes the dial entirely.

## Approach
One-line change on `autoresearch/exp-036`: `LABEL_SMOOTHING = 0.1` → `0.2` (train.py L27). The constant feeds the two training-side `F.cross_entropy(..., label_smoothing=...)` sites (timed step + compile warmup); eval (`evaluator.evaluate`) untouched. No other changes; signatures expected baseline-identical. Standard composite gated launch with baseline watchdog thresholds (contention 4×>27ms, STARTUP_KILL tick 10, NaN guard, divergence eval<15% after ep5, wall cap 600s); the elevated train-loss trace from the higher LS CE floor was pre-flagged as expected arithmetic, not divergence.

## Execution
Single clean run, exactly per plan. Gates cleared on poll 1 (GPU-0 idle, load 4); launched 18:46:23. All 31 watchdog windows 21.7–22.7ms, no guard ever fired; process exited normally at tick 33. rc=0, total 479.9s, startup 9.2s, VRAM 1613.0MB, 139 epochs / 13,439 steps, params 4,286,026 — every signature byte-identical to baseline, confirming implementation purity. No retries, no adjustments, no errors.

## Results
- **best_test_acc 96.58** (final 96.58, final_test_loss 0.2860). Last-8 evals 96.45–96.58, tight converged plateau centered ≈96.52; plateau onset (first eval ≥96.0) ep 127 vs family ~120.
- **The read is a precise null**: 96.58 sits within 0.01 of the baseline-recipe mean (≈96.57, σ≈0.16). At full epoch count and identical signatures, doubling the smoothing dose produced **zero measurable level shift** — not a small positive, not a degradation; the dial is FLAT on [0.1, 0.2] under this recipe.
- The hypothesis is falsified in its primary form (no level rise) and lands in the pre-registered falsification arm: LS=0.1 is bracketed as at-or-above optimal from the 0.2 side. Notably the OVER-smoothing harm risk also failed to materialize — the margin-compression both directions predicted simply does not move top-1 here.
- test_loss 0.286 vs family ~0.185 is the pre-flagged hard-label CE arithmetic under compressed logits (Eval measures CE against hard labels; LS=0.2 caps target max-prob at 0.8-ish logit gaps). Equal accuracy at much higher hard-label CE confirms the logit compression happened as designed — the margin structure changed, the boundary quality did not.
- Trajectory fit: 31st consecutive miss; this extends the local-optimum law to the FULL recipe constant set — every constant in train.py has now been individually dosed at least once, and every probe returned the incumbent as optimal or flat. The in-domain-anchor transfer heuristic also takes a hit: cifar10-fast/airbench use 0.2 with derandomized-flip/cutout augmentation, while under our TA+RE budget the LS dose is simply inert — augmentation strength absorbs the regularization role LS plays in lighter-aug recipes (consistent with Müller et al.'s task-dependent dose-response and with EXP-035's finding that the implicit-flatness/regularization budget is saturated).

## Verification
First-failure-stop per plan-036. Pre-condition (clean-run profile): 267 windows, mean 22.3ms, 0 slow >27ms (≤2 required), 139 epochs within 139±4 → uncontaminated, read honest. Integrity: params 4,286,026 ✓, training_seconds 300.0 ✓, eval_lines 139 = num_epochs ✓. **Condition 1 FAILED on merits**: 96.58 < 96.81. Conditions 2–3 skipped per protocol (incidentally: rc=0, 479.9s ≤600; 139 ≤139). No false-failure risk: summary block parsed directly from run.log, plateau confirmed across last 8 evals (no single-eval artifact). Verdict basis: valid clean run, necessary condition failed → **no-improvement**.

## Unexplored Avenues
- **LS=0 / 0.05 (down-bracket)**: the flat [0.1,0.2] read makes a large gain at 0 unlikely, but the dial's lower half is technically unmeasured. Expected value is low: flatness at the top end plus the known hard-target overconfidence risk under heavy TA suggests 0.1→0 is at best noise-level. Only worth running as a cheap slot-filler if no better instrument exists.
- **Margin-shaping losses beyond LS** (logit-norm penalty, complement-class mass): brainstorm Candidate 3. The LS null weakens the premise — if doubling complement-class mass does nothing, fine-grained margin shaping through the same mechanism is unlikely to be live. Unanchored coefficient tuning under σ=0.16 remains a fishing-expedition risk.
- **Anchor-recipe transfer caveat now measured**: future brainstorms should weight in-domain anchors by AUGMENTATIONregime match, not just dataset/schedule match — LS 0.2 transfers only where augmentation is lighter.

## Next Steps
1. **Treat the loss-target axis as closed alongside recipe/schedule/optimizer/sharpness/data/capacity**; the next idea must come from a structurally different mechanism class — e.g., training-data curriculum/ordering effects within the fixed budget, or BN/eval-state manipulation that respects the Eval() boundary (low confidence any remains).
2. **Re-audit for remaining unmeasured degrees of freedom** the way EXP-036 was found (audit gaps, not new mechanisms): e.g., the fc-init damping note from EXP-035's report (kaiming on the final Linear is unusually hot; zero/small-init fc is standard in modern recipes) — zero-dt, single-line, unmeasured (medium confidence it's a real, if small, lever).
3. **σ-aware strategy**: with 31 misses and every axis returning the incumbent, consider that the recipe sits at a genuine local optimum whose remaining headroom (~0.2–0.3 to bar) is below single-run resolvability for most single-dial moves; prioritize ideas with plausible effect ≥ +0.3 only (high confidence in the framing).

## Key Learning
Every recipe constant in train.py has now been individually dosed and the incumbent won or tied every time — EXP-036 closes the audit gap with a precise null: LS is FLAT on [0.1,0.2] under TA+RE, because heavy augmentation absorbs the regularization role label smoothing plays in the lighter-aug anchor recipes it was imported from. In-domain anchors transfer only when the augmentation regime matches.

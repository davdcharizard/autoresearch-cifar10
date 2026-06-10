# Report EXP-048: GridMask occlusion — distributed grid vs Cutout's single hole
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md
- **Plan**: plans/plan-048.md
- **Log**: logs/exp-log-048.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s budget on a single H20. Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. This experiment probed the one untested augmentation sub-lever — occlusion PATTERN — by swapping Cutout's single hole for a distributed GridMask grid at matched occlusion strength.

## Idea & Hypothesis
Chosen idea: replace `cutout_batch` (one 16×16 hole) with a GPU-vectorized GridMask (Chen et al. 2020) that deletes a periodic grid of squares (per-image random period d∈[8,16], removed-square side = 0.5·d → removed-area ≈ 25%, matched to Cutout-16 to isolate pattern from strength), holding the rest of the recipe fixed. Reasoning: occlusion is a proven, non-redundant lever here (Cutout +1.1pp, orthogonal to TrivialAugment) but only its size was ever tuned; GridMask's distributed deletion preserves more global structure while forcing redundancy, and literature reports it beats Cutout on CIFAR. Hypothesis: throughput-neutral (dt 8ms, ~91 ep); IF a distributed pattern regularizes better, best_test_acc ≥ 96.32; else within ±0.25pp (closing the sub-lever).

## Approach
Single-file change to `train.py`: added `GRIDMASK_D_MIN=8`, `GRIDMASK_D_MAX=16`, `GRIDMASK_RATIO=0.5` and a `gridmask_batch` function mirroring `cutout_batch`'s vectorized GPU style (coordinate-grid mask via `torch.arange`+broadcasting, `((coord-offset) % d) < side` intersected on both axes, `masked_fill`, seeded GPU RNG, no `.item()` syncs). Swapped the single training-loop call Cutout→GridMask. A CPU self-test confirmed removed-area mean 0.254 (matched to Cutout's ~25%, per-image 0.15-0.32). No recipe/optimizer/schedule/seed/compile-mode/param changes. Two deliberate decisions: matched removed-area to Cutout (fair pattern-not-strength test, avoids over-regularization); omitted GridMask's optional grid rotation (keeps a single static-shape vectorized op → CUDA-graph-safe, dt-neutral).

## Execution
One clean run on idle GPU 1 (foreign job isolated on GPU 0 — no contention), 402.0s wall, no retries, no NaN/traceback.

## Results
- **Primary metric**: best_test_acc 95.60% (baseline 96.22, delta **−0.62pp**, −0.64%) — below the 96.32 bar.
- **Observations**:
  - **Throughput-neutral, clean fair test**: dt steady 8ms (620×8ms, 76×9ms, 2×11ms), num_epochs 90 ≈ baseline ~91 — unlike GhostBN (EXP-047) this added no dt, so there is NO epoch confound; the −0.62pp is a genuine matched-strength, matched-epoch comparison.
  - final_test_loss 0.2100 > baseline 0.195 (worse on loss too), slightly slower early convergence (ep1 35.72% vs ~45.7%).
- **Analysis**: The hypothesis is answered negatively and cleanly: at matched ~25% removed-area AND matched epochs, GridMask's distributed grid-of-squares occlusion is a LESS effective regularizer than Cutout's single contiguous hole on this net (−0.62pp, worse loss). The likely mechanism: Cutout removes one contiguous region, leaving the rest of the 32×32 image intact and forcing the net to classify from the remaining whole-object context; GridMask scatters small deletions across the entire image, degrading fine 32×32 features everywhere and providing a weaker "use the rest of the object" signal — a softer but more pervasive corruption that this small-image net tolerates worse. This fits the EXP-037 pattern (augmentation QUALITY/pattern tweaks don't help on this saturated recipe) and extends it: occlusion PATTERN, like border-mode, is null/negative. It also confirms the prior tuning was well-chosen — single-hole Cutout-16 is the better occlusion here.
- **Key Learning**: At matched occlusion strength and matched epochs, distributed GridMask occlusion is WORSE than Cutout's single hole (95.60, −0.62pp, loss 0.21>0.195) on this net — the occlusion-PATTERN sub-lever is closed; single contiguous Cutout is the better pattern.

## Verification
- **Conditions**: Condition 3 (`best_test_acc ≥ 96.32`) FAILED at 95.60; conditions 1 (clean run within budget), 2 (throughput-neutral — dt 8ms, 90 ep, a genuinely fair test), and 4 (no hard-constraint violations — `train.py` only, eval untouched, once/epoch, no new deps, seed 42, deterministic mask → no seed hacking) all PASSED.
- **Review Notes**: Results trustworthy and notably CLEAN — throughput-neutral (8ms) at matched epochs (90≈91) and matched occlusion strength (~25% removed, self-test-verified), so this is an uncontaminated apples-to-apples occlusion-pattern comparison (no epoch or strength confound). No integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean fair run, primary necessary condition failed (−0.62pp).

## Unexplored Avenues
- **Stronger GridMask (literature defaults, ~40-64% removed)**: would test pattern at higher strength, but adding MORE occlusion to a saturated recipe regressed every time (CutMix/dropout/GhostBN) and would underfit further — almost certainly worse, not better. Not worth a loop.
- **GridMask ADDED on top of Cutout (not swapped)**: double occlusion → over-regularization → near-certain regression (the ADD-a-regularizer failure mode). Not worth a loop.
- **GridMask with rotation**: adds a rotation invariance flavor but breaks the static-shape vectorized op (dt/graph risk) and rotation is orthogonal to the pattern question; low value.
- The occlusion sub-lever is best considered closed: single-hole Cutout-16 (already in the recipe) is the better occlusion, and the distributed-pattern alternative is worse.

## Next Steps
- **The augmentation axis is now fully mapped and closed** (strength, policy, mixing, cooldown, border-quality, AND occlusion-pattern) — alongside every other accuracy axis. The 96.22 k=4/300s ceiling is comprehensively mapped (high confidence this is the practical frontier).
- **Next loop: the directive's "combine previous near-misses"** (medium-low confidence) — e.g., the EXP-048 brainstorm's alternate #2 (aug-cooldown EXP-034, the only >baseline result, + throughput-neutral Gradient Centralization EXP-031) tests whether two orthogonal sub-noise levers add to clear +0.1; higher implementation complexity, components are noise-level.
- **Failing a novel mechanism, a clean confirmation run documenting the 96.22 ceiling** (low confidence of a gain, high information) — per NEVER STOP, continue attempting radical/combination ideas rather than stopping.

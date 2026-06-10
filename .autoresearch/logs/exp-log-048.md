# EXP-048: GridMask occlusion — distributed grid replacing Cutout's single hole

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md
- **Plan**: plans/plan-048.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-048
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the GridMask swap per plan-048 (Milestone 1). Added `GRIDMASK_D_MIN=8`, `GRIDMASK_D_MAX=16`, `GRIDMASK_RATIO=0.5` to the hyperparameter block and a `gridmask_batch(x, d_min, d_max, mask_ratio)` function mirroring `cutout_batch`'s vectorized GPU style (coordinate-grid mask via `torch.arange` + broadcasting, `masked_fill`, seeded GPU RNG `torch.randint`/`torch.rand`, no `.item()` syncs). It deletes a periodic grid of squares: per-image random period `d∈[8,16]`, removed-square side `round(0.5*d)`, random per-image offset; `((coord-offset) % d) < side` on both axes intersected → grid of holes. Swapped the single training-loop call `cutout_batch(inputs, CUTOUT_SIZE)` → `gridmask_batch(...)`; left `cutout_batch` defined (now unused) for baseline reference. A CPU self-test confirmed removed-area fraction mean 0.254 (matched to Cutout-16's ~25%), per-image range 0.15-0.32, correct shape and masking. AST parses; the swap is the only training-path change; recipe/optimizer/schedule/seed/compile-mode/param-count all unchanged.

### Surprises & Discoveries
- **Genuinely throughput-neutral (dt 8ms, 90 ep)** — unlike GhostBN (EXP-047, +1ms→78ep), GridMask's vectorized mask added no measurable dt (620×8ms, 76×9ms, 2×11ms), confirming the same-op-class prediction. So this is a CLEAN fair test at matched epochs (90≈91) — no epoch confound.
- **GridMask is WORSE than Cutout at matched strength**: 95.60 vs baseline 96.22 (−0.62pp), final_test_loss 0.2100 > 0.195, slightly slower early convergence (ep1 35.7% vs ~45.7%). At matched ~25% removed-area and matched epochs, the distributed grid-of-squares occlusion regularizes LESS effectively than Cutout's single contiguous hole on this net.

### Decisions
- Matched GridMask's removed-area to Cutout (~25%, side=0.5·d) rather than using GridMask's literature-typical aggressive defaults (~40-64% removed) — to isolate occlusion PATTERN from STRENGTH (a fair test), and to avoid the over-regularization that sank every ADD-a-regularizer experiment on this saturated recipe.
- Omitted GridMask's optional grid ROTATION — keeps it a single static-shape vectorized op (CUDA-graph-safe, dt-neutral), and rotation is not essential to the distributed-occlusion hypothesis.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (402.0s total wall)

Description:
- Ran the baseline k=4 ResNet-20 recipe with Cutout-16 replaced by matched-strength GridMask (~25% removed-area, distributed grid of squares) on idle GPU 1 (foreign job isolated on GPU 0, no contention). Tests whether a distributed occlusion pattern regularizes more effectively than a single hole.

Observations:
- **Throughput-neutral, clean fair test**: dt steady 8ms (620×8ms, 76×9ms, 2×11ms), num_epochs 90 ≈ baseline ~91 → matched-epoch fair comparison, no epoch confound. Banner params 4,299,866 (unchanged). (source: run.log + dt extraction)
- **GridMask REGRESSED −0.62pp**: best_test_acc 95.60 < baseline 96.22, final_test_acc 95.53, final_test_loss 0.2100 > baseline 0.195. Slightly slower early convergence (ep1 35.72% vs ~45.7% baseline). At matched ~25% occlusion AND matched epochs, the distributed grid is a LESS effective regularizer than Cutout's single hole. (source: run.log eval lines + summary)
- Clean completion: training_seconds 300.0, total_seconds 402.0 < 600, startup 2.4s, peak_vram 453.8 MB, no NaN/traceback. (source: run.log summary)

Key Metrics:
- best_test_acc: 95.60% (−0.62pp vs baseline 96.22); final_test_acc 95.53% @ ep90; final_test_loss 0.2100
- num_epochs: 90; num_steps: 34,924; training_seconds: 300.0; total_seconds: 402.0; startup_seconds: 2.4; peak_vram_mb: 453.8; num_params: 4,299,866 (unchanged)
- dt: steady 8ms (620×8ms, 76×9ms, 2×11ms) — throughput-neutral, same as Cutout

## Verification Results

### Conditions Checked
1. **Run completes cleanly within budget** — PASS. Summary printed; total_seconds 402.0 < 600; training_seconds 300.0; num_params 4,299,866 (unchanged); no crash/NaN. (run.log summary)
2. **Throughput-neutrality** — PASS. dt steady 8ms (620×8ms, 76×9ms, 2×11ms), num_epochs 90 ≈ baseline ~91 → a genuinely fair matched-epoch comparison (no confound). (dt extraction)
3. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 95.60 < 96.32 (−0.62pp vs baseline). → no-improvement.
4. **No hard-constraint violations** — PASS. `git diff --name-only` = `train.py` only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps (GridMask uses only torch); seed 42 unchanged; deterministic mask math on the existing seeded RNG → no seed hacking.

Verdict: **no-improvement** (primary condition fails, −0.62pp; clean fair throughput-neutral test). Run completed cleanly → Outcome = completed.

### Informational Metrics
- num_epochs / num_steps: 90 / 34,924 (throughput-neutral, ≈ baseline).
- peak_vram_mb: 453.8 (≈ baseline).
- final_test_loss: 0.2100 (worse than baseline 0.195) — GridMask is a less effective occlusion regularizer than Cutout even on loss, not just top-1.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)

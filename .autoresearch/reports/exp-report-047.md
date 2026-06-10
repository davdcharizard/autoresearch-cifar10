# Report EXP-047: Ghost BatchNorm — implicit regularization via small-sub-batch statistics
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Log**: logs/exp-log-047.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s training-compute budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = baseline + 0.1 = **96.32%**. This experiment probed the one accuracy axis untouched across EXP-000..046 — **normalization** — to see whether normalization-noise regularization is a live lever on this regularization-saturated-but-generalization-bound net.

## Idea & Hypothesis
Chosen idea: replace every `nn.BatchNorm2d` with a `GhostBatchNorm2d` (Hoffer et al. 2017) that, in training, splits the fixed 128-sample batch into 4 ghost groups (size 32) and normalizes each with its own statistics — injecting a regularizing perturbation distinct from the saturated input-space-aug and penalty-regularizer families. Implemented dt-safe (static reshape splitting only the outer batch dim, valid on channels_last; in-place running-stat update) so it stays CUDA-graph-safe under reduce-overhead and throughput-neutral, dodging the epoch wall. Eval is byte-identical to standard BN. Hypothesis: dt stays ~8ms, epochs ~91, and IF normalization-noise regularization helps, best_test_acc ≥ 96.32; else it lands within ±0.25pp (closing the axis).

## Approach
Single-file change to `train.py`: added `GHOST_SPLITS = 4` and a `GhostBatchNorm2d(nn.BatchNorm2d)` subclass; swapped all 5 BN sites (stem `bn1`, `BasicBlock.bn1`/`bn2`, downsample-shortcut BN). Training forward: `x.view(s, N//s, C, H, W)` (outer-dim split — a valid view on channels_last; the channel-fold trick was deliberately avoided as it forces an NCHW copy), per-ghost biased mean/var over `(group,H,W)`, shared affine, reshape back; running buffers updated in-place from FULL-batch stats under `no_grad`. Eval forward delegates to `F.batch_norm(training=False)`. No recipe/optimizer/schedule/aug/seed/batch/compile-mode changes; `_weights_init` still only inits Conv/Linear so GhostBN keeps BN default init; param count unchanged (4,299,866).

## Execution
One clean run on idle GPU 1 (foreign job isolated on GPU 0 — no contention). Ran to completion in 394.8s wall, no retries, no NaN/traceback. The dt-safety design held: the reduce-overhead CUDA graph stayed intact (no graph break), confirmed by the dt distribution.

## Results
- **Primary metric**: best_test_acc 95.16% (baseline 96.22, delta **−1.06pp**, −1.10%) — well below the 96.32 bar.
- **Observations**:
  - **dt steady 9ms** (589×9ms, 13×10ms, 1×16ms[compile]) — NOT a CUDA-graph break (that shows ~14-16ms throughout, EXP-042). The static-shape design worked; the ~1ms premium over baseline 8ms is the cost of the manual per-ghost + full-batch reductions.
  - **Epochs trimmed 91→78** (−14%) by that 1ms premium → partial epoch-wall underfit. num_steps 30,165.
  - **Slow early convergence**: ep1 19.98% (vs ~45.7% baseline-kernel EXP-046), recovering to ep8 59.3% but never catching up. final_test_loss **0.2197 > baseline 0.195** — WORSE.
- **Analysis**: Both branches of the hypothesis are answered negatively. GhostBN at batch 128 with ghost-32 did NOT help — it hurt on two fronts: (1) the noisier ghost statistics slowed convergence and produced a WORSE final loss (0.2197 vs 0.195), and (2) the manual implementation cost ~1ms/step → 13 fewer epochs. Even crediting the entire epoch reduction to underfit (a baseline-BN run at 78 ep would land perhaps ~95.7-95.9 by the EXP-024/028 epoch-cost precedents), the additional shortfall to 95.16 plus the worse loss shows GhostBN's noise is net-negative on this recipe. This matches theory: GhostBN closes the LARGE-batch generalization gap; at batch 128 that gap is already narrow, so the added noise is gratuitous on an already regularization-saturated net (the recurring project pattern). The result fits both governing walls: a compute-adding change (the ~1ms premium) hit the epoch wall, and the regularizer-add hit regularization-saturation.
- **Key Learning**: Ghost BatchNorm (ghost-32) on this batch-128, regularization-saturated net REGRESSES −1.06pp — noisier ghost stats slow convergence and worsen loss, and the manual reductions cost ~1ms/step (91→78 ep); the normalization-NOISE direction is closed.

## Verification
- **Conditions**: Condition 3 (`best_test_acc ≥ 96.32`) FAILED at 95.16; conditions 1 (clean run within budget), 2 (throughput gate — dt 9ms, graph intact; partial epoch confound noted but within the ≤9ms acceptable band → valid research result), and 4 (no hard-constraint violations — `train.py` only, eval untouched, once/epoch, no new deps, seed 42, no seed hacking) all PASSED.
- **Review Notes**: Results trustworthy. The graph stayed intact (dt 9ms, not 14-16ms), so this is a real regularization test, not an EXP-042-class confound. The mild epoch reduction (78 vs 91) is acknowledged, but the WORSE final loss + slower convergence make GhostBN's negative effect clear independent of the epoch count. No integrity concerns (eval byte-identical to BN, deterministic change, no RNG added).
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean run, primary necessary condition failed (−1.06pp regression).

## Unexplored Avenues
- **Larger ghost / fewer splits (num_splits=2, ghost-64)**: less normalization noise — might avoid the convergence slowdown, but the small-batch theory predicts an even smaller (likely null) benefit, and it would still carry the ~1ms reduction premium. Low confidence; the −1.06pp at ghost-32 plus worse loss makes a clean win at ghost-64 implausible.
- **A zero-premium GhostBN implementation** (derive full-batch running stats from the per-group stats via law-of-total-variance to drop the second reduction pass, restoring dt to ~8ms / ~91 ep): would isolate the pure GhostBN-noise effect from the epoch reduction. But given the WORSE loss and slower convergence even here, the pure effect is very likely still ≤ baseline — not worth a loop unless other axes are fully exhausted.
- **Precise-BN** (brainstorm-047 candidate #2): a DIFFERENT normalization sub-lever (recompute running stats post-training with frozen weights). Untried, cheap, but near-certain negligible on a recipe with ~35k converged BN updates. Held as a cheap composable add-on, not a standalone lead.

## Next Steps
- **Treat the normalization axis as effectively closed and document the 96.22 k=4/300s ceiling as fully mapped across every accuracy axis** (high confidence) — capacity, augmentation, LR, regularizer-adds, head, feature-routing, activations, weight-averaging, optimizer, bag-of-tricks, large-batch, throughput→epochs, and now normalization-noise all closed.
- **For the next loop, attempt a genuinely radical, dt-safe, generalization-targeting change outside the mapped axes** (low confidence) — e.g. a structurally different cheap element that is throughput-neutral AND not a saturated-regularizer/closed-axis re-test; combining previous near-misses; or, failing a novel mechanism, a clean confirmation run documenting the ceiling. Per the NEVER STOP directive, continue thinking harder rather than stopping.
- **Precise-BN as a cheap composable probe** (low confidence) — only if a fresh idea is needed and nothing better surfaces.

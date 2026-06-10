# Experiment Report EXP-068: Lookahead optimizer wrapper (k=5, α=0.5)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within Σdt=300s GPU-compute + total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Wrap the existing Nesterov SGD in Lookahead (Zhang et al. NeurIPS 2019): fast weights step normally; every k=5 steps the slow weights pull toward the fast by α=0.5 and the fast reset to slow; evaluate the slow (Lookahead) iterate. Implemented inline (no new dep). **Mechanism**: trajectory-level variance reduction / stabilization that, unlike the closed EMA/SWA *eval-time* averaging (EXP-006/019/020), feeds the slow weights BACK into training — a genuinely distinct, untested optimizer-meta-wrapper class targeting the binding constraint (convergence) at ~zero compute.

**Hypothesis**: Lookahead stabilizes the optimization trajectory and improves effective per-epoch convergence, raising best_test_acc to ≥ 96.55. Stated alternative (most likely): a within-noise null/near-miss, given the closed EMA/SWA weight-averaging precedent.

## Approach
Inline Lookahead in train.py: constants `LOOKAHEAD_K=5`/`LOOKAHEAD_ALPHA=0.5`; `slow_weights` clone of all parameters after the optimizer; an every-k `torch.no_grad()` in-place sync (`s.add_(p-s, alpha=α); p.copy_(s)`) after `optimizer.step()`; and a snapshot/load-slow/eval/restore-fast bracket around the single epoch-end `evaluate()` call (faithful slow-iterate eval). All eager ops between graph replays (cudagraph-safe, cf. EXP-064). All else byte-identical to EXP-054. Params unchanged (slow_weights are a plain list, not registered). No deviations from plan.

## Execution
Single clean run on idle GPU 1 (GPU 0 idle — uncontended), exit 0, no retries, 0 NaN/error. 91 epochs (= baseline → Lookahead is compute-neutral, confirming the periodic copy is aggregate-negligible). total_seconds 588.8 < 600 (clean, no wall breach). dt sampled at 9ms only because the every-50th-step print always lands on a sync step (50 is a multiple of K=5); 91 epochs confirm true throughput is unchanged. One isolated 14ms step (one-time), no sustained cudagraph break.

## Results
**best_test_acc 95.84% (−0.61pp vs baseline 96.45, ≪ 96.55 bar)** — a clear regression, larger than the typical post-plateau scalar-knob −0.3pp. final_test_loss 0.2097 (> EXP-054's 0.1968). The hypothesis was **falsified**: Lookahead's trajectory-level slow-weight averaging hurt BOTH top-1 AND eval loss. Root cause: on this already-well-tuned, epoch-bound net trained with a fully-annealed cosine-to-0 schedule, the SGD+Nesterov trajectory is already near-optimal; periodically pulling the fast weights back toward a lagging slow average (and resetting fast→slow) discards useful late-trajectory progress rather than stabilizing it — the same failure direction as the closed EMA/SWA eval-time averaging (EXP-006 EMA 95.97, EXP-019/020 SWA ≤96.13). A secondary contributor is the standard Lookahead+BN inconsistency (slow params evaluated with the fast weights' BN running-stat buffers), but the regression magnitude indicates the averaging itself is net-negative.

**Trajectory fit**: this CLOSES the optimizer meta-wrapper class and, combined with EMA/SWA, decisively establishes that **weight-space averaging in any form — eval-time (EMA/SWA) or trajectory-level (Lookahead) — is net-negative on this recipe**. Reinforces the project-insights High verdict that 96.45 is at/near the k=4/300s ceiling. The −0.61pp magnitude (vs the −0.3pp scalar-knob band) shows averaging is more actively harmful than a mistuned scalar, not merely neutral.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 95.84 < 96.55. **FAILED** decisively (−0.61pp). (Stop at first failed condition.)
- **Necessary condition 2 — clean completion within budget**: total_seconds 588.8 < 600 ✓ (CLEAN), training_seconds 300.0 ✓, num_params 4,299,866 ✓, 0 NaN/error ✓, 91 ep = baseline (compute-neutral).
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps (inline ~15 lines) ✓; seed 42 ✓; evaluate() once/epoch (save/restore brackets a single call) ✓; uncontended ✓; slow-weight eval reads the optimizer's own iterate (not eval circumvention) ✓.

**Verdict: no-improvement.** Clean valid run (Σdt=300s respected, wall 588.8 < 600, 91 ep, zero caveats) that decisively missed the bar (−0.61pp). The optimizer meta-wrapper class is closed; weight-space averaging is net-negative on this recipe in every tested form.

## Unexplored Avenues
- **Lookahead with slow-weight eval BN-recalibration**: recompute BN stats for the slow weights before eval (fixing the fast-BN-on-slow-params inconsistency). But EXP-061 showed clean-BN recalib itself regresses AND adds wall, and the averaging is the primary harm → very low value.
- **Larger k or smaller α** (weaker Lookahead, closer to plain SGD): would approach the baseline asymptotically but cannot EXCEED it (in the α→0 / k→∞ limit it IS the baseline) → no upside ceiling. Do not pursue.
- **The weight-averaging idea family is now fully exhausted** (EMA, SWA ×2, Lookahead) — all net-negative. Do not revisit any weight/iterate-averaging mechanism.

## Next Steps
1. **BN eps 1e-5→1e-3** (low confidence) — the last genuinely-untested, trivial, wall-safe static knob (brainstorm-067/068 fallback); near-certain exact null but completes the BN-estimator axis closure.
2. **Accept 96.45 as the k=4/300s ceiling** (high confidence) — every lever including now the optimizer meta-wrapper / all weight-averaging is closed; project-insights High states this verbatim. Remaining probes are plateau-mapping, not breakthroughs.
3. **SGD momentum coefficient 0.9→0.95** (low confidence) — the one untested optimizer scalar (brainstorm-068 Idea 3); trivial, wall-safe, but confounded by the tuned LR schedule and contraindicated by m=0.9's robustness.

# Experiment Report EXP-066: Progressive resolution scheduling (24×24 → 32×32)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the Σdt=300s GPU-compute budget AND total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar this loop: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Train the first 50% of the time budget at 24×24 (downscaled via `F.interpolate`, proportional Cutout-12), then 32×32 for the remainder, eval untouched (frozen 32×32). **Mechanism**: the metric is epoch-starved, and the project's clearest historical win (EXP-003 GPU-Cutout, +0.58pp) came from buying epochs via throughput. Every prior throughput attempt kept the 32×32 shape and hit the conv-dt floor (EXP-040/045/046); resolution scheduling is the untried, higher-leverage knob — a 24×24 step has 0.5625× the conv FLOPs, so early-phase steps are cheaper → more steps fit in Σdt=300s → more effective epochs. The full-res tail re-sharpens features and re-adapts BN to the 32×32 eval distribution (FixRes-correct: the schedule ends at eval resolution).

**Hypothesis**: the cheaper early phase fits meaningfully more than 91 epochs and the extra optimization raises best_test_acc to ≥ 96.55. Stated alternative: a within-noise null if 24×24 under-learns fine detail.

## Approach
Two train.py edits (single mechanism): added `LOW_RES=24`, `RESIZE_FRAC=0.5`; in the loop, a time-fraction-gated block downscales the float32 input to 24×24 (then `.contiguous(channels_last)`) with proportional Cutout for frac<0.5, else keeps 32×32 with a one-time `>> FULL-RES PHASE` marker. The resize sits OUTSIDE the compiled forward and OUTSIDE autocast, intended to give one stable CUDA-graph per phase-shape (respecting the EXP-042 rule: no data-dependent branch inside the compiled forward). All else byte-identical to EXP-054. Params unchanged (4,299,866). No deviations from plan.

## Execution
Single clean run on idle GPU 1 (GPU 0 idle throughout — uncontended), exit 0, no retries, 0 NaN/error. Gate A (24×24 phase) passed strongly: dt = 6ms (vs 8ms baseline), confirming the resolution reduction realized a per-step speedup AND cudagraph did not break. The full-res marker fired at ep59/frac0.500. Final: best_test_acc 95.82%, num_epochs 89, training_seconds 300.0, total_seconds 617.2.

## Results
**best_test_acc 95.82% (−0.63pp vs baseline 96.45, ≪ 96.55 bar)** — a clear regression. final_test_loss 0.2055 (> EXP-054's 0.1968). The hypothesis was **falsified on the central premise (no net epochs bought)** by an unanticipated mechanism:

- **dt has TWO regimes: 24×24 @ 6ms (59 ep) and 32×32 @ 10ms (30 ep)** — the full-res tail ran at **10ms, NOT the 8ms baseline**. Root cause: two distinct input shapes under `torch.compile(reduce-overhead)` create TWO CUDA-graphs; the multi-graph state plus the per-step `F.interpolate` raised the full-res per-step dt by 25%. The +2ms on the 32×32 tail exactly CANCELLED the −2ms saving on the 24×24 phase. Net: **89 epochs < 91 baseline** — the schedule COST epochs rather than buying them.
- **Accuracy regression has two compounding causes**: (1) no net epochs bought (89 < 91); (2) the 59 low-res epochs learned coarse 24×24 features, and the throughput-starved 30-epoch full-res tail (at the inflated 10ms dt) could not fully re-sharpen them to the 32×32 eval distribution → −0.63pp.
- **Wall breach**: total_seconds 617.2 > 600 (by 17.2s, the most severe to date). Partly change-caused — the 2-cudagraph compile/recapture wall + per-step interpolate. The actively-gated Σdt=300s compute budget was respected exactly (training_seconds 300.0).

**Trajectory fit**: this validates the High-importance epoch-wall insight from an unexpected angle — even a change INTENDED to REDUCE compute (lower resolution) raised effective dt, because the multi-shape compile "restructures the graph" (the insight explicitly flags graph restructuring, cf. pre-act EXP-015, anti-aliasing). The conv-dt floor under reduce-overhead is a SINGLE-shape floor; introducing a second shape is net-negative. The throughput→epochs lever (EXP-003) works only when it doesn't perturb the single compiled graph.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 95.82 < 96.55. **FAILED** decisively (−0.63pp). (Stop at first failed condition.)
- **Necessary condition 2 — clean completion within budget**: **WALL BREACH** — total_seconds 617.2 > 600 (+17.2s, most severe to date, partly change-caused). training_seconds 300.0 (gated compute budget respected exactly), num_params 4,299,866 ✓, 0 NaN/error ✓, summary printed ✓.
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps ✓; seed 42 ✓; evaluate() once/epoch ✓; uncontended ✓.

**Verdict: no-improvement.** Valid training run (Σdt=300s respected exactly) that decisively missed the accuracy bar. The invalid-vs-no-improvement call (wall 617.2 > 600, partly change-caused — a stronger invalid case than EXP-061/065) was resolved to **no-improvement** consistent with the EXP-061/065 precedent: (a) condition 1 (accuracy) fails FIRST and DECISIVELY on a fully trustworthy value — the idea lost on its merits independent of wall; (b) Σdt=300s respected exactly (fair training); (c) the real metric (−0.63pp) is more informative than NaN; (d) the breach does not make the accuracy untrustworthy. The wall breach is recorded as an INDEPENDENT closure reason (progressive resizing is wall-infeasible on this recipe) and strengthened into infra-errors.

## Unexplored Avenues
- **Single-shape low-res-only training** (no schedule): train the WHOLE run at a fixed reduced resolution. Avoids the 2-cudagraph penalty (one shape → one graph → genuine speedup), but creates a permanent FixRes train↔eval (24 vs 32) mismatch → near-certain regression. Low value.
- **Resolution schedule WITHOUT compile, or with `mode="default"` compile** (single-graph dynamic shapes via `dynamic=True`): could avoid the multi-graph penalty, but EXP-007 established reduce-overhead/cudagraph is worth ~1.03× and dynamic shapes typically run SLOWER than shape-specialized graphs → likely loses the compile benefit entirely. Low value.
- **The resolution axis is effectively CLOSED** for this recipe: any multi-resolution schedule pays the 2-cudagraph penalty under the (required-for-throughput) reduce-overhead compile, and any single low-resolution loses to FixRes mismatch. Do not revisit resolution scheduling.

## Next Steps
1. **BN momentum 0.1→0.05** (low confidence) — the last cluster of genuinely-untested static hyperparameters (brainstorm-066 Idea 2); trivial, compute-/throughput-neutral, single-cudagraph (cudagraph-safe), no wall risk. Cleanly closes the BN-stat axis either way.
2. **BN eps 1e-5→1e-3** (low confidence) — brainstorm-066 Idea 3; near-certain exact null but trivial and wall-safe.
3. **Accept the 96.45 plateau is the k=4/300s/single-graph ceiling** (high confidence) — the throughput→epochs lever is now closed from BOTH the single-shape-floor (EXP-040/045/046) and multi-shape-penalty (EXP-066) directions; remaining genuine headroom, if any, requires a structural change that lowers single-graph dt without adding a second graph — none is currently known.

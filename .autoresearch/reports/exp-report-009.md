# Report EXP-009: Compiled k=5 WideResNet (capacity, threading the k4–k6 gap)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Log**: logs/exp-log-009.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s training budget on one H20, editing only `train.py`.
Higher is better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%** (baseline + 0.1).

## Idea & Hypothesis
Chosen idea: raise width `WIDTH_MULT` 4→5 (stages {80,160,320}, 6.7M params) with the validated
`torch.compile(reduce-overhead)` enabler. Capacity is the project's only proven high-magnitude lever (+2.84pp at
k=1→k=4, EXP-001); the one reason more width failed (EXP-004 k=6: eager 22ms/step → 35 epochs → underfit 95.26)
was the *epoch wall*, which compile (~30% throughput, EXP-007) was expected to partially lift. k=5 was the
untested intermediate width. Hypothesis: ~55–65 epochs fit and the added capacity lifts acc to ~96.2–96.5%.

## Approach
`train.py`-only edits (+8/−2): `WIDTH_MULT` 4→5; `compiled_model = torch.compile(model, mode="reduce-overhead")`
after the model is on device, with the training forward routed through `compiled_model`; eval left on the eager
`model`. Everything else byte-identical to the EXP-003 recipe (Cutout(16), PEAK_LR 0.2, WD 1e-4, label smoothing,
batch 128, bf16, channels_last, Nesterov, cosine, seed 42). Per EXP-007, compiled-k4 alone = 95.92 ≈ baseline
(null standalone accuracy effect), so any gain over ~96.0 would be attributable to the k=5 width.

## Execution
One run, no retries/errors, clean compile (no graph breaks), exit 0. **Throughput came in far worse than planned**:
steady-state dt = **18ms/step (~7,000 img/s)** from step 50 onward, vs the planned ~11–12ms. k=5 is substantially
more compute-bound than k=4, so reduce-overhead's CUDA-graph win (which mainly removes *launch* overhead) helped
much less than on launch-bound k=4 (8ms). Consequently only **41 epochs** fit (vs k=4's 77) — squarely in the
epoch-starvation zone the plan flagged as "record, not abort." Completed in 363.3s total, peak VRAM 599.9 MB.

## Results
- **Primary metric**: **94.21%** (baseline 96.00, delta **−1.79 pp**, −1.86%) — far below both the +0.1 bar and baseline.
- **Observations**: final_test_loss **0.2440** (vs k=4's converged 0.204 and compiled-k4's 0.208) and still falling
  at the last evals (0.246→0.244) — the k=5 net was clearly **under-fit**, not converged. num_epochs 41, eval count
  41 (eval once/epoch confirmed), num_params 6,712,314 (k=5 confirmed), dt 18ms steady.
- **Analysis**: A decisive negative for the capacity lever. The hypothesis was wrong on its key premise — compile
  did **not** keep the epoch count viable: k=5 compiled (18ms) is 2.25× compiled-k4 (8ms) despite only a 1.56× FLOP
  ratio, because the compile benefit shrinks as the net moves from launch-bound to compute-bound. With only ~half
  of k=4's epochs on a 56%-larger model, the result is severe under-training. Notably it regressed below *even*
  eager k=6 (95.26 @ 35 ep, EXP-004); the k5<k6 inversion is volatile under-trained-regime noise, but the trend is
  unmistakable and monotone: at this 300s budget, **width↑ → effective-epochs↓ → accuracy↓**. This closes the
  capacity axis as a lever even with the compile enabler — the last clearly-untried high-upside direction. 96.0%
  (k=4) is confirmed the capacity sweet spot; a SIXTH axis (compiled-capacity-scaling) is now exhausted.
- **Key Learning**: A compiled k=5 WideResNet (6.7M) fits only 41 epochs at the 300s budget → severe under-training
  (94.21%, loss 0.244); compile's ~30% boost cannot make k≥5 trainable, so the capacity axis is closed at k=4.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.10) **FAIL** (94.21); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (41=41), num_params
  confirms k=5, compile is execution-only with EXP-007-established null standalone accuracy effect, so the
  regression is attributable to under-training from the width, not a confound or hack. No reward-hacking surface.
  The −1.79pp is a genuine, well-understood epoch-starvation result (corroborated by loss 0.244 still falling).
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, clean run; primary metric far below the +0.1 bar (cond 2 failed).

## Unexplored Avenues
- **Capacity via depth (ResNet-32, n=5) at k=4** — *brainstorm idea 3*. Adds capacity along a different axis, but
  depth grows sequential kernel launches (bad in this launch-bound regime) and lengthens the gradient path; given
  k=5 width already starved on epochs, depth is very likely to starve too. Low priority — probably the same trap.
- **Compiled k=4.5-ish (e.g. non-uniform widths, wider only in late stages)** — partial capacity adds that cost
  fewer epochs than uniform k=5. Plausible but the EXP-004→009 trend (width monotonically hurts) makes a clean win
  unlikely. Low priority.
- **The capacity idea is now exhausted at this budget** — no width/depth variant is likely to escape the epoch wall.

## Next Steps
1. **SiLU activation in place of ReLU** (brainstorm-009 idea 2) — *low-medium confidence*; the one architectural
   lever completely untried (nonlinearity axis), near-zero epoch cost, orthogonal to all exhausted axes. The
   natural next cheap probe now that capacity is closed. *Best next experiment.*
2. **LR-schedule / optimizer tuning on k=4** — *low-medium confidence*; peak-LR and warmup-fraction were never
   swept (only WD in EXP-005). A different, genuinely-untried recipe knob on the converged k=4 sweet spot.
3. **Accept 96.0% as a hard plateau** — *strategic*; SIX axes now exhausted (width, regularization, weight-
   averaging, training-length, channel-attention, and now compiled-capacity-scaling). Remaining moves (SiLU, LR
   tuning) are low-ceiling/noise-scale; convergence is near.

## Exit Action Results
- None defined for this goal — skipped.

# Experiment Report EXP-071: BatchNorm eps 1e-5 → 1e-3

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within Σdt=300s GPU-compute + total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Raise the BatchNorm numerical-floor `eps` from the PyTorch default 1e-5 to 1e-3 at all four `nn.BatchNorm2d(...)` construction sites (BasicBlock bn1/bn2, downsample-shortcut BN, stem bn1), via a single named `BN_EPS` constant; everything else byte-identical to EXP-054. **Mechanism**: eps is the floor in `y=(x−μ)/sqrt(σ²+eps)·γ+β`; for high-variance channels (σ²≫eps) it is negligible, for low-variance channels a larger eps mildly shrinks the normalized output — a tiny smoothing. It was selected as the *last genuinely-untested cell on the BN-estimator axis* (goal-learnings line 181) and the *safest possible knob*: optimization-stable (a larger eps is strictly more numerically stable, unlike the EXP-070 readout that destabilized training), logit-scale-neutral, effective-LR-neutral (unlike a momentum change), dt-neutral, and param-neutral.

**Hypothesis**: best_test_acc lands within the ±0.25pp noise band of 96.45 (most likely 96.2–96.45, NOT ≥ 96.55), confirming the BN floor is inert on this well-conditioned net and closing the BN-estimator axis. Stated upside (very low probability): mild dampening of low-variance channels nudges a few borderline cases over the bar.

## Approach
Two-part edit to train.py per plan-071: (1) added `BN_EPS = 1e-3` after L28 (`CUTOUT_SIZE`); (2) passed `eps=BN_EPS` to all four BatchNorm2d construction sites. Because `_make_layer` builds BasicBlocks in a loop, the four *source* sites cover all **22** instantiated BN modules. Pre-launch smoke confirmed: AST OK, forward (2,3,32,32)→(2,10), num_params **4,299,866 UNCHANGED** (eps adds no params), all 22 BN modules report `.eps==1e-3`, `git diff --name-only` == train.py only. No deviations from plan.

## Execution
Single clean run on idle GPU 1 (GPU 0 lightly loaded at 10%/1043 MiB, uncontended), exit 0, no retries, 0 NaN/error. dt steady **8ms** (occasional 9ms) — eps is a scalar in the BN epilogue, no cudagraph break, no throughput change. 89 epochs, num_steps 34343, training_seconds 300.0, total_seconds 584.4 < 600 (clean, no wall breach), peak_vram 461.5 MB. **Early gate (≤ep3) was healthy and explicitly distinct from the EXP-070 divergence**: eval climbed normally (ep1 34.53%, ep2 45.88%) rather than sticking at random ~10%, confirming eps did not destabilize the high-LR trajectory as predicted.

## Results
**best_test_acc 95.92% (−0.53pp vs baseline 96.45) — a clean miss, NOT the predicted inert null.** The hypothesis was directionally right (no improvement, no destabilization) but its "near-certain *exact* null" framing was wrong: a 100× larger BN eps cost a real −0.53pp. The mechanism is the mild one anticipated — eps=1e-3 dampens low-variance channels' normalized outputs enough to slightly under-utilize them — but the magnitude is non-trivial, landing squarely in the **"every scalar/static-knob retune lands −0.2 to −0.6pp"** band (EXP-067 insight; cf. BN-momentum EXP-067 −0.30pp, clean-BN EXP-061 −1.6pp). Crucially the degradation is *uniform across the run* (healthy early trajectory, final_test_loss 0.2050 vs EXP-054's 0.1968 — a small consistent gap), NOT an early-epoch catastrophe like EXP-070's logit-scale blowup. This is the key contrast: EXP-070 (dt-neutral but optimization-UNSTABLE) collapsed to random for 3 epochs; EXP-071 (dt-neutral AND optimization-stable) trains cleanly throughout but at a slightly worse operating point. Both confirm 96.45 is a finely-tuned brittle ceiling — the EXP-054 recipe's *implicit* BN eps=1e-5 is itself part of the tuned configuration.

**Trajectory fit**: 17th consecutive no-improvement since EXP-054. Closes the BN-estimator axis from its last untested angle (eps), complementing BN-momentum↓ (EXP-067) and clean-BN recalib (EXP-061). The BN-estimator axis is now fully mapped — every BN knob (eps, momentum, recalibration) regresses.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 95.92 < 96.55. **FAILED** (−0.63pp below bar). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget** (recorded for completeness): training_seconds 300.0 ✓, total_seconds 584.4 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 NaN/error ✓, 89 ep.
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps ✓; seed 42 ✓; evaluate() once/epoch ✓; uncontended (dt 8ms) ✓.

**Verdict: no-improvement.** Clean valid run (Σdt=300.0 respected, wall 584.4<600, dt 8ms/no graph break, train.py only) that missed the bar. Results trustworthy — direct metric parse, no NaN, healthy trajectory. NOT invalid (no constraint breach; eps adds no params) and NOT crash (real interpretable metric).

## Unexplored Avenues
- **BN eps *lower* than default (1e-5→1e-6)**: the symmetric probe — if a larger eps under-utilizes low-variance channels, a smaller one might sharpen them. But: (a) at 1e-5 the floor is already negligible for all but near-dead channels, so 1e-6 is even more inert (near-certain exact null this time, lower information than this run), and (b) it risks marginal numerical noise. Very low priority.
- **Per-stage / selective eps**: leave stem+early stages at 1e-5, raise only deep low-variance stages. Multi-knob, uninterpretable, no mechanism for upside. Reject.
- The BN-estimator axis (eps, momentum, recalibration) is now **closed from all three angles** — do not propose further BN-estimator knob retunes without a fundamentally new mechanism.

## Next Steps
1. **Accept 96.45 as the comprehensively-mapped k=4/300s ceiling** (high confidence) — 17 consecutive misses; every axis closed (architecture incl. readout, optimizer+meta-wrapper, schedule, normalization incl. now all BN knobs, weight-averaging, regularizers, batch, capacity, ALL augmentation sub-levers). Remaining feasible probes are pure plateau-mapping.
2. **SGD momentum 0.9→0.95** (low confidence) — the one untested optimizer scalar; but contraindicated (≈doubles effective LR → likely a real regression per the closed peak-LR axis, EXP-016). A clean axis-closer but expected to regress, not null.
3. **Nesterov on→off** (low confidence) — the last untouched optimizer boolean; near-certain small regression (Nesterov is the tuned setting), completes the optimizer-scalar map. Safest remaining untested knob after momentum.

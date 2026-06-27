# Experiment Report EXP-035: Periodic SAM — sharpness-aware ascent-descent every 5th step (ρ=0.05, BN-protected)

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Exp-log**: logs/exp-log-035.md
- **Verdict**: **no-improvement** (96.25 vs bar 96.81; baseline 96.71, Δ −0.46; −0.32 vs the run-level mean ≈ 2σ)

## Goal
Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget, train.py only. Baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81.

## Idea & Hypothesis
**Idea**: Open the last unprobed level-mechanism class — explicit loss-landscape sharpness. Every 5th step applies SAM (Foret et al. ICLR 2021): perturb weights to the worst-case point in a ρ=0.05 ball (global grad-norm scaled), take the descent gradient there, restore, step. Periodic application (LookSAM, CVPR 2022) makes the wall-clock arithmetic survivable: ~26-epoch deficit vs full SAM's ~65. Distinct from the bracketed gradient-noise axis (implicit stochastic flatness): SAM is explicit worst-case geometry, targeting EXP-032's decision-boundary-limited diagnosis.

**Hypothesis**: Flat-minima pressure raises the converged plateau LEVEL by more than the deficit cost (~0.35) → best ≥ 96.81. Falsified by cost-model gate-kill or a clean converged plateau below the bar.

## Approach
38 pure insertions in train.py at 4 sites: SAM_RHO/SAM_EVERY constants; one eager warmup pass (cudnn algo selection for the eager path lands in uncharged startup); bn_modules list; SAM branch between `loss.backward()` and `optimizer.step()`. Key engineering: perturbed pass through the EAGER `base_model` (shared weights) so BN-momentum toggling never touches a compiled graph (no guard recompiles); perturb/restore via in-place `p.data.add_/sub_` (same tensor identity for the compiled first pass); BN running stats frozen on the perturbed pass (momentum→0, davda54/sam pattern — EXP-029 normalization law); loss trace logs the unperturbed first-pass loss.

## Execution
Single clean run, gates passed at poll 1. rc=0, 112 epochs / 10,835 steps, total 430.6s, startup 11.4s (+1.4 for the eager warmup), VRAM 1639.8MB. Profile: 215 windows, mean 27.7ms, 0 slow >33ms — the cost model was EXACT (back-solved per-SAM-step cost 48.9ms vs predicted ~48; mixed 4:1 average 27.7 vs predicted 27.6). No instability at peak LR, no NaN, no retries.

## Results
**best_test_acc 96.25 (ep108); final 96.18; final_test_loss 0.1945. The decomposition is unusually clean and the mechanism verdict is unambiguous:**

1. **The epoch deficit explains the ENTIRE result.** −27 epochs × the measured ~0.014/epoch ≈ −0.38 predicted; −0.32 vs the baseline mean observed. SAM's retained level gain ≈ +0.06 — statistically zero.
2. **No flatness signature anywhere.** A real flat-minimum effect should show in test_loss even when accuracy doesn't move (the EXP-011/032 smoothing signature was loss-BETTER/acc-equal). Here test_loss finished at 0.1945, slightly WORSE than the baseline family's ~0.185 — SAM bought nothing in either currency. Contrast: published SAM gains assume basic augmentation; under TA+RE+LS+occlusion the implicit-regularization budget is already saturated (the dose-response peak measured across EXP-003…015), and explicit sharpness pressure is redundant with the heavy-aug gradient noise.
3. **Mechanism-class closure**: this was the first experiment to modify WHICH gradient is descended (worst-case-point gradients on 20% of steps), and the training dynamics were measurably indistinguishable in outcome from baseline-minus-epochs. Combined with the bracketed implicit-noise axis (EXP-023/024: both directions lose at byte-identical signatures), the flatness/sharpness family is now closed at both the implicit and explicit ends.
4. The plateau was still rising at cutoff (best at ep108/112) — the deficit also truncated the plateau-length the max-statistic harvests, a compounding penalty any per-step-cost mechanism must overcome. 30 consecutive misses; the certified recipe remains the measured optimum of every axis it touches.

## Verification
- Condition 1 (best ≥ 96.81): **FAIL** — 96.25. Pre-condition profile PASS (215 win, mean 27.7ms ≤ 31, 0 >33ms; epochs 112 = 139×22.4/27.7 ±0.4; params 4,286,026 exact; evals 112 = epochs). Clean and trustworthy; conditions 2–3 informationally pass (430.6s, 112=112).
- Trustworthiness: high — cost model landed exactly, no contamination, smooth family-normal trajectory.
- Verdict basis: clean miss, deficit-explained → **no-improvement**.

## Key Learning
Explicit sharpness minimization adds NOTHING once implicit flatness is saturated: under TA+RE+LS at batch 512 the SGD gradient noise already buys all the flat-minimum benefit, so periodic SAM (k=5, ρ=0.05, correctly BN-protected) returns exactly its epoch-deficit (−0.32 ≈ −27 ep × 0.014) with zero retained gain and no loss improvement. SAM's published CIFAR gains are calibrated to weak-augmentation regimes — screen sharpness-family techniques by the recipe's existing implicit-regularization budget, not by their headline numbers. Per-SAM-step cost measured at 48.9ms (2.18× baseline).

## Unexplored Avenues
- **ρ retune (0.02/0.1) or k retune (10/20)**: the k=5 result shows zero retained gain, not negative-but-promising — scaling the dose down only interpolates toward baseline-minus-fewer-epochs; scaling up adds deficit. Closed by the zero-gain reading.
- **ASAM / adaptive-ρ variants**: same mechanism class, same saturation argument; their published deltas over SAM (~+0.1–0.2) are smaller than the deficit they'd still pay. Closed by arithmetic.
- **Efficient SAM approximations (ESAM, SAF)**: cheaper per step, but the binding problem is the mechanism's zero retained gain here, not its price. Closed.

## Next Steps
1. **Strategic assessment (high confidence)**: 30 consecutive misses; the mechanism inventory is now closed at every probed point: recipe constants (both directions), schedule (family/shape/heat), optimizer (internal/geometry), gradient target (sharpness, both implicit and explicit), data (dose/schedule/resolution/composition), eval-side (BN/averaging/head), capacity (magnitude/allocation/depth/width-shape). The baseline's 96.71 recorded-max sits at the top of its own σ≈0.16 distribution. Honest remaining moves are genuinely novel mechanism classes only.
2. **Candidate genuinely-untouched class — the loss target's GEOMETRY (low confidence)**: every experiment kept one-hot CE+LS 0.1; never probed: logit-margin shaping that changes ARGMAX training pressure specifically (e.g., complement-class penalty or logit-norm regularization, both 1-line, dt-free). The boundary-limited diagnosis (EXP-032) nominally points here, but LS dose was bracketed and these are LS-adjacent — screen hard against the pressure-peak law before committing.
3. **Candidate dt-free init-class probe (low confidence)**: final-layer init scale (fc init × 0.1, "fixup-style" head damping) — zero dt, one line, affects early-heat trainability which the heat laws say is load-bearing; EXP-018/019 closed init tricks that ADD stability headroom, but head-only damping changes early class-logit magnitudes, a different sub-mechanism. Gate on ep1 eval signature.

## Exit Action Results
(no exit actions defined for this goal)

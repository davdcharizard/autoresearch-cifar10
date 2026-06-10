# Sharpness-Aware Minimization (SAM) — Foret et al., ICLR 2021

**Topic**: optimizer that seeks flat minima for better generalization.

## Core idea
Instead of minimizing `L(w)`, SAM minimizes the worst-case loss in an ε-ball: `min_w max_{‖ε‖≤ρ} L(w+ε)`. Practical per-step recipe:
1. forward-backward → grad `g`;
2. ascend to the worst-case neighbor `ε = ρ·g/‖g‖` (global L2 norm), `w += ε`;
3. second forward-backward at `w+ε` → grad `g_sam`;
4. restore `w -= ε`;
5. `optimizer.step()` using `g_sam`.

Cost: **2 forward-backward passes per step (~2×)**. Standard CIFAR radius `ρ ≈ 0.05`. Variants: ASAM (adaptive ρ), LookSAM / periodic-SAM (Liu et al. CVPR 2022 — ascent only every k steps to cut cost, retains most of the gain).

## Reported benefit
+0.3–1.0pp top-1 on CIFAR-10/100 / ImageNet at FIXED architecture — a genuine *generalization* gain (distinct from loss/calibration polish like SWA/EMA/GC). BUT the gains are demonstrated at **long schedules (100–200+ epochs)** on deeper nets.

## Result on THIS project (EXP-036 — NEGATIVE)
Sparse SAM (ρ=0.05, ascent every 5th step, plain SGD otherwise) to limit the compute wall → still **mean dt 8→10.2ms (1.27×) → 76 ep** (vs baseline 91) → **95.89% (−0.33pp vs 96.22 baseline)**, final_test_loss 0.197 ≈ baseline (converged, flat tail — not a gross underfit). torch.compile(reduce-overhead) handled the two-pass step with in-place `torch._foreach_` perturbation cleanly (no cudagraph error). reports/exp-report-036.md.

**Takeaway for this project**: SAM's irreducible 2× cost (≥16% fewer epochs even when sparse) outweighs a flat-minima benefit that does NOT transfer to a shallow 9-block 32×32 ResNet-20 already trained cleanly with BN+warmup+tuned recipe at the 300s budget. SAM/sharpness sub-axis CLOSED here — do NOT retry denser/full SAM (worse epoch wall → ~45 ep), ASAM, or larger ρ. Fits the compute-wall + "deep/long-schedule tricks don't transfer to shallow short-budget CIFAR" patterns (project-insights).

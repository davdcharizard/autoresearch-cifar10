# SAM: Sharpness-Aware Minimization (Foret et al., ICLR 2021, arXiv 2010.01412) + LookSAM (Liu et al., CVPR 2022, arXiv 2203.02714)

## Core method
- Minimize worst-case loss in an ε-ball: per step, (1) compute grads g1 at w, (2) perturb w' = w + ρ·g1/‖g1‖ (global L2 norm across all params), (3) compute grads g2 at w', (4) restore w and step with g2. Canonical ρ=0.05 for CIFAR; perturbs ALL trainable params.
- Cost: 2× forward-backward per step. LookSAM: the ascent direction drifts slowly — applying the full ascent every k=5 steps retains most of the gain at ~1.2× average cost (ImageNet/ViT evidence; smooth degradation with k).

## Published results (fixed-epoch regime)
- CIFAR-10 WRN-28-10 + basic aug: error ~3.5% → ~2.7%; with stronger aug the delta shrinks (regularizers partially compose).
- The gain is a converged-plateau LEVEL effect (flat minima → better generalization), not a transit-speed effect — the right currency for max-over-checkpoints metrics.

## Implementation gotchas (measured/established)
- **BN running stats**: freeze on the perturbed pass (set each BN's momentum to 0, restore after — davda54/sam `disable_running_stats` pattern). Normalization still uses batch stats; only the buffers stop updating. Violating this leaks perturbed-weight activation stats into eval-time constants (project law: EXP-029, −10.9 from mis-scaled BN constants).
- **torch.compile**: in-place `p.data.add_/sub_` between two compiled forwards is guard-safe (same tensor identity), BUT toggling `bn.momentum` can be baked/guarded in the compiled graph → recompile storm. Safe pattern in this repo: first pass through compiled `model`, perturbed pass through the eager `base_model` (shared weights) where attribute mutation is trivially safe. Eager pass costs ~1.22× the compiled pass (EXP-006 speedup ratio).
- **cudnn.benchmark**: first eager fwd/bwd triggers algo search — add one eager warmup iteration in the uncharged startup block.
- Use the FIRST-pass (unperturbed) loss for any loss logging so traces stay comparable to baseline.

## Project-regime arithmetic (EXP-035 context)
- Baseline dt 22.4ms; SAM step ≈ 22.4 + eager fwd/bwd ~24 + perturb/restore overhead ≈ ~48ms. k=5 average ≈ 27.6ms → ~113 epochs (−26 ≈ −0.35 by the linear deficit law). Full SAM (~46-48ms) → ~67 epochs (−0.9) — pre-lost under the measured deficit arithmetic.

## MEASURED RESULT (EXP-035, 2026-06-10)
- k=5 ρ=0.05 ran exactly on the cost model (48.9ms/SAM-step, 27.7ms mixed, 112 epochs) and returned best 96.25 = baseline-mean − deficit, retained gain ≈ 0; test_loss 0.1945 vs family 0.185 (no flatness signature). Conclusion: under TA+RE+LS @ batch 512 the implicit flatness budget is saturated — SAM-family techniques are redundant in heavy-augmentation recipes. Axis closed (with EXP-023/024 closing the implicit end).

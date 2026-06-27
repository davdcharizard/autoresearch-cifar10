# FreezeOut: Accelerate Training by Progressively Freezing Layers (Brock et al. 2017, arXiv:1706.04983)

**Status in this project**: tested and REFUTED for the fixed-time regime (EXP-055).

## Paper claims (fixed-epoch regime)
- Progressively freeze early layers during training; each layer follows its own cosine anneal completing at its freeze time t_i (optionally amplitude-scaled by 1/t_i to preserve integrated LR).
- Frozen layers are excluded from backward → up to ~20% training-time reduction on CIFAR DenseNets/ResNets with ~no accuracy loss.
- Sweet spot for first-layer freeze time around 0.5–0.8 of training; "unscaled" and "scaled" variants both viable.

## What EXP-055 measured here (fixed 300s charged budget, ResNet-20 4x, heavy-aug certified recipe)
- Freeze of stem+layer1 (~⅓ conv FLOPs, 5.2% of params) at p = 0.70 after an unscaled compressed anneal: backward saving DELIVERED in full — dt 22.5 → 15.8ms (31%), +1,550 tail steps, +16 plateau evals, zero recompile toll (graph-visible detach-flag + dual-variant warmup).
- Result 96.32 = recipe mean − 1.6σ, converged-flat plateau: the freeze package costs ~0.3 of plateau LEVEL — the paper's "freeze is free" inverts under fixed time even when the freed compute is fully recycled into steps. Tail-pressure law is two-sided (data EXP-025/033; parameters EXP-055).

## Implementation gotcha (cross-project value)
- Mid-run `requires_grad_(False)` on module params is a SILENT NO-OP under torch.compile (no recompile, no backward saving). Working pattern: a bool module attr read in forward gating `tensor.detach()` at the freeze boundary; pre-warm both flag values in the uncharged warmup; verify engagement by the step-time drop, never by prints. See infra-errors.md (EXP-055) and reports/exp-report-055.md.

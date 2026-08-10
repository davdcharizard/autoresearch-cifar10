# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` in this directory defines which academic proceedings `/lit-search` searches for this goal — seeded with the default top CS/ML conferences (NeurIPS, ICML, ICLR)
- Edit it to match this goal's research domain; venues are per-goal, so different goals can search different domains

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `papers/time-matters-regularization.md` | Phase-dependent regularization | Strong early regularization can be relaxed late without losing its generalization benefit. |
| `papers/mixed-sample-analysis.md` | Mixup/CutMix mechanisms | Mixed-sample methods provide efficient but distinct input-gradient regularization. |
| `papers/shakedrop.md` | Residual stochastic regularization | Residual-branch disturbance can improve ResNet families when strength and stability are controlled. |
| `papers/stochastic-weight-averaging.md` | Weight-space averaging | Averaging diverse late SGD iterates can improve CIFAR generalization with little overhead. |
| `papers/when-label-smoothing-helps.md` | Output-space regularization | Mild soft targets can improve generalization, but may overlap with mixed-sample augmentation. |
| `papers/sharpness-aware-minimization.md` | Flatness-aware optimization | SAM offers strong CIFAR upside but must be compute-aware under a fixed wall-clock budget. |
| `papers/last-minibatch-self-distillation.md` | Temporal consistency regularization | Half-overlapping batches provide fresh self-distillation targets without another model forward. |
| `papers/manifold-mixup.md` | Hidden-representation interpolation | One-pass mixing at early hidden boundaries can improve CIFAR residual-network generalization. |
| `papers/adaptive-sharpness-aware-minimization.md` | Scale-aware flatness optimization | ASAM improves full-run SAM on CIFAR, but sparse-dose transfer must discount the effect. |
| `papers/how-to-scale-your-ema.md` | EMA horizon scaling | EMA decay should be derived from an effective exposure horizon when update cadence or throughput changes. |
| `papers/modern-sharpness-generalization.md` | Sharpness/generalization limits | Adaptive sharpness is setup-dependent and cannot substitute for direct accuracy evidence. |
| `papers/randaugment.md` | Reduced-search image augmentation | Transfer a capacity-matched CIFAR policy explicitly; isolate RNG and preserve validated clean phases. |
| `papers/squeeze-and-excitation.md` | Channel recalibration | Strong capacity prior, but compact models need identity-safe initialization and measured kernel-launch cost. |
| `papers/deep-pyramidal-residual-networks.md` | Residual channel allocation | CIFAR generalization depends on how width is distributed, but projection overhead must be measured under fixed time. |
| `papers/ricap.md` | Spatial crop/patch augmentation | WideResNet CIFAR evidence supports center-sampled clipped Cutout; complement gating can preserve validated CutMix. |
| `papers/polyloss.md` | Polynomial classification losses | Poly-1 is a target-confidence-dependent gradient rescale; derive its coefficient from a bounded inflation budget. |
| `papers/temperature-cosine-softmax.md` | Cosine classifier temperature | Fixed scale can improve CIFAR accuracy but is feature-dimension-sensitive and must not be selected post hoc. |
| `papers/when-where-why-average.md` | Checkpoint averaging with annealing | Averaging is broadly efficient, but implemented kernel weights and BatchNorm state must be audited explicitly. |
| `papers/lookahead-optimizer.md` | Online optimizer variance reduction | Lookahead is low-forward-cost but can redundantly smooth a trajectory already summarized by EMA. |
| `papers/gradient-centralization.md` | Projected gradient optimization | Centralizing eligible weight gradients adds no forward pass, but its removed component and charged overhead must be audited. |
| `papers/deeply-supervised-nets.md` | Intermediate companion objectives | A lightweight training-only auxiliary head can shape hidden features without another backbone forward. |
| `papers/auxiliary-training.md` | Disposable auxiliary classifiers | Auxiliary heads can improve deployed representations, but full gains may depend on distillation/corruption machinery. |
| `papers/supervised-contrastive-learning.md` | Same-class batch geometry | SupCon can improve representations, but paired views, projection heads, and long training are central transfer caveats. |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add reference notes here)_ | | |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |

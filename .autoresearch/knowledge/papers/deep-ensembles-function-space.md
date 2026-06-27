# Deep Ensembles / Function-Space Averaging (family note)

**Papers**: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (arXiv 1612.01474, NeurIPS 2017); Fort, Hu, Lakshminarayanan, "Deep Ensembles: A Loss Landscape Perspective" (arXiv 1912.02757); Kondratyuk et al., "When Ensembling Smaller Models is More Efficient than Single Large Models" (arXiv 2005.00570); Wen et al., "BatchEnsemble" (arXiv 2002.06715, ICLR 2020); Havasi et al., "Training independent subnetworks for robust prediction" / MIMO (arXiv 2010.06610, ICLR 2021).

## Key claims relevant to this project

1. **Multi-mode vs single-mode averaging (Fort et al.)**: independently-initialized nets land in DIFFERENT loss-landscape modes; subspace/weight-averaging methods (SWA, EMA, dropout-subspace) explore ONE mode and capture far less functional diversity than even 2 independent solutions. Prediction (logit/prob) averaging across modes is where ensemble accuracy gains come from.
2. **Effect size**: 2-member deep ensembles on CIFAR-10 mid-size CNNs typically gain +0.4–1.0pp over a single member; gains shrink as members correlate (shared data order, shared augmentation realizations, shared hyperparameters) — init diversity alone retains most of the gain (Fort et al. §5).
3. **Matched-cost trade (Kondratyuk et al.)**: at EQUAL total FLOPs, an ensemble of smaller nets often beats one larger net — capacity → multiplicity reallocation is a real positive-slope direction when single-model capacity has diminishing returns.
4. **In-one-network ensembles**: BatchEnsemble (rank-1 factor members) and MIMO (subnetworks via multi-input/multi-output) get most of the 2–4-member gain at ~1× train/inference cost; MIMO WRN-28-10 CIFAR-10 ≈ +0.3–0.5 over single. Published under standard (crop+flip) augmentation — heavy-aug in-regime evidence does NOT exist (absorption-law caveat, though ensembling is multiplicity, not regularization).

## Project arithmetic (as of EXP-042, baseline mean 96.57 σ0.16, bar 96.81)

- Grouped-conv 2×3x design (per-member 48/96/192, total 96/192/384, groups=2): FLOPs 1.125× baseline → projected dt ≈ 24.1ms via ∂dt/∂FLOPs ≈ 13.3ms/unit (EXP-034) IF grouped kernels price like dense ones — the >256-total-channel cliff (EXP-040) on grouped kernels is UNKNOWN; always dt-gate.
- Sum-of-CE losses (not mean) keeps each member's per-step gradient/LR/noise byte-identical to the certified single-model recipe.
- Alternating-step variant (2 full-width members, one trains per step): zero kernel risk, but each member gets ~half the steps → starvation ladder (EXP-002/005/007) prices member plateaus ~96.0–96.2; needs top-of-range ensemble gain.
- Internal motivation: EXP-032 (SWA null with correct BN handling) + EXP-011 (EMA loss) measured the single-mode half of the dichotomy at zero/negative; the decision-boundary-limited diagnosis is exactly what function-space averaging targets.

## Outcome pointer

- **EXP-042 (grouped 2×3x)**: GATE_KILLed at 63.0ms — grouped convs cost 2.8× dense dt on the H20+compile+channels_last+bf16 stack at near-equal FLOPs; the in-one-kernel implementation family (grouped/BatchEnsemble-style) is hardware-closed here. See reports/exp-report-042.md.
- **EXP-043 (full-alternation 2×4x, dense kernels)**: 96.07 — the mechanism IS real (+0.3–0.5 decorrelation gain over starvation-priced members, literature-consistent) but unaffordable: half-step starvation costs ~−0.9. With SWA/EMA as the zero-diversity endpoint, the whole diversity-starvation curve is bracketed sub-bar; ensemble multiplicity closed under this fixed budget. See reports/exp-report-043.md.

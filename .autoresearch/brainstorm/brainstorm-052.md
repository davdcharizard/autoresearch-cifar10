# Brainstorm EXP-052
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Goal, metric, direction, constraints, verification live in the goal file.
     Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.22, EXP-012, 6c417a4). Bar = 96.32. -->

## Web Search & Literature Review

No new external search — the candidate is a standard, in-`torchvision` augmentation:
- **AugMix** (Hendrycks et al., ICLR 2020): augments an image by mixing several (default 3) independently-sampled augmentation CHAINS together with the original via random convex weights. Produces far more diverse augmented samples than a single-op policy, while keeping samples semantically close to the original (the skip/mix to the clean image bounds the distribution shift). Available as `torchvision.transforms.AugMix` (verified present: torchvision 0.24.1) → **no new dependency**. CPU-side transform (like TrivialAugment), but ~3× the per-image augmentation work (mixture_width=3 chains).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4). Bar = 96.32. **44 consecutive no-improvements (EXP-013..051)**; 6 lifetime improvements (EXP-000..012 era).

**The one lever that EVER broke a plateau is strong, diverse augmentation** (project-insights, High Importance): EXP-012 (TrivialAugment stacked on Cutout) lifted 96.00→96.22 AFTER WD (EXP-005) and mild Mixup (EXP-011) had read as "saturated." The explicit High-Importance implication: **do NOT declare a regularization/augmentation axis "closed" from weak-variant nulls — test the STRONGEST, most diverse variant before concluding.**

**What's been tried on the augmentation axis**: occlusion strength (EXP-013/021, Cutout-16 optimal), policy swap (EXP-014 RandAugment(2,9) ≈ TA, "policy saturated"), mixing (EXP-011/018 Mixup/CutMix underfit), cooldown (EXP-033/034/035), border-mode (EXP-037), occlusion-pattern (EXP-048 GridMask worse). **Not tried: a mix-of-chains augmentation (AugMix)** — qualitatively distinct from the single-chain auto-aug policies (TA, RandAugment): it superimposes multiple augmentation chains, raising sample DIVERSITY beyond what any single chain achieves.

**Counter-prior**: EXP-014 found swapping the single-chain POLICY (TA↔RandAugment) doesn't matter ("policy saturated"). AugMix is not just another single-chain policy, but the saturation result is a real caution. **Two governing walls** remain firm (epoch wall for compute-adds; polish-vs-top1 for compute-neutral); batch and residual-scaling axes were just closed (EXP-050/051).

**Tail honesty**: after 44 no-improvements with every other axis closed, EV is low. AugMix is the single most defensible remaining attempt because it is the only untried variant of the lever that has EVER worked here, and the High-Importance insight explicitly endorses testing it before closing the augmentation axis.

## Candidate Ideas

### 1. AugMix replacing TrivialAugmentWide (keep Cutout) — strongest diverse augmentation
**Summary**: Swap `transforms.TrivialAugmentWide()` → `transforms.AugMix()` (torchvision defaults: severity=3, mixture_width=3, chain_depth=-1, alpha=1.0) in the train transform, keeping RandomCrop+Flip, Cutout, and everything else fixed. A clean single-policy swap mirroring EXP-014's TA↔RandAugment test, but to a strictly more DIVERSE mix-of-chains augmentation.

**Reasoning**: Strong diverse augmentation is the ONLY lever that broke a plateau here (EXP-012, +0.22pp), and the High-Importance insight says test the strongest diverse variant before closing the axis. AugMix increases sample diversity beyond the single-chain TA/RandAugment (it superimposes 3 chains), directly targeting the generalization bound through the proven-effective intervention class. CPU-side (GPU-throughput-neutral on the Σdt budget).

**Sources**: Hendrycks et al. 2020 (AugMix); project-insights High-Importance "strong diverse augmentation" (EXP-012); `reports/exp-report-014.md` (policy-swap null caution).

**Estimated Effort**: low — one-line transform swap. Plus a wall-clock feasibility check (AugMix is ~3× TA's CPU cost).

**Risk Assessment**: TWO real risks. (a) **CPU-cost / wall-clock**: AugMix's 3-chain cost is heavier per image; if the dataloader can't keep up with NUM_WORKERS, the GPU idles (Σdt budget still fills, but WALL-clock balloons toward the 600s/10-min hard limit). Must monitor wall-clock; if it projects >600s, that's an infeasibility finding. (b) **Policy-saturation null** (EXP-014): the auto-aug policy may simply be saturated and AugMix lands within ±0.25pp. Worst case is a clean no-improvement or an infeasibility (too slow). No crash/divergence risk. Severity tunable down if needed.

### 2. Stochastic Depth (masked, graph-safe) — branch-level depth regularization
**Summary**: During training, multiply each BasicBlock's residual branch by a per-sample Bernoulli mask (survival prob ~0.8, inverted-scale), implemented as a static elementwise multiply (NOT data-dependent skipping) to stay CUDA-graph-safe.

**Reasoning**: A distinct generalization regularizer (drops whole residual branches, unlike unit-dropout EXP-022); throughput-neutral if masked (graph-safe).

**Sources**: Huang et al. 2016 (Stochastic Depth); `reports/exp-report-022.md` (dropout underfit).

**Estimated Effort**: medium — per-block Bernoulli mask in forward, graph-safe.

**Risk Assessment**: Strong null/underfit prior — it is a regularizer-ADD on a short budget (dropout EXP-022, GhostBN, SAM all underfit/regressed), AND its benefit is depth-driven (this is a shallow 9-block net, cf. zero-init-γ/LayerScale/deep-supervision all null). Likely underfits.

### 3. PReLU — learnable-slope activation
**Summary**: Replace ReLU with `nn.PReLU` (learnable per-channel negative slope) at the three sites.

**Reasoning**: Distinct from the smooth SiLU already tested (piecewise-linear, learnable).

**Sources**: He et al. 2015; `reports/exp-report-028.md`.

**Estimated Effort**: low.

**Risk Assessment**: Activation axis closed (SiLU null ×2); the per-channel PReLU likely adds ~1ms (non-fusing, like SiLU/LayerScale) → mild epoch wall, and the activation lever is generalization-null here. Low EV.

## Idea Evaluation

After 44 no-improvements, the selection criterion is which remaining attempt has the best mechanism-story tied to a lever that has actually worked here, at acceptable risk.

**Mechanism / evidence**: Candidate 1 (AugMix) is the only candidate in the intervention class that EVER produced a gain on this project (strong diverse augmentation, EXP-012), and the High-Importance insight explicitly endorses testing the strongest diverse variant before closing the axis. Candidates 2 (stochastic depth) and 3 (PReLU) are both in families with strong null/underfit priors on this shallow short-budget net (regularizer-add; activation-closed).

**Risk**: Candidate 1's risks are feasibility (CPU cost → wall-clock) and a policy-saturation null — both fail gracefully (infeasibility or no-improvement), and the CPU cost is monitorable. Candidates 2/3 carry the same epoch-wall / shallow-net-null risks as the just-closed LayerScale, with weaker mechanism stories.

**Decision**: Lead with **AugMix** — the single most defensible remaining attempt (the only untried variant of the only lever that ever worked), explicitly endorsed by the High-Importance augmentation insight. Stochastic depth and PReLU are weaker alternates held for later loops.

## Chosen Idea
**Selected**: Candidate 1 — AugMix replacing TrivialAugmentWide (keep Cutout), torchvision defaults.

**Why this idea**:
Strong, diverse augmentation is the only intervention class that has ever broken a plateau on this project (EXP-012, +0.22pp), and the High-Importance project insight explicitly directs testing the STRONGEST diverse augmentation variant before declaring the axis closed. AugMix (mix-of-3-chains, in torchvision, no new dep) is the untried strongest-diverse variant — strictly more diverse than the single-chain TA/RandAugment whose swap was saturating (EXP-014). It is a clean one-line swap targeting the generalization bound through the proven-effective class, with only graceful failure modes (infeasibility from CPU cost, or a saturation null).

**Hypothesis**:
Replacing TrivialAugmentWide with AugMix (keeping Cutout) is GPU-throughput-neutral on the Σdt budget (dt ~8ms), though wall-clock rises with AugMix's heavier CPU augmentation (must stay < 600s). IF the extra augmentation diversity regularizes better on this generalization-bound net, best_test_acc ≥ 96.32. Falsified if within ±0.25pp of baseline (auto-aug genuinely saturated, confirming EXP-014 extends to mix-of-chains → closes the augmentation axis decisively) or if the CPU cost pushes wall-clock > 600s (infeasibility — would then retry with reduced mixture_width/severity).

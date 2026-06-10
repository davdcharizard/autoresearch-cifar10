# Brainstorm EXP-057
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **AugMix (Hendrycks et al., ICLR 2020)** (knowledge: AugMix; train.py comment): AugMix's DEFINING property — the one that makes it work where naive heavy aug fails — is that it produces several independently-augmented versions ("chains") of an image, convex-mixes them (Dirichlet weights), then convex-mixes the RESULT with the ORIGINAL CLEAN image (Beta weight). The clean-mix BOUNDS the per-image distribution shift; the multi-chain mix adds diversity without any single harsh distortion dominating. EXP-056 omitted BOTH the multi-chain and the clean-mix (it applied one harsh 5-op stack) → −1.06pp. This loop restores them GPU-side.
- **TrivialAugment (knowledge/papers/trivialaugment.md)**: applies exactly ONE random op per image at random magnitude — also a shift-bounding structure (never stacks). Lifted CIFAR k=4 WRN to 96.22.
- `affine_grid`/`grid_sample` core torch (no new dep, confirmed EXP-056). The EXP-056 op primitives (per-sample affine + photometric) are reusable.

## Experimental History Review
- **Current best 96.45** (EXP-054): `RandomApply([AugMix() w3], p=0.5)` — full 3-chain AugMix (clean-mixed, the torchvision default) on a ~50% SUBSET. The augmentation-diversity lever is the only thing that lifts top-1 here.
- **The CPU-side lever is MAPPED at its wall-limited frontier** (EXP-052/054/055): rich AugMix is wall-infeasible uniformly on 8 CPU workers → forced to a 50% subset. Magnitude (EXP-053), width>3 (EXP-055), coverage<50% (EXP-055) all closed.
- **EXP-056 VALIDATED the GPU-augmentation throughput unlock**: moving augmentation into the train loop on the idle GPU is cheap (+1ms dt → 84 ep, dt-bound wall ~390s) and affords FULL coverage. BUT the naive policy (5 ops STACKED — rotate+shear+scale+brightness+contrast — on 100% of images, NO clean-mix) was far too harsh → 95.39 (−1.06pp), high test loss 0.224 (over-distorted train set). [goal-learnings § Failed Approaches EXP-056]
- **The precise fix (from EXP-056 analysis)**: the GPU path is right and cheap; the POLICY must BOUND the per-image shift like the augmentations that WORK — AugMix-style clean-image convex mixing, and/or TA-style single-op, and/or stochastic p<1 coverage. [project-insights § Experimental; goal-learnings]
- **Untried gap**: FULL-COVERAGE faithful AugMix — impossible on CPU (wall), now affordable on GPU. EXP-054 could only afford it on 50%; this tests whether the proven recipe at 100% coverage beats the 50% subset.

## Candidate Ideas

### 1. GPU AugMix (faithful: multi-chain + clean convex-mix), full-coverage — the true throughput-unlock realization
**Summary**: Replace EXP-056's harsh single-stack `gpu_augment` with a faithful GPU AugMix `gpu_augmix(x)`: build W=3 independently-augmented versions of the batch (each = per-sample affine[rotate/shear/scale] + photometric[brightness/contrast] with INDEPENDENT random draws), convex-mix the 3 chains with per-image Dirichlet(α=1) weights, then convex-mix that blend with the ORIGINAL CLEAN image with a per-image Beta(1,1) weight: `out = m·clean + (1−m)·Σ wᵢ·augᵢ`. Full-coverage, in the train loop before Cutout; CPU stays light (crop+flip). This is the torchvision-default AugMix structure (w3, the EXP-054 winner) reproduced GPU-side at 100% coverage (vs EXP-054's wall-forced 50%). ~3 grid_samples ≈ +1.5ms dt (cheap, per EXP-056's 0.52ms/transform).
**Reasoning**: Directly fixes both EXP-056 failure modes (restores multi-chain diversity + clean-mix shift-bounding) by replicating the EXACT mechanism that achieved 96.45 — but now at full coverage, the one thing CPU could never afford. EXP-054 showed richer-AugMix-on-subset beats shallower-on-all; EXP-055 showed coverage<50% hurts → the unexplored beneficial direction is coverage>50% toward 100%, which this delivers. Strong evidence (AugMix is the proven recipe; the clean-mix is its documented key property).
**Sources**: AugMix ICLR 2020; EXP-054 (w3 AugMix subset 96.45); EXP-056 (GPU infra validated, policy fix identified); goal-learnings § Failed Approaches EXP-056; knowledge/papers/trivialaugment.md.
**Estimated Effort**: medium (extend EXP-056 primitives to 3 chains + Dirichlet/Beta mix, ~30 lines; dt/epoch gate; idle-GPU launch).
**Risk Assessment**: (a) **epoch wall** — 3 grid_samples ≈ +1.5ms (dt ~9.5-10ms → ~78-82 ep); gated (abort if dt>11ms), can drop to W=2. (b) **100% coverage may still be too much even when clean-mixed** — if so, compose with stochastic p (candidate 3). (c) implementation correctness of Dirichlet/Beta per-image mixing on normalized data (mixing is linear → valid post-normalization). Graceful failure: a no-improvement that further localizes coverage vs the EXP-054 50% optimum.

### 2. GPU TrivialAugment-style single random op per image, full-coverage
**Summary**: Per image, pick ONE op uniformly from {rotate, shear, scale, brightness, contrast} at random magnitude (identity-ish otherwise), via the EXP-056 primitives — never stack. Full-coverage, before Cutout, CPU light.
**Reasoning**: TA's single-op structure is shift-bounded by construction and is a proven CIFAR recipe (96.22). Simplest correct fix to EXP-056's stacking. Cheapest (1 grid_sample, ~+0.5ms).
**Sources**: knowledge/papers/trivialaugment.md (TA 96.22); EXP-056.
**Estimated Effort**: low (~15 lines; a per-image op-index select).
**Risk Assessment**: TA on CPU got 96.22 < current baseline 96.45 — a GPU TA-equivalent likely lands ~96.2, BELOW the bar, unless full coverage + the modern recipe lifts it. Lower EV for beating 96.55; more of a sanity/diagnostic of the GPU primitives.

### 3. GPU AugMix on a stochastic subset (p≈0.5) — combine the clean-mix fix with the proven subset structure
**Summary**: Candidate 1's faithful `gpu_augmix`, but applied to only a random ~50% of each batch (the rest crop+flip+Cutout) — mirroring EXP-054's proven p=0.5 subset structure, now full-strength on the GPU subset (GPU cost no longer the constraint).
**Reasoning**: Hedge — if full-coverage faithful AugMix (candidate 1) is still too much, the proven 50% coverage with the clean-mixed GPU AugMix is the safest bet to at least MATCH 96.45 and possibly beat it (full-strength chains vs CPU's affordable chains).
**Sources**: EXP-054 (p=0.5 subset 96.45); candidate 1.
**Estimated Effort**: low-medium (candidate 1 + a per-image apply mask).
**Risk Assessment**: by construction close to the EXP-054 winner → likely lands near 96.45 (may not clear +0.1). Safer but lower ceiling than candidate 1.

## Idea Evaluation
All three operate within scope (train.py only) and on the now-validated GPU-augmentation path. Candidate 2 (TA single-op) has the weakest ceiling — its CPU analog scored 96.22, below the current 96.45 baseline, so it is unlikely to clear 96.55 and is better seen as a diagnostic. Candidate 3 (AugMix on 50%) is the safe hedge but by construction sits right at the EXP-054 winner (96.45) and may not clear the +0.1 bar — its value is robustness, not ceiling.

Candidate 1 (full-coverage faithful GPU AugMix) is the highest-EV and the most principled: it reproduces the EXACT recipe that achieved 96.45 (multi-chain + clean-mix, the torchvision AugMix default) while changing the ONE variable the CPU could never explore — coverage from 50% toward 100%. The evidence is the strongest available (AugMix is the proven plateau-breaker; clean-mix is its documented shift-bounding property, the exact thing EXP-056 lacked), the mechanism is clear (more images receive the proven gentle diverse augmentation), and the expected impact is the highest (it is the genuine realization of the throughput-unlock thesis: full-coverage AugMix, affordable for the first time). The epoch-wall risk is well-understood and gated, with W=2 and candidate-3 (subset) as graceful fallbacks. It dominates on evidence, mechanism, and impact; risk is managed.

## Chosen Idea
**Selected**: Candidate 1 — Full-coverage faithful GPU AugMix (multi-chain + Beta clean-mix).

**Why this idea**: It fixes both EXP-056 failure modes by replicating the exact mechanism that produced the 96.45 baseline (AugMix's multi-chain + clean-image convex mixing), now delivered at full coverage — the single beneficial direction (coverage>50%) that was wall-infeasible on CPU and is the whole point of the GPU throughput unlock. Highest evidence, clearest mechanism, highest ceiling; epoch-wall risk gated with W=2/subset fallbacks.

**Hypothesis**: A faithful GPU AugMix (3 independently-augmented affine+photometric chains, Dirichlet-mixed, then Beta-mixed with the clean image) applied full-coverage will hold dt ≈ 9.5-10ms (epochs ≥ ~78) and wall < 600s, and — because the clean-mix bounds the per-image shift (unlike EXP-056's harsh stack) while full coverage exposes ALL images to the proven diverse augmentation (vs EXP-054's 50%) — best_test_acc ≥ 96.55 (bar = baseline 96.45 + 0.1). A within-noise result near 96.45 would indicate coverage beyond 50% adds nothing (the EXP-054 subset is already optimal); a regression would indicate even clean-mixed full-coverage AugMix mildly over-regularizes at this budget.

# Brainstorm EXP-060
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **AutoAugment (Cubuk et al., CVPR 2019)** (well-established prior work, torchvision `transforms.AutoAugment(AutoAugmentPolicy.CIFAR10)`): a learned policy of 25 sub-policies (each = 2 ops with learned probability+magnitude) found by RL search to MAXIMIZE CIFAR-10 val accuracy. Reported WRN-28-10 CIFAR-10 error 2.6% (AA) vs 3.9% (Cutout-only) vs ~3.3% (baseline+std aug). The canonical strong CIFAR recipe is **AutoAugment + Cutout** — exactly our Cutout setup. AA is CIFAR-SPECIFIC (policy learned on CIFAR), unlike the dataset-agnostic TrivialAugment/RandAugment/AugMix we have already tried.
- **TrivialAugment (Müller & Hutter, ICCV 2021)** (already in project as EXP-012): TA is a tuning-free SIMPLIFICATION of AA that matched/slightly beat AA on average across datasets, but on CIFAR-10 specifically the learned AA policy and TA are within ~0.1–0.2% of each other — i.e. AA is a genuinely distinct point on the policy frontier, not dominated.
- **AugMix (Hendrycks et al., ICLR 2020)** (current best lever, EXP-052/054): targets robustness via Dirichlet chain-mixing; its clean-image mix BOUNDS per-image shift. AA differs mechanistically — AA applies a single 2-op learned sub-policy at full magnitude with NO clean-mix bounding, a stronger-but-CIFAR-targeted signal.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + CPU `RandomApply([AugMix()], p=0.5)` + GPU Cutout(16) + cosine(peak0.2)/Nesterov/LS0.1/WD1e-4 + compile(reduce-overhead), 91 ep, dt 8ms, 593s wall (tight).
- **THE lever that has repeatedly moved top-1 is augmentation DIVERSITY**, every other axis is closed:
  - Cutout +0.58 (EXP-003), TrivialAugmentWide +0.22 (EXP-012), AugMix-w2 +0.12 (EXP-052), AugMix-w3-p0.5 +0.11 (EXP-054). Monotonic diminishing but real.
- **Auto-aug POLICIES tried**: TrivialAugmentWide (EXP-012, 96.22), RandAugment(2,9) (EXP-014, 96.19 ≈ TA), AugMix family (EXP-052–055). **AutoAugment with the learned CIFAR10 policy has NEVER been tried** — the one remaining major policy, and the only CIFAR-SPECIFIC one.
- Augmentation sub-levers closed: severity/magnitude (EXP-053), coverage 35/50/100% (EXP-054/055/057 — ~50% effective optimal), occlusion pattern/size (EXP-013/021/048), label-mix Mixup/CutMix (EXP-011/018, underfit at 91-ep budget), GPU-delivery (EXP-056/057/059 — epoch-disadvantaged, CPU aug is epoch-FREE).
- Aug cooldown (disable aug in final epochs) NEAR-MISS on the OLD TA recipe: EXP-034 @0.10 → 96.26 (+0.04 over TA-baseline 96.22) — never re-tested on the stronger AugMix recipe where the clean-fit gap is larger.
- Closed axes (do not revisit): capacity ×4 directions (width/depth/realloc all hit the dt wall), optimizer family (AdamW EXP-043), LR peak/schedule (EXP-016/017/029), normalization (GhostBN EXP-047), residual scaling (LayerScale EXP-051), head (EXP-032/039), batch (EXP-025/050), SE/dropout/SAM/EMA/SWA/GC/PolyLoss/bag-of-tricks (all ≈baseline or worse), throughput→epochs (epoch-saturated at ~91).

## Candidate Ideas

### Idea 1 — AutoAugment(CIFAR10 policy) replacing AugMix (full coverage) + Cutout
- **Summary**: Replace `RandomApply([AugMix()], p=0.5)` with `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)` at native full coverage (every image gets one of 25 learned sub-policies), keeping GPU Cutout(16) and everything else byte-identical. This is the canonical AutoAugment+Cutout CIFAR-10 recipe.
- **Reasoning**: Augmentation diversity is the ONLY lever that has lifted top-1 here, and AutoAugment is the single major policy never tried — and the only CIFAR-SPECIFIC one (its 25 sub-policies were RL-searched to maximize CIFAR-10 accuracy). Each sub-policy applies ops with internal probabilities, so effective per-image strength is moderate (many images get near-identity) — this naturally matches the project's repeatedly-found "~50% effective coverage is optimal" without needing an explicit RandomApply wrapper. CPU-delivered → FREE w.r.t. the Σdt/epoch budget (runs in the 8 dataloader workers, off the timed step), so it preserves the full 91-epoch budget that the GPU-aug path sacrificed.
- **Sources**: AutoAugment paper (Web Search above); EXP-012/014/052/054 (policy frontier); EXP-054 recipe (in-scope train.py L171).
- **Estimated Effort**: Trivial — one-line transform swap (AugMix → AutoAugment(CIFAR10)).
- **Risk Assessment**: Low-risk failure mode (no-improvement at worst — a converged regression like other aug-policy swaps). Two real risks: (a) full-coverage AA could over-regularize vs AugMix's p=0.5 (mirroring EXP-057's full-coverage GPU AugMix regression) — mitigated by AA's internal per-op probabilities keeping effective coverage moderate, and by a p=0.5-coverage fallback if the gate shows underfit; (b) wall: AA applies ~2 ops/image at full coverage vs AugMix w3 on 50% — roughly comparable CPU cost, must gate the wall early (AugMix-w3-p0.5 was 593s, tight).

### Idea 2 — Aug cooldown @0.10 on the current AugMix p=0.5 recipe (combine near-miss with current best)
- **Summary**: Keep the EXP-054 AugMix recipe; disable TA/AugMix + Cutout for the final 10% of training (crop+flip only), letting the model fit clean-distribution data in the tail. Port the EXP-034 cooldown logic onto the current best recipe (it was only ever tested on the older TA recipe).
- **Reasoning**: EXP-034 cooldown @0.10 gave +0.04 over its TA baseline — a near-miss. The cooldown's mechanism (remove train-time aug noise so the final iterates fit the clean test distribution) should COMPOUND with stronger augmentation: AugMix creates a larger train↔test distribution gap than TA, so the clean-tail fitting headroom is larger. Throughput-neutral (just skips aug late), preserves 91 epochs.
- **Sources**: EXP-033/034/035 (cooldown sweep), EXP-054 (current best).
- **Estimated Effort**: Low — add a frac-based switch in the train loop to drop aug in the final 10%.
- **Risk Assessment**: Low risk; failure mode is no-improvement. EXP-049 (cooldown+GC) regressed but that was on the TA recipe and confounded by GC. The expected lift is small (~+0.04–0.1pp), right at the bar.

### Idea 3 — AutoAugment(CIFAR10) at p=0.5 RandomApply (coverage-matched variant)
- **Summary**: As Idea 1 but wrap AutoAugment in `RandomApply([...], p=0.5)` to match the proven ~50%-coverage optimum directly, rather than relying on AA's internal op-probabilities for effective coverage.
- **Reasoning**: The project has THREE times found ~50% coverage optimal (EXP-054 beat EXP-057's full coverage; EXP-055's lower 35% also hurt). If full-coverage AA over-regularizes, p=0.5 coverage is the validated sweet spot. This hedges Idea 1's main risk.
- **Sources**: EXP-054/055/057 (coverage frontier); AutoAugment paper.
- **Estimated Effort**: Trivial.
- **Risk Assessment**: Low. Largely redundant with Idea 1 — better treated as Idea 1's built-in fallback (run full-coverage first; if the early gate shows clear underfit/over-reg, switch to p=0.5) than as a separate experiment.

## Idea Evaluation
- **Evidence strength**: Idea 1 is strongest — AutoAugment has direct, well-cited CIFAR-10 evidence (it was the SOTA aug policy and the canonical AA+Cutout recipe), and it is the precise gap in this project's well-mapped policy frontier (TA/RA/AugMix tried, AA never). Idea 2's evidence is a within-noise +0.04 near-miss. Idea 3 shares Idea 1's evidence.
- **Mechanism clarity**: Idea 1 — clear: a CIFAR-learned policy delivers a distinct, possibly-better diversity distribution than the dataset-agnostic policies tried, through the same CPU-free delivery that preserves epochs. Idea 2 — clear but small-magnitude (clean-tail fitting). Idea 3 — same mechanism as Idea 1.
- **Expected impact**: Idea 1 highest — augmentation diversity is the only lever with a track record of clearing the bar here, and AA is a genuinely new, CIFAR-optimized point. Idea 2 is marginal-over-baseline by construction.
- **Risk profile**: All three fail gracefully (no-improvement). Idea 1's over-regularization risk is mitigated by AA's internal probabilities + a p=0.5 fallback (folding in Idea 3).
- **Feasibility**: All trivial/low effort, CPU-free, wall-gateable.
- **Conclusion**: Idea 1 (AutoAugment full-coverage, with Idea 3 folded in as a coverage fallback) is the lead: best evidence, the exact untried gap in the winning lever, and zero epoch cost. Idea 2 is the natural next loop if Idea 1 lands near-baseline.

## Chosen Idea
- **Selected**: AutoAugment(CIFAR10 policy) replacing AugMix at full native coverage, keeping GPU Cutout(16); all else byte-identical to EXP-054. Idea 3 (p=0.5 coverage) folded in as a built-in fallback if the early gate shows over-regularization/underfit.
- **Why this idea**: Augmentation diversity is the sole lever that has repeatedly lifted top-1 on this deeply-mapped plateau, and AutoAugment is the one major auto-aug policy never tried — and the only CIFAR-SPECIFIC one (25 sub-policies RL-searched to maximize CIFAR-10 accuracy). It is delivered on the CPU dataloader, so it is FREE w.r.t. the Σdt/epoch budget and preserves the full 91-epoch budget — unlike the GPU-aug path that repeatedly failed on epoch cost. One-line, well-evidenced, fits the winning pattern.
- **Hypothesis**: Swapping AugMix-p0.5 for full-coverage AutoAugment(CIFAR10)+Cutout, at a fair ~91-epoch run, will lift best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp), because the CIFAR-learned policy supplies a stronger, dataset-matched augmentation diversity distribution than the dataset-agnostic AugMix, through the same epoch-free CPU delivery.

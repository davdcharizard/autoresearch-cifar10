# Brainstorm EXP-053
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.34, EXP-052, 292a9e2). Bar = 96.44. -->

## Web Search & Literature Review

No new external search — this loop tunes the just-validated AugMix lever, already grounded in:
- **AugMix** (Hendrycks et al., ICLR 2020): mixes `mixture_width` independently-sampled augmentation chains (each `chain_depth` ops at a given `severity`, default sev=3 on a 1–10 scale) with the clean image via random convex weights. `severity` scales per-op MAGNITUDE; `mixture_width`/`chain_depth` scale chain COUNT/length. torchvision-native (no new dep).
- Project knowledge: `goal-learnings § Patterns` (augmentation-diversity is the only top-1 lever, EXP-012 + EXP-052) and `§ Protocol Findings` (CPU-augmentation wall-feasibility: ≲13ms/batch on 8 workers to fit <600s).

## Experimental History Review

**Current best / baseline**: **96.34%** (EXP-052, 292a9e2) — AugMix(mixture_width=2, chain_depth=1) replacing TrivialAugment, +0.12pp. Bar = 96.44. 7 lifetime improvements; EXP-052 just broke a 44-experiment no-improvement streak.

**The validated lever (now confirmed TWICE)**: augmentation DIVERSITY is the only thing that lifts top-1 here — TrivialAugment (EXP-012, +0.22) and AugMix multi-chain mixing (EXP-052, +0.12). Both raised top-1 with flat/lower loss (pure generalization). High-Importance insight: when every other axis is closed, increasing augmentation diversity is the move.

**Key constraint discovered in EXP-052**: the FULL-strength AugMix (default w3,d-1 = 21.1ms/batch, ~792s; even w2,d2 = 17.9ms, ~670s) is WALL-INFEASIBLE on 8 workers (budget gates on Σdt so epochs are preserved, but wall balloons past the 600s hard limit). The shipped w2,d1 (12.6ms/batch) is a deliberately-WEAKENED diverse aug — the lever likely has headroom. Feasible envelope: ≲~13ms/batch. severity is CPU-NEUTRAL (probed: w2,d1,sev5 = 12.1ms ≈ w2,d1,sev-default 12.6ms) — magnitude can be raised for free; chain COUNT/length cannot (CPU-bound).

**What's been tried on the augmentation axis**: occlusion strength (EXP-013/021, Cutout-16 optimal), policy swap (EXP-014 RA≈TA saturated), mixing (EXP-011/018 Mixup/CutMix underfit), cooldown (EXP-033/34/35 closed), border-mode (EXP-037), occlusion-pattern (EXP-048 GridMask worse), and now mix-of-chains (EXP-052 AugMix WORKS). **Not yet tried: tuning AugMix's own knobs** (severity, mixture_width/chain_depth under the wall, stochastic application).

**Caution from priors**: Cutout occlusion-STRENGTH is a clean interior optimum (both directions regress, EXP-013/021) — magnitude knobs tend to be interior-optimal here. But AugMix's clean-image mix bounds the distribution shift, so higher severity is safer in AugMix than raw Cutout. "Aug on every image" has been the working regime (any coverage reduction is a risk).

## Candidate Ideas

### 1. AugMix(w2,d1) with increased severity — push op MAGNITUDE on the new winner (single-variable, all-coverage, CPU-neutral)
**Summary**: On the EXP-052 winner `AugMix(mixture_width=2, chain_depth=1)`, raise `severity` from the default 3 to **6** (keep w2,d1, keep Cutout, all else fixed). A clean single-variable change: stronger per-op magnitude → a more spread-out augmented distribution → more diversity in the strength dimension, while every image still gets AugMix (no coverage change).
**Reasoning**: Augmentation diversity is the validated lever (EXP-012/052). Within the feasible wall envelope, severity is the ONLY diversity dial that is CPU-free (probed: severity doesn't change op count, so ~12ms/batch, certain-feasible). All-image coverage is preserved (the working regime). AugMix's clean-image mix caps the shift, mitigating the over-augmentation that made raw Cutout-20 regress.
**Sources**: Hendrycks et al. 2020 (severity param); reports/exp-report-052.md (w2,d1 winner + severity probe 12.1ms); goal-learnings § Patterns (diversity lever), § Failed Approaches (Cutout-strength interior optimum, EXP-013/021).
**Estimated Effort**: low — one-keyword change `AugMix(mixture_width=2, chain_depth=1, severity=6)`. Certain-feasible (CPU-neutral).
**Risk Assessment**: Graceful failure modes only. (a) Magnitude-knob interior-optimum (Cutout precedent) → severity=3 may already be near-optimal → within-noise null. (b) Over-augmentation at sev=6 → mild regression (clean-mix should bound this). No crash/feasibility risk (CPU-neutral, all-coverage). May land within ±0.25pp.

### 2. Intermittent FULL-strength AugMix via RandomApply (p≈0.5) — deliver the paper-validated w3 config to a subset
**Summary**: `transforms.RandomApply([transforms.AugMix()], p=0.5)` — apply the FULL default AugMix (w3, d-1, the literature-validated config) to ~50% of images; the rest get only crop+flip(+Cutout). Average CPU cost ≈ 0.5×21ms + 0.5×cheap ≈ ~12ms/batch (feasibility to be probed).
**Reasoning**: EXP-052's gain came from a deliberately-weakened AugMix; the validated full w3 was wall-infeasible uniformly. Stochastic application delivers the RICH full-diversity config to part of each epoch's images under the wall, testing whether per-image diversity depth (3 chains) beats uniform-but-shallow (w2,d1 on all).
**Sources**: reports/exp-report-052.md § Unexplored Avenues (throughput-recovery via stochastic application); torchvision `RandomApply`.
**Estimated Effort**: low-medium — wrap in RandomApply; REQUIRES a dataloader feasibility probe (the 50% that get full w3 cost ~21ms each — avg must land ≲13ms/batch).
**Risk Assessment**: TWO risks. (a) **Coverage reduction** — half the images get NO photometric/geometric aug; "aug on every image" has been the working regime, so this could wash the diversity benefit. (b) **Confounded interpretation** — coverage↓ and per-image-diversity↑ move together, so a null/regression can't be cleanly attributed. Feasible-but-confounded; weaker as a clean test than Candidate 1.

### 3. Re-test a complementary regularizer on the new AugMix base (e.g., mild Mixup α=0.2)
**Summary**: Add back a previously-saturating complementary lever now that the augmentation base has moved (e.g., mild Mixup on top of AugMix(w2,d1)+Cutout).
**Reasoning**: The base shifted; levers that read as saturated on the TA base might interact differently with AugMix's mixing.
**Sources**: goal-learnings § Failed Approaches (Mixup EXP-011 null, CutMix EXP-018 regressed).
**Estimated Effort**: low.
**Risk Assessment**: Low confidence — Mixup/CutMix are CLOSED (count:2, underfit at the short budget on the saturated recipe); base-jitter dominated the cooldown retests (EXP-049). Most likely a retread null. Held as a low-priority alternate.

## Idea Evaluation

After EXP-052 confirmed augmentation diversity is the live lever, the selection criterion is which feasible knob best extends it as a CLEAN, interpretable single-variable test.

**Mechanism / evidence**: Candidate 1 (severity↑) is the cleanest single-variable continuation — it isolates op magnitude with all-image coverage and every other variable fixed, and is CPU-neutral (certain-feasible). Candidate 2 (intermittent full AugMix) has higher diversity-fidelity (real w3) but confounds diversity with coverage and needs a feasibility probe — a null would be uninterpretable. Candidate 3 retreads closed levers (low confidence).

**Risk**: Candidate 1 fails only gracefully (null or mild over-aug regression), no feasibility/crash risk. Candidate 2 risks washing the benefit via coverage loss AND an uninterpretable result. Candidate 3 is a likely retread null.

**Expected impact**: Both 1 and 2 target the validated lever. Candidate 1's impact is bounded (magnitude knobs trend interior-optimal here, per Cutout EXP-013/021) but its result is always clean and informative (maps the severity axis on the new base). Candidate 2 has higher ceiling but lower interpretability and feasibility certainty.

**Decision**: Lead with **Candidate 1 (AugMix severity↑ to 6)** — the clean, certain-feasible, all-coverage single-variable push on the validated diversity lever, consistent with the project's strong preference for clean fair tests. Candidate 2 (intermittent full AugMix) is the higher-upside alternate for a later loop; Candidate 3 is low priority.

## Chosen Idea
**Selected**: Candidate 1 — AugMix(mixture_width=2, chain_depth=1, **severity=6**) replacing the default-severity AugMix(w2,d1), keeping Cutout.

**Why this idea**:
EXP-052 confirmed (for the 2nd time) that augmentation diversity is the only top-1 lever here, but shipped a deliberately-weakened AugMix (w2,d1) because the full config breached the 600s wall. severity is the one diversity dial that is CPU-NEUTRAL (probed feasible, ~12ms/batch) and preserves all-image coverage (the working regime), making it the cleanest single-variable way to push the validated lever further within the wall. It fails only gracefully and always yields an interpretable read on the severity axis.

**Hypothesis**:
Raising AugMix severity 3→6 on the w2,d1 base is throughput-neutral (dt ~8ms GPU, ~12ms/batch dataloader, total wall < 600s, ~91 ep). IF stronger op magnitude adds useful augmentation diversity on this generalization-bound net, best_test_acc ≥ 96.44. Falsified if within ±0.25pp of 96.34 (severity=3 already near-optimal — magnitude is interior-optimal here, as for Cutout EXP-013/021) or if sev=6 over-augments and mildly regresses (clean-image mix expected to bound this). Clean single-variable test: only `severity` changes.

# Brainstorm EXP-054
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.34, EXP-052, 292a9e2). Bar = 96.44. -->

## Web Search & Literature Review

No new external search — grounded in prior work already in context:
- **AugMix** (Hendrycks et al., ICLR 2020): mixes `mixture_width` augmentation chains with the clean image. The diversity comes from the NUMBER/distinctness of chains (mixture_width × chain_depth), not per-op magnitude (severity) — confirmed by EXP-053.
- **Stochastic strong augmentation** (`transforms.RandomApply([T], p)`): applying a strong transform to a random fraction of samples is a standard, effective regularization pattern (e.g. SimCLR-style pipelines). torchvision-native (no new dep). Lets the augmented subset receive a much RICHER transform than would be affordable uniformly.

## Experimental History Review

**Current best / baseline**: **96.34%** (EXP-052, AugMix w2,d1). Bar = 96.44. 7 lifetime improvements.

**Two AugMix results so far**:
- EXP-052 (+0.12): AugMix(w2,d1) multi-chain mixing replacing TrivialAugment → diverse mixing is the live lever (2nd confirmation after EXP-012).
- EXP-053 (−0.05, no-improvement): AugMix severity 3→6 → op MAGNITUDE is interior-optimal (default near-best); lowered loss, top-1 flat. **Sharpened insight: diversity = chain COUNT, not magnitude.**

**The tension** (goal-learnings § Protocol Findings + § Failed Approaches): the actual diversity dial (chain count: w2,d2/w3) is wall-INFEASIBLE at 8 workers — w2,d2 ~670s, w3 ~792s, both breach the 600s limit (budget gates on Σdt so epochs survive, but wall balloons). w2,d1 (~12.6ms/batch) is the feasible uniform ceiling. To push chain-count diversity, throughput must be recovered or the rich config applied to only a SUBSET (avg CPU cost ≤ ~13ms/batch).

**Closed augmentation sub-levers**: strength/Cutout (EXP-013/021), policy swap (EXP-014), label-mixing (EXP-011/018), cooldown (EXP-033/34/35), border (EXP-037), occlusion-pattern (EXP-048), AugMix magnitude (EXP-053). **Live/untried: chain-COUNT diversity delivered to a subset (stochastic full-strength AugMix).**

## Candidate Ideas

### 1. Intermittent full-strength AugMix via RandomApply (p≈0.5) — deliver the rich w3 config to a subset
**Summary**: Replace `AugMix(mixture_width=2, chain_depth=1)` with `transforms.RandomApply([transforms.AugMix()], p=0.5)` — apply the FULL default AugMix (mixture_width=3, chain_depth=-1, the literature-validated config) to ~50% of images; the other ~50% get only RandomCrop+Flip(+GPU Cutout). Keeps everything else fixed.
**Reasoning**: EXP-053 showed the diversity lever is chain COUNT, and EXP-052 showed AugMix mixing works — but the rich w3 config is wall-infeasible uniformly. Stochastic application is the feasible way to expose training to genuine 3-chain diversity: the augmented half sees much richer/more-diverse samples than w2,d1-on-all, at an average CPU cost (~0.5×21ms + 0.5×~3.5ms ≈ 12ms/batch) that fits the 600s wall. Directly targets the live lever.
**Sources**: reports/exp-report-053.md § Next Steps (lead); exp-report-052.md § Unexplored Avenues; torchvision `RandomApply`; goal-learnings § Protocol Findings (≲13ms/batch envelope).
**Estimated Effort**: low-medium — one-line wrap in RandomApply; REQUIRES a dataloader feasibility probe (the w3 half costs ~21ms each → avg must land ≲13ms/batch; tune p down to 0.45 if needed).
**Risk Assessment**: TWO risks, both graceful. (a) **Coverage reduction** — ~half the images get NO photometric/geometric aug; "aug on every image" has been the working regime (w2,d1-on-all worked), so fewer-but-richer may not beat all-but-weaker → within-noise null or mild regression. (b) **Confounded** — coverage↓ and per-image-diversity↑ move together. No crash risk; feasibility gated by probe. This is the most direct feasible test of the live chain-count lever.

### 2. Replicate EXP-052 (AugMix w2,d1, unchanged) — confirm 96.34 sits above the ±0.25pp noise band
**Summary**: Re-run the exact EXP-052 config (no code change) to check the +0.12pp gain reproduces above noise before investing more loops on the AugMix direction.
**Reasoning**: EXP-052's +0.12pp is near the documented ±0.25pp noise band; EXP-053 landed −0.05 below it. A replication de-risks the foundation. Honest, zero-implementation-risk.
**Sources**: goal-learnings § High Importance (epoch/throughput jitter → sub-0.2pp deltas are noise); exp-report-052.md § Review Notes.
**Estimated Effort**: trivial — re-run current `train.py` (no change). But note: same seed 42 → largely deterministic, so a "replication" mostly re-confirms the same run, not a fresh noise draw (throughput jitter gives some variation in epoch count). Limited new information.
**Risk Assessment**: No metric upside (cannot exceed baseline — it IS the baseline config) → would record as no-improvement by construction. Diagnostic value only. NOT a goal-advancing experiment.

### 3. AugMix mixing-weight `alpha` tuning (Dirichlet/Beta concentration) — CPU-neutral knob on the mix distribution
**Summary**: On AugMix(w2,d1), change `alpha` (default 1.0) — controls the Dirichlet/Beta concentration of the convex mix weights. Lower alpha pushes mass toward extreme mixes (nearer pure-clean or pure-augmented); higher alpha toward even blends.
**Reasoning**: A CPU-neutral, all-coverage AugMix knob distinct from severity — changes the DISTRIBUTION of augmentation strength per sample, a different diversity dimension.
**Sources**: Hendrycks et al. 2020 (alpha); torchvision AugMix(alpha=).
**Estimated Effort**: low — one keyword. CPU-neutral (certain-feasible).
**Risk Assessment**: Likely second-order, like severity (EXP-053) — a mix-weight knob, not chain count. Low-moderate EV; held as a cheap alternate if Candidate 1 proves infeasible.

## Idea Evaluation

EXP-053 established the key direction: the live lever is chain-COUNT diversity, and the only feasible way to push it (chains are wall-infeasible uniformly) is to deliver the rich config to a subset.

**Mechanism / evidence**: Candidate 1 (RandomApply full AugMix) is the only candidate that targets the LIVE lever (chain count) — it exposes training to genuine 3-chain diversity within the wall. Candidate 3 (alpha) is another magnitude-like knob (likely second-order, cf. severity null). Candidate 2 (replicate) advances nothing — by construction it cannot beat baseline.

**Risk**: Candidate 1 fails gracefully (coverage-loss null/mild regression), gated by a feasibility probe. Candidate 3 is safe but low-EV. Candidate 2 has zero upside.

**Expected impact**: Candidate 1 has the only real shot at clearing +0.1 by pushing the validated chain-count lever; its risk (coverage) is the open question worth resolving. Candidate 3 is incremental; Candidate 2 is diagnostic.

**Decision**: Lead with **Candidate 1 (RandomApply full-strength AugMix, p≈0.5)** — the most direct feasible test of the live chain-count diversity lever. Candidate 3 (alpha) is the cheap CPU-neutral alternate; Candidate 2 (replication) is deferred (diagnostic, no goal upside).

## Chosen Idea
**Selected**: Candidate 1 — `RandomApply([AugMix()], p=0.5)` (full default w3 AugMix on ~50% of images) replacing AugMix(w2,d1), keeping Cutout.

**Why this idea**:
EXP-052 proved AugMix multi-chain mixing lifts top-1; EXP-053 proved the lever is chain COUNT, not magnitude — but the rich w3 config is wall-infeasible applied uniformly. Stochastic application (RandomApply) is the feasible way to expose training to genuine 3-chain diversity: the augmented subset gets much richer samples than w2,d1-on-all, at an average CPU cost (~12ms/batch) that fits the 600s wall. It is the single most direct test of the live, validated lever.

**Hypothesis**:
`RandomApply([AugMix()], p=0.5)` keeps dt ~8ms (GPU step unchanged) and average dataloader ~12ms/batch → wall < 600s, ~91 ep (feasibility probe + early real-load check gate this). IF richer 3-chain diversity on half the images regularizes better than uniform w2,d1, best_test_acc ≥ 96.44. Falsified if within ±0.25pp of 96.34 (the coverage reduction cancels the diversity gain — confirming uniform coverage matters more than per-image chain depth at this budget) or if the probe shows wall > 600s (retry at lower p or record infeasibility).

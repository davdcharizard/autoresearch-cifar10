# Brainstorm EXP-065
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (none consulted — no new external technique. After 65 experiments every literature-sourced lever is closed. This loop completes the label-smoothing bracket and probes the last untested BN hyperparameters. Label smoothing (Szegedy 2016) and its interaction with strong augmentation are standard, well-understood; no high-signal source adds beyond the project's own EXP-023.)
- **CODEBASE FINDING (prepare.py, the FROZEN eval harness)**: `Eval.__init__` hardcodes `mean=(0.4914,0.4822,0.4465), std=(1,1,1)` for the test transform. This KILLS the brainstorm-064 "per-channel std normalization" idea: changing train.py's std would scale train inputs differently from the frozen-eval test inputs → a train↔eval distribution mismatch (invalid/regression). The std=(1,1,1) (mean-subtract only, no std-divide) is a deliberate harness design constant that BOTH train and eval must share. Std-normalization avenue CLOSED.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + AugMix-p0.5 + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile, 91 ep, dt 8ms.
- **65 experiments, 8 improvements. The plateau is mapped across EVERY major lever** (project-insights High): augmentation (all sub-axes + both delivery paths), capacity (×4 directions), optimizer (family/grad-dynamics/objective, + gradient clipping EXP-064), LR (peak/shape/warmup), normalization-as-regularizer, eval-BN, residual scaling, head, batch, activation, regularizers, weight-averaging, throughput→epochs. Both near-miss combinations closed (EXP-049, EXP-063).
- **Label smoothing — only the LOWER direction tested**: EXP-023 (LS 0.1→0.05) regressed −0.19pp to 96.03 ("0.1 near-optimal"). The conclusion "0.1 near-optimal" was drawn from a SINGLE-SIDED probe — the HIGHER direction (0.15, 0.2) is genuinely untested. EXP-023 ran on the OLD TrivialAugment recipe (commit 6c417a4), NOT the current AugMix-p0.5 best.
- **Genuinely UNTESTED knobs remaining** (all low-ceiling): label smoothing UPPER direction (0.15); BN momentum (0.1, never tuned); BN eps (1e-5, never tuned). Std-normalization now also closed (frozen eval, above).

## Candidate Ideas

### 1. Higher label smoothing (LS 0.1 → 0.15) — complete the LS bracket on the AugMix recipe
**Summary**: Raise `LABEL_SMOOTHING` from 0.1 to 0.15 (train.py L27). Single-variable, all else byte-identical to EXP-054. Probes the UPPER side of the LS optimum, which EXP-023 (lower side) left untested, on the current best recipe.

**Reasoning**: LS is the one regularizer ALREADY in the recipe whose strength was only probed downward (EXP-023: 0.05 hurt → optimum is at/above 0.1). The upper side is a genuine open question, and there is a specific mechanism for it to differ on THIS recipe: the current best uses heavy AugMix (multi-chain convex mixing produces soft, distribution-shifted images on 50% of the batch). Strong/mixing augmentation and label softness interact — softer targets can better match the softened, mixed inputs (this is why Mixup pairs with soft targets). EXP-023's "0.1 optimal" was concluded on the OLD TrivialAugment recipe (single-op, harder images), so the optimum may sit higher under AugMix. If 0.15 helps, it clears the bar; if it regresses, it cleanly brackets the LS optimum at 0.1 (closing the axis from both sides).

**Sources**: train.py L27 (LABEL_SMOOTHING), L242-244 (cross_entropy with label_smoothing); EXP-023 (lower-direction probe); project-insights Medium (adding regularizers hurts — but this RETUNES an existing one, not adds a new penalty).

**Estimated Effort**: Trivial (one constant). Compute-neutral, throughput-neutral, params unchanged, cudagraph-safe (LS is a host-side scalar in F.cross_entropy, outside the compiled forward).

**Risk Assessment**: Low. Failure mode no-improvement. project-insights Medium ("adding regularizers hurts at this short budget") leans against, and EXP-023's lower-side regression suggests 0.1 may already be the peak → a small regression is the most likely outcome. But LS is a RETUNE (not a new penalty), and the AugMix-interaction mechanism is a real, untested reason the optimum could differ. No scope/wall/throughput risk.

### 2. BN momentum reduction (0.1 → 0.05)
**Summary**: Set `momentum=0.05` on all `BatchNorm2d` constructors (longer EMA window for running stats). Single-variable.

**Reasoning**: Under heavy AugMix the per-batch BN stats are noisy; a longer EMA window gives eval-time running stats with lower estimation variance over the (same, augmented) operating distribution — distinct from EXP-061's clean-recalib, which CHANGED the distribution. Reduces variance, keeps the trained-in augmented operating point.

**Sources**: train.py BN constructors (L71/75/83/103); EXP-061 (BN-stat operating point).

**Estimated Effort**: Trivial (momentum kwarg). Compute-neutral, cudagraph-safe (static arg).

**Risk Assessment**: Low-to-moderate, low-evidence. With cosine-to-0, the final epochs are near-frozen-weight, so the default-momentum running stats are already stable; a longer window would include slightly-higher-LR (staler) batches → could mildly hurt at eval. Near-noise, mild-regression-possible.

### 3. BN eps increase (1e-5 → 1e-3)
**Summary**: Set `eps=1e-3` on all `BatchNorm2d` constructors. Single-variable.

**Reasoning**: Larger eps shrinks the normalized output of low-variance channels (BN divides by sqrt(var+eps)), mildly down-weighting less-informative channels — a soft implicit regularization, untested.

**Sources**: train.py BN constructors; standard BN.

**Estimated Effort**: Trivial. Compute-neutral, cudagraph-safe.

**Risk Assessment**: Low, very-low-evidence. On well-activated k=4 channels eps 1e-5 vs 1e-3 is negligible for most channels → near-certain exact null.

## Idea Evaluation
- **Evidence strength**: all three are low-evidence micro-probes on an exhausted plateau (the honest state at experiment 65). Idea 1 has the most concrete grounding — a single-sided prior (EXP-023) it completes, plus a specific recipe-dependent mechanism (LS×AugMix-softness interaction) that makes the AugMix-recipe optimum genuinely open. Idea 2 is weakly contraindicated by the cosine-to-0/near-frozen-tail argument; Idea 3 is near-certain exact null.
- **Mechanism clarity**: Idea 1 clear (target softness ↔ soft mixed inputs); Idea 2 plausible-but-the-tail-is-already-stable; Idea 3 real-but-negligible-magnitude.
- **Expected impact**: all near-noise. Idea 1 is the only one with both an open prior and a recipe-specific reason to differ.
- **Risk profile**: Idea 1 safest and most informative (brackets the LS optimum either way — a genuine closure on a no-improvement). Ideas 2/3 lean toward null/mild-regression with less interpretive value.
- **Feasibility**: all trivial.
- **Conclusion**: Lead with **Idea 1 (LS 0.1 → 0.15)** — it completes a single-sided prior on the CURRENT recipe, has the clearest mechanism (LS×AugMix interaction), is trivial/compute-neutral/safe, and yields a clean axis-closure regardless of outcome. Honest expectation: near-noise null (likely a small regression confirming 0.1 is the peak), run per NEVER-STOP. BN-momentum/BN-eps are weaker fallbacks for later loops.

## Chosen Idea
**Selected**: Higher label smoothing (LABEL_SMOOTHING 0.1 → 0.15) on the AugMix-p0.5 best recipe.

**Why this idea**: It is the cleanest genuinely-untested probe remaining with a real, recipe-specific mechanism. EXP-023 probed LS only downward (0.05 hurt) and concluded "0.1 near-optimal" — but on the OLD TrivialAugment recipe and without testing the upper side. The current best uses heavy AugMix (soft, multi-chain-mixed images on 50% of the batch), and target softness interacts with input softness, so the LS optimum could genuinely sit higher under AugMix than it did under TrivialAugment. It retunes an existing regularizer (not a new penalty), is trivial/compute-neutral/cudagraph-safe, and brackets the LS optimum either way. With the std-normalization avenue now closed (frozen eval uses std=1) and every other lever mapped, this is the most defensible next probe.

**Hypothesis**: Raising LS to 0.15 will better match the softened AugMix-mixed inputs and raise best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). Given EXP-023's lower-side regression suggests 0.1 is at/above optimum and the "adding regularizers hurts at this budget" pattern, the most likely outcome is a within-noise null or small regression that brackets the LS optimum at 0.1 — but the upper side is genuinely untested on the AugMix recipe and the probe is trivial and clean.

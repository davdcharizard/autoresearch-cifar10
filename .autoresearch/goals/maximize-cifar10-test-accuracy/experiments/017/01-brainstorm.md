# Brainstorm EXP-017
**Created**: 2026-06-30

## Web Search & Literature Review
No fresh lit-search this loop — the per-goal knowledge base is mature and already covers the finalists' techniques. Grounding entries:
- **knowledge/references/ghost-batchnorm.md** — GhostBN mechanism + the EXP-016 throughput finding (ghosting breaks the fused channels_last BN kernel ~50%). Basis for idea-01's cheap surrogate.
- **knowledge/references/fast-cifar10-recipes.md** — DavidNet/hlb/airbench recipe lineage (pooling, whitening).
- **knowledge/references/torch-compile-throughput.md** — EXP-014 banked +12% throughput recipe (off-budget warmup), available to fund any per-step cost.
- **knowledge/references/rezero-identity-init.md** — near-identity-init discipline (EXP-004) reused by idea-03's SE init.
- External (textbook, not re-fetched): Zhang ICML 2019 anti-aliased CNNs / BlurPool (arXiv:1904.11486); Hu et al. CVPR 2018 SENet (arXiv:1709.01507); Hoffer 2017 GhostBN (arXiv:1705.08741).

## Experimental History Review
- **Current best: 96.38 (EXP-008)**, Cutout12+RandomErasing on the whitened ResNet-9 + ReZero + EMA + flip-TTA stack. **10 straight no-improvements since (EXP-006→016).**
- **What worked**: recipe/throughput (EXP-001 95.22), EMA+TTA (EXP-002 +0.50), whitening (EXP-003 +0.15), one ReZero block@layer2 (EXP-004 +0.13), stronger aug (EXP-008 +0.38).
- **What's saturated (do NOT retry)**: capacity/width (EXP-005/007/014 — generalization ceiling, not epoch-bound), optimizer swap (Muon EXP-009/010 ties), input-aug across all 3 mechanisms (occlusion/mixing/transform EXP-008/011/015), reg-scalars (wd-shaping/LS EXP-012), loss-geometry (tail-SAM EXP-013), eval-TTA (EXP-006).
- **#1 failure mode (project-insights High)**: any per-step compute/param cost trades against epochs at the fixed 300s TIME budget → under-anneal; literature gains quoted at matched epochs do NOT transfer. Read num_epochs first on every cost-adding change.
- **Freshest signal (EXP-016)**: layer3-only GhostBN BEAT its same-session control +0.24pp (96.38 vs 96.14) DESPITE 16 fewer epochs — the FIRST positive regularization signal on this goal. Missed the 96.48 bar only because ghosting breaks the fused BN kernel (~50% slower → under-anneal). Caveat: the c0 control drew low (96.14 vs typical ~96.3–96.5), so part of the +0.24 may be a weak-control draw; cA's absolute 96.38 = baseline.
- **Untried gaps**: (a) BN/activation-statistic noise applied THROUGHPUT-FREE (the EXP-016 mechanism without the tax); (b) inductive-bias changes that are NOT capacity — shift-invariance (BlurPool), channel-attention (SE); (c) schedule-shape (cosine/longer tail), throughput-free, flagged untried in EXP-012.

## Collected Ideas
- **Cheap BN-affine noise** (throughput-free GhostBN surrogate): perturb BN output per-(sample,channel) to emulate ghost-stat noise without breaking the fused kernel. [exploit history / EXP-016]
- **Anti-aliased BlurPool downsampling**: replace strided MaxPool with maxpool+binomial-blur subsample for shift-invariance. [literature inductive-bias / pivot]
- **Squeeze-Excitation channel attention**: content-adaptive per-channel recalibration on residual branches. [literature inductive-bias / pivot]
- **Compile-funded layer3 GhostBN**: use EXP-014 torch.compile +12% to run the FAITHFUL EXP-016 GhostBN at ~150ep. [combine near-misses A+B]
- **Schedule-shape change**: cosine anneal or longer low-LR tail / retune PCT_START (throughput-free). [orthogonal lever]
- **Different stem / patch-embedding**: replace the conv stem with a richer whitened patch embed. [representation; moonshot-ish]
- **Stochastic depth on the ReZero/residual blocks**: drop-path regularization (a noise mechanism distinct from BN-noise). [simplification/regularization]
- **DisturbLabel / per-batch label noise**: target-space stochastic regularizer (distinct from LS). [moonshot, low prior given EXP-012 LS saturation]

## Combinations
- **BlurPool + torch.compile**: compile (banked +12%) absorbs BlurPool's per-step blur cost so the shift-invariance gain is tested at full epochs (no under-anneal confound — the trap that sank EXP-016).
- **BN-noise + EMA (already in stack)**: EMA averages the noisy iterates → captures the regularization while cancelling eval-time variance (the same synergy GhostBN relied on; no extra code).
- **SE + near-identity init (ReZero discipline)**: init the SE gate ≈1 so the net starts bit-close to the proven recipe → clean single-variable capacity-free test, no LR retune.

## Candidate Ideas

### 1. Throughput-free BN-affine noise (cheap GhostBN surrogate)
**Summary**: Wrap `conv_bn`'s BN with a `NoisyBN` that runs standard FULL-batch fused BN (no speed loss) then, in training only, perturbs the output per-(sample,channel): `y = BN(x)·(1+σ_mul·ε1) + σ_add·ε2`, ε~N(0,1), shape [N,C,1,1]. Env `BN_NOISE=σ` (0=exact baseline). Same-session cells σ∈{0, 0.10, 0.20}. Full proposal: `proposals/idea-01.md`.
**What it targets**: The ~96.3–96.5 generalization ceiling via the ONE regularization axis with a positive signal (BN-stat noise, EXP-016) — but applied THROUGHPUT-FREE, removing the under-anneal tax (#1 failure mode) that capped EXP-016. Ghost-stat noise is algebraically a per-channel multiplicative+additive jitter of the normalization, which `NoisyBN` injects directly as one elementwise op (fused kernel preserved → ~150 epochs).
**Reasoning**: EXP-016 is the only loop in 10 to beat its same-session control (+0.24pp at fewer epochs). Decoupling its mechanism from its throughput cost is the highest-information, lowest-risk exploit. project-insights High (under-anneal trap; ghost breaks fused kernel) + experiments/016/04-analysis.md.
**Sources**: experiments/016/04-analysis.md; knowledge/references/ghost-batchnorm.md; Hoffer 2017 arXiv:1705.08741; project-insights High.
**Estimated Effort**: low.
**Risk Assessment**: post-BN affine jitter ≈ but ≠ exact ghost-stat perturbation → could tie (mechanism redundant at full epochs, implying EXP-016's +0.24 was a weak-c0 draw) or under-fit at σ=0.2. Throughput-free → no under-anneal confound, so a tie is a clean negative. Worst case: another no-improvement, but it definitively closes the BN-noise question.

### 2. Anti-aliased (BlurPool) downsampling
**Summary**: Replace the stride-2 `MaxPool2d(2)` in layer1/2/3 with MaxBlurPool (dense max stride-1 → fixed binomial-blur conv stride-2) for shift-invariance. Frozen blur kernel (optimizer-excluded). Env `BLURPOOL`. Cells: off / all-pools / later-pools-or-lighter-kernel. Full proposal: `proposals/idea-02.md`.
**What it targets**: The generalization ceiling via an INDUCTIVE-BIAS change (shift-invariance), not capacity — sidestepping the capacity-saturation verdict (EXP-014) and the regularization-axis saturation. Strided pooling aliases; anti-aliasing is a documented +0.5–1pp CIFAR generalization lever at matched epochs.
**Reasoning**: The dominant strategic read (project-insights High, EXP-014) is "pivot to a different mechanism." BlurPool is the cheapest well-grounded such pivot — ~0 added params, throughput-light, composes with whitening+ReZero+EMA. Zhang ICML 2019.
**Sources**: Zhang arXiv:1904.11486; knowledge/references/fast-cifar10-recipes.md; project-insights High; EXP-014 compile recipe.
**Estimated Effort**: medium.
**Risk Assessment**: the blur's per-step cost may cut epochs <142 → under-anneal (mitigation ladder: lighter 2-tap kernel / later-pools-only / torch.compile fund). Gain may not transfer at 150ep near ceiling. Interaction with the MaxPool(4) head.

### 3. Squeeze-Excitation channel attention
**Summary**: Add SE blocks (global-avg-pool → FC↓ → ReLU → FC↑ → sigmoid → channel rescale, r=8/16) to the residual branches, near-identity-init (gate≈1). Env `SE_REDUCTION`. Cells: off / r=16 / r=8-or-later-only. Full proposal: `proposals/idea-03.md`.
**What it targets**: The generalization ceiling via content-adaptive per-channel recalibration — a different mechanism (lightweight attention) with near-zero params/FLOPs, distinct from raw capacity (saturated EXP-007/014). High value-per-param CIFAR lever.
**Reasoning**: A throughput-light representational change respecting the "different mechanism" mandate; near-identity init reuses the EXP-004 ReZero discipline for a clean single-variable test. Hu et al. CVPR 2018.
**Sources**: Hu et al. arXiv:1709.01507; knowledge/references/rezero-identity-init.md; project-insights High.
**Estimated Effort**: low-medium.
**Risk Assessment**: per-step cost cuts epochs (mitigated: SE is cheap / later-only); sigmoid gate interaction with ReZero α + EMA; gain may not transfer near ceiling; whitening may already make the net channel-efficient (SE redundant).

## Review
Cross-model (Codex) adversarial review in `01-idea-review.md`. Verdict: **Idea 1 (BN-affine noise) is the pick** — 8/10 evidence (the only candidate with direct positive evidence in THIS harness: EXP-016's BN-stat-noise signal, with the fused-kernel throughput tax as the sole blocker), 7/10 impact. Idea 2 (BlurPool) 5/6 — solid paper mechanism but shift/transform levers already mostly saturated here (translate-TTA sub-noise EXP-006, RandAugment tied EXP-015) + shape/throughput traps. Idea 3 (SE) 4/6 — capacity-adjacent against the EXP-014 saturation verdict + risky identity-init.

Top concerns + resolutions (folded into the chosen idea):
- **Surrogate ≠ exact GhostBN** (independent per-sample Gaussian jitter vs data-dependent, group-shared, scale-dependent ghost noise). RESOLVED: gate the noise FIRST to the proven EXP-016 site (layer3, `BN_NOISE_MIN_CH=512`) rather than all-site fixed σ; add one all-site arm only if throughput stays clean. Note σ-calibration-from-measured-stat-ratio as a refinement for the plan.
- **Residual under-anneal risk** (RNG draws at 10 BN sites can still cost steps). RESOLVED: keep the `num_epochs ≥142` rejection gate; throughput pre-smoke; layer3-first limits RNG sites to 3.
- **Both gates, not just same-session delta** (EXP-016's c0 drew low at 96.14; bar is absolute ≥96.48 AND >c0 by >0.1pp). RESOLVED: verification requires BOTH; a hairline win triggers a confirmation re-run.

## Idea Evaluation
Adopt the reviewer's pick (Idea 1) — it aligns with both the freshest positive evidence (EXP-016) and the #1-failure-mode discipline (throughput-free → no under-anneal). The two pivots (BlurPool, SE) are deferred as next-loop options if BN-noise ties. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Throughput-free BN-affine noise (cheap GhostBN surrogate) — refined to **layer3-first** per review (`proposals/idea-01.md`).

**Why this idea**:
It is the only EXP-017 candidate grounded in a positive result measured in THIS exact harness: EXP-016's layer3 GhostBN beat its same-session control by +0.24pp (at 16 FEWER epochs), and the sole blocker was the fused-kernel throughput tax that halved epochs. `NoisyBN` injects the algebraically-equivalent per-(sample,channel) normalization jitter as a single elementwise op on the fused-BN output → preserves ~150 epochs → decouples the mechanism (which showed signal) from the cost (which capped it). It is low-effort, throughput-free (dodging the #1 failure mode), and decisive either way: a win clears the bar; a tie at healthy epochs cleanly closes the BN-noise axis and reveals EXP-016's +0.24 as a weak-control draw, hardening the backbone-pivot mandate. The pivots (BlurPool/SE) lean on matched-epoch external literature with weaker local support and carry shape/throughput/init traps.

**Refinements folded in (from review)**:
- Primary cell gates noise to **layer3 only** (`BN_NOISE_MIN_CH=512`, the proven EXP-016 site); one all-site arm only if the throughput smoke confirms ~150 epochs.
- Keep `num_epochs ≥142` rejection gate + throughput pre-smoke.
- Verify BOTH gates (≥96.48 absolute AND >same-session c0 by >0.1pp); hairline → confirmation re-run.
- Plan may calibrate σ from measured full-batch-vs-ghost stat ratios instead of fixed σ (optional).

**Hypothesis**:
Per-(sample,channel) BN-affine Gaussian noise (σ≈0.1–0.2) applied at layer3 (and optionally all sites), composed with the existing EMA, reproduces the EXP-016 ghost-stat regularization at FULL ~150 epochs and lifts best_test_acc to ≥96.48, clearing the same-session control by >0.1pp. If all cells tie at healthy epochs/ep25, BN/activation-statistic noise is redundant with the existing stack at full epochs — EXP-016's +0.24pp was a weak-c0-draw artifact and the ceiling is not noise-movable, redirecting the next loop to an inductive-bias backbone pivot (BlurPool/SE).

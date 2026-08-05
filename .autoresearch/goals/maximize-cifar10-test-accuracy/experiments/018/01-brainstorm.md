# Brainstorm EXP-018
**Created**: 2026-06-30

## Web Search & Literature Review
- **Zhang 2019, "Making Convolutional Networks Shift-Invariant Again", ICML** (arXiv:1904.11486; ref impl github.com/adobe/antialiased-cnns): naive max-pool / strided-conv / avg-pool subsampling violates the Nyquist sampling theorem → aliasing → loss of shift-equivariance. Inserting a fixed low-pass **blur between the dense pool and the stride-2 subsample** (BlurPool) restores approximate shift-equivariance and — the key finding for us — **increases clean classification accuracy** across ImageNet architectures, with the paper explicitly framing anti-aliasing as **effective regularization**. Fixed binomial kernels (size 2–5); applies to all three downsampling types. No new params, no new deps (pure depthwise conv). → grounds idea-01.
- **Hu et al. 2018, "Squeeze-and-Excitation Networks", CVPR** (arXiv:1709.01507): content-adaptive per-channel recalibration (GAP→bottleneck MLP→sigmoid gate) adds cross-channel dependency modeling at <1% params/FLOPs; ~0.5–1pp ImageNet top-1 across backbones. → grounds idea-02.
- **fastai AdaptiveConcatPool2d / Lin 2014 Network-in-Network** (arXiv:1312.4400): global avg and global max pooling capture complementary statistics; concatenating both gives the classifier a richer readout than max-only. → grounds idea-03.

## Experimental History Review
- **Current best: 96.38 (EXP-008)**, baseline commit 07c3760. 17 experiments; **12 straight no-improvements (EXP-005, 006, 007, 009–017)** — only EXP-008 (aug) improved since EXP-004.
- **What worked**: ResNet-9/DavidNet + time-one-cycle (EXP-001 95.22), EMA+flip-TTA (EXP-002 +0.50), frozen ZCA whitening (EXP-003 +0.15), ReZero block@layer2 (EXP-004 +0.13), stronger aug Cutout12+RandomErasing (EXP-008 +0.38).
- **What's saturated (approach-specific nulls)**: capacity width/depth (EXP-005/007/014), optimizer Muon (EXP-009/010 tie), input-aug across ALL 3 mechanisms — occlusion/mixing/transform (EXP-008/011/015), reg-scalars wd-shaping+LS (EXP-012), loss-geometry SAM (EXP-013), epoch/throughput via torch.compile (EXP-014 flat), BN/activation-statistic noise GhostBN+NoisyBN (EXP-016/017 — closed: calibrated noise ties control exactly, EXP-016's +0.24 was a low-c0 artifact).
- **Decisive diagnosis (EXP-014, project-insights High)**: buying +12% epochs AND +capacity both gave flat accuracy → the ~96.3–96.5 plateau is a **genuine generalization ceiling for this architecture at 300s**, NOT epoch/throughput/capacity-bound. The repeated strategic mandate: **a different architectural INDUCTIVE BIAS**, not another within-architecture regularization/optimizer/capacity lever.
- **Untried gap**: the net's **downsampling operators** (`MaxPool2d(2)`×3 + `MaxPool2d(4)`) and **head pooling** have never been touched — naive max-pool is the classic aliasing / shift-variance weakness. **Channel-attention** (SE) and **richer readout pooling** are also untried representational mechanisms. These are the genuinely-different inductive-bias levers the mandate calls for.
- **Protocol constraints carried in**: same-session control mandatory (~0.1–0.2pp noise floor; stored 96.38 too weak); num_epochs is the first-class under-anneal diagnostic on ANY per-step-cost change; CPU-aug is epoch-free but the budget excludes loader wait; confirmation re-run mandatory on any apparent win (low-c0-draw lesson). All runs `CUDA_VISIBLE_DEVICES=1`.

## Collected Ideas
- (Literature/inductive-bias) **BlurPool anti-aliased downsampling** — fixed low-pass blur before each stride-2 subsample → shift-equivariance, "anti-aliasing as regularization."
- (Algorithm/representation) **Squeeze-Excitation channel attention** — content-adaptive per-channel gating in the residual branches.
- (Representation/readout) **AdaptiveConcatPool head** — global avg ⊕ max pooling into the classifier instead of max-only.
- (Simplification) Replace all pool/stride downsampling with stride-2 convs (more params/compute — rejected: under-anneal risk, EXP-005/007).
- (Regularization, outside-field) Manifold/feature-space mixup — rejected: still a regularization mechanism, axis saturated (EXP-011 mixing tie).
- (Optimizer) Lookahead/SWA-explicit wrapper — rejected: optimizer axis saturated (EXP-010), EMA already present.
- (Stem) Second-scale whitening / learnable dirac stem — deprioritized: whitening front-end already tuned (EXP-003), lower-EV than downsampling.
- (Moonshot) Anti-aliased BlurPool + SE + concat-head full backbone refresh — too many simultaneous changes for clean attribution; decompose into the leads first.

## Combinations
- **BlurPool + AdaptiveConcatPool (idea-01 + idea-03)**: anti-alias the intermediate subsamples AND enrich the final readout — both touch pooling, are complementary (intermediate shift-equivariance vs richer global statistics), and are throughput-light/free; natural cB rider if idea-01 wins or as a combined cell.
- **BlurPool + SE (idea-01 + idea-02)**: shift-equivariant features fed into adaptive channel gating — orthogonal mechanisms (spatial anti-aliasing × channel recalibration); reserve as a follow-up if either wins alone (avoid confounding two new mechanisms in one cell).

## Candidate Ideas

### 1. BlurPool anti-aliased downsampling (MaxBlurPool)
**Summary**: Replace each `nn.MaxPool2d(2)` (layer1/2/3) with MaxBlurPool — dense max at stride 1, then a fixed binomial **blur** (depthwise, stride 2) to subsample; optionally blur the final 4×4 head pool too. Fixed kernel = a registered buffer (no params, no new deps; pure `F.conv2d` with `groups=C`). `BLUR_KSIZE`/`BLUR_FINAL` env toggles. See `proposals/idea-01.md`.
**What it targets**: The generalization ceiling (project-insights High, EXP-014) via the one structural weakness never touched — the aliasing, shift-variant `MaxPool` subsampling. Restores approximate shift-equivariance, which Zhang shows acts as effective regularization that *raises clean accuracy*, lifting the ceiling without adding capacity (no epoch cost from params).
**Reasoning**: Zhang 2019 (ICML) — anti-aliased downsampling increased ImageNet accuracy across architectures; it is the canonical inductive-bias change for convnets and exactly the "different mechanism" class the strategic mandate calls for. Standard convs (no fused-kernel break, unlike GhostBN EXP-016) → expected near-throughput-free.
**Sources**: `proposals/idea-01.md`; arXiv:1904.11486; adobe/antialiased-cnns; project-insights High (EXP-014 ceiling); train.py:149-152.
**Estimated Effort**: low-medium.
**Risk Assessment**: throughput from 3 depthwise blurs at full spatial (mitigated: standard conv, layer-restriction ladder, num_epochs ≥135 gate); CIFAR's small images + strong aug + flip may make the shift-equivariance gain small/within-noise (honest prior — Zhang's gains are ImageNet); over-smoothing the head pool (cA keeps head unblurred).

### 2. Squeeze-Excitation channel attention
**Summary**: Insert a lightweight SE block (GAP → `C→C/r→C` bottleneck → sigmoid → per-channel rescale) into the residual branches at layer2/3 (inside the ReZero α-gate to preserve identity-init). `SE_RATIO`/`SE_MIN_CH` env. See `proposals/idea-02.md`.
**What it targets**: The generalization ceiling via a modeling capability absent from all saturated axes — content-adaptive cross-channel dependency. Adds a new *function form* (channel attention), not raw width (width saturated EXP-007/014), at <1% params and near-zero compute.
**Reasoning**: Hu et al. 2018 (CVPR) — consistent ImageNet gains at negligible cost; one of the few accuracy levers that does not add width/depth, so it sidesteps the under-anneal trap (EXP-005/007/013).
**Sources**: `proposals/idea-02.md`; arXiv:1709.01507; knowledge/references/rezero-identity-init.md; train.py:109-137.
**Estimated Effort**: low-medium.
**Risk Assessment**: SE's gains are ImageNet-scale and may not transfer to a 7.8M-param CIFAR net with heavy aug+EMA (within noise); per-block GAP could add a sync stall (mitigated: layer3-restriction, num_epochs gate); SE inside the ReZero branch must preserve identity-init (verified by smoke).

### 3. AdaptiveConcatPool head (avg ⊕ max)
**Summary**: Replace `MaxPool2d(4)`+fc(512→10) with global-avg ⊕ global-max concat → fc(1024→10). `HEAD_POOL` env (max/avgmax/avg). Throughput-free; only fc input grows 512→1024 (+5,120 params). See `proposals/idea-03.md`.
**What it targets**: The generalization ceiling at the **readout** — max-only pooling discards the spatial-average signal; concat gives the linear classifier complementary, lower-variance global statistics. Genuinely throughput-free (sidesteps the #1 failure mode entirely).
**Reasoning**: fastai AdaptiveConcatPool / Lin 2014 NiN — avg and max pooling are complementary; the DavidNet lineage's max-only head is a convention, not a tuned choice. Cheapest, safest probe of whether the head leaves accuracy on the table.
**Sources**: `proposals/idea-03.md`; fastai layers docs; arXiv:1312.4400; train.py:152-178.
**Estimated Effort**: low.
**Risk Assessment**: smallest upside — a linear readout change may only shuffle ~0.1pp (deep net's final features may already be max-separable); SCALE_OUT may need a retune for the concat magnitude (cB rider); likely best as a free rider on idea-01/02 rather than a standalone win.

## Review
Cross-model (Codex) adversarial review in `01-idea-review.md`. Scored verdict: **idea-01 BlurPool 7.5/10** (evidence 8, impact 7) > idea-02 SE 6/10 > idea-03 ConcatPool 4.5/10. Pick: **idea-01**.

Top concerns + resolutions (folded into the Chosen Idea below; the rest carry into the plan):
1. **BlurPool is the only finalist that directly attacks the diagnosed structural limiter** (untouched aliasing `MaxPool` downsampling) → adopt as lead.
2. **Main risk is hidden throughput + padding/phase correctness, not scope** — small grouped convs + `F.pad` can be kernel-launch/memory-bound. → Resolution: hard `num_epochs ≥ ~135` gate; shape/kernel-sum/memory-format smokes; verify padding/phase against the Zhang/Adobe pattern (not just output-shape equality); **precompute the dtype-matched blur buffer once** (avoid per-forward `.to(x.dtype)` allocation).
3. **Do NOT blur the final 4×4 head pool in the primary cell** — under-specified, over-smoothing/shape risk. → Resolution: primary cA blurs only layer1/2/3, leaves `MaxPool2d(4)` unchanged; the second operating point cB is a **lighter ksize=2 rect filter** (not a head blur). Head-blur deferred to a follow-up.
4. **idea-02 SE underspecified around init/residual scaling** at the un-gated `Residual(512)` (vanilla sigmoid SE shrinks the branch at init). → Not selected; if pursued later, identity-init the SE gate (zero-init final projection or `2·sigmoid`).
5. **All finalists require same-session control + confirmation** (stored 96.38 too near the noise floor; EXP-016/017 low-c0 lesson). → Resolution: c0 same-session anchor; win = cell ≥96.48 AND >c0+0.1pp, **confirmation re-run mandatory** on any apparent win.

## Idea Evaluation
Adopting the reviewer's pick (idea-01 BlurPool) — it aligns with the standing strategic mandate (project-insights High, EXP-014/017: pivot to a different architectural inductive bias) and is the only finalist that changes the feature extractor's structural prior rather than tweaking the readout (idea-03, "too shallow for the stated limiter") or adding channel-recalibration capacity into an already-saturated net (idea-02, "may sit inside the ~0.1pp noise floor"). No override. Full scored critique in `01-idea-review.md`; concerns summarized in ## Review.

## Chosen Idea
**Selected**: idea-01 — BlurPool anti-aliased downsampling (MaxBlurPool) at layer1/2/3.

**Why this idea**:
The diagnosis is a generalization ceiling (~96.3–96.5) for this architecture at 300s — EXP-014 proved it is not epoch/throughput/capacity-bound, and every regularization/optimizer axis is saturated (12 straight no-improvements). The repeated mandate is a *different architectural inductive bias*. The net's naive `MaxPool2d` downsampling (`train.py:149-152`) is the one structural weakness never touched — it aliases (violates Nyquist) and makes the net shift-variant. BlurPool (Zhang 2019, ICML) is the canonical, literature-validated fix that restores approximate shift-equivariance AND raises clean accuracy ("anti-aliasing as effective regularization"), using a fixed blur kernel (no params, no new deps, standard convs that do not break the fused BN kernel — unlike GhostBN EXP-016). The Codex reviewer scored it highest (7.5/10) as the clearest inductive-bias change attacking the actual limiter. Refined per review: primary cell = MaxBlurPool at layer1/2/3 with ksize=3 and the final 4×4 head pool left unchanged; precompute the blur buffer in the activation dtype; hard num_epochs ≥~135 gate with a layer1/2-only fallback if under-anneal; same-session c0 control + mandatory confirmation re-run on any apparent win.

**Hypothesis**:
Replacing the aliasing `MaxPool2d(2)` subsampling at layer1/2/3 with anti-aliased MaxBlurPool (fixed binomial blur, stride-2) restores approximate shift-equivariance and, per Zhang's anti-aliasing-as-regularization effect, lifts best_test_acc to ≥96.48 over the same-session control by a clear >0.1pp margin at near-full epochs (≥~135), replicated on a confirmation re-run. If it ties at healthy epochs/ep25, the downsampling inductive bias is also not ceiling-moving on this small-image, heavily-augmented net at 300s, and the residual headroom (if any) lies in the stem/representation or is a genuine data/architecture limit — further narrowing the search.

# Brainstorm EXP-019
**Created**: 2026-06-30

## Web Search & Literature Review
(No new external search this loop — the candidate space was already grounded by the EXP-018 brainstorm's literature review; reusing those sources.)
- **Hu et al. 2018, "Squeeze-and-Excitation Networks", CVPR** (arXiv:1709.01507): content-adaptive per-channel recalibration (GAP→bottleneck MLP→sigmoid gate); ~0.5–1pp ImageNet gains at <1% params/FLOPs. → idea-01.
- **Loshchilov & Hutter SGDR / Smith one-cycle** (arXiv:1608.03983): cosine vs linear anneal shape changes which minimum SGD selects; the low-LR tail dominates final generalization. → idea-02.
- **fastai AdaptiveConcatPool2d / Lin 2014 NiN** (arXiv:1312.4400): avg and max pooling are complementary readout statistics. → idea-03.

## Experimental History Review
- **Current best: 96.38 (EXP-008)**, commit 07c3760. 18 experiments; **13 straight no-improvements (EXP-005, 006, 007, 009–018)** — only EXP-008 (aug) improved since EXP-004.
- **Saturated axes (approach-specific nulls)**: capacity width/depth (005/007/014), optimizer Muon (009/010), input-aug ALL 3 mechanisms occlusion/mixing/transform (008/011/015), reg-scalars wd+LS (012), loss-geometry SAM (013), epoch/throughput via torch.compile (014 — flat), BN/activation-statistic noise GhostBN+NoisyBN (016/017 — closed), and **the downsampling-operator inductive bias BlurPool (018 — lost −0.08/−0.15pp, monotonically worse with blur; strong RandomCrop+flip aug already supplies translation invariance, 32×32 has little aliasing)**.
- **Decisive diagnosis (EXP-014, project-insights High; reinforced 016/017/018)**: buying +12% epochs AND +capacity gave flat accuracy → the ~96.3–96.5 plateau is a **genuine generalization ceiling** for this whitened ResNet-9 at 300s, NOT epoch/throughput/capacity-bound. Even axes outside regularization (BN-noise, anti-aliasing) now tie. Mandate: a genuinely different MECHANISM, throughput-free (under-anneal is the #1 failure mode behind 5+ nulls).
- **Untried gaps**: (a) **channel attention** (SE) — a different *functional form*, never tried; (b) **schedule SHAPE** (cosine/extended-tail one-cycle vs the EXP-001 linear triangular) — EXP-012 EXPLICITLY flagged this as "an untried lever with ceiling clearly above noise"; throughput-free; (c) **readout pooling** (avg⊕max concat head vs max-only) — cheap, untried. SE and ConcatPool were the EXP-018 deferred finalists (idea-02 6/10, idea-03 4.5/10).
- **Protocol carried in**: same-session control mandatory (~0.1–0.2pp noise floor, stored 96.38 too weak); hard num_epochs≥135 under-anneal gate; mandatory confirmation re-run on any apparent win (low-c0-draw lesson EXP-016/017); anti-gaming integrity (summary==per-epoch max); all runs `CUDA_VISIBLE_DEVICES=1`.

## Collected Ideas
- (Algorithm/representation) **Squeeze-Excitation channel attention** — content-adaptive per-channel gating in residual branches.
- (Optimization/schedule) **One-cycle schedule shape** — cosine decay / extended low-LR tail vs linear triangular (throughput-free; EXP-012-flagged).
- (Representation/readout) **AdaptiveConcatPool head** — avg⊕max global pooling vs max-only.
- (Regularization) Stochastic depth / DropPath on residual branches — rejected: only 3 residual blocks (little to drop), and stochastic regularization just gave two nulls (BN-noise 016/017).
- (Stem) 5×5 whitening / second-scale stem — deprioritized: whitening already tuned (003), lower-EV.
- (Optimizer) Lookahead/SWA wrapper — rejected: optimizer axis saturated (010), EMA already present.
- (Eval) Richer TTA — rejected: eval-side saturated (006).
- (Moonshot) Wholesale different backbone funded by compile — high-risk/large; defer until the cheap genuinely-different mechanisms (SE/schedule/head) are exhausted.

## Combinations
- **SE + AdaptiveConcatPool (idea-01 + idea-03)**: channel-attention features feeding a richer readout — orthogonal (feature recalibration × pooling statistics); reserve as a follow-up if either wins alone (avoid confounding two new mechanisms in one cell).
- **Schedule-shape + winner (idea-02 + idea-01/03)**: a throughput-free schedule tweak composes with any architectural winner as a free rider in a later loop.

## Candidate Ideas

### 1. Squeeze-Excitation channel attention
**Summary**: Insert lightweight SE blocks (GAP→`C/r` bottleneck→sigmoid gate→per-channel rescale) into the residual branches at layer2/3, inside the ReZero α-gate to preserve identity-init; zero-init the SE output (0.5 gate) for the un-gated blocks. `SE_RATIO`/`SE_LAYERS` env. See `proposals/idea-01.md`.
**What it targets**: The generalization ceiling via a modeling capability absent from all saturated axes — content-adaptive cross-channel dependency (a new *functional form*, not raw width which is saturated EXP-007/014), at <1% params and near-zero compute.
**Reasoning**: Hu et al. 2018 — consistent ImageNet gains at negligible cost; one of the few accuracy levers that adds no width/depth, sidestepping the under-anneal trap. The strongest genuinely-different *architectural* mechanism remaining (EXP-018 reviewer ranked it 2nd at 6/10, behind only the now-disproven BlurPool).
**Sources**: `proposals/idea-01.md`; arXiv:1709.01507; EXP-018 idea-02 + idea-review; project-insights High; train.py:109-137.
**Estimated Effort**: low-medium.
**Risk Assessment**: SE's gains are ImageNet-scale; on a small heavily-augmented CIFAR net may sit in the ~0.1pp noise floor; per-block GAP sync stall (mitigated: layer3-restriction, num_epochs gate); init must preserve the validated recipe (zero-init fc2 / 2·sigmoid).

### 2. One-cycle schedule shape (cosine / extended tail)
**Summary**: Add a `SCHEDULE` env varying only the post-warmup LR decay shape — `tri` (current linear), `cos` (cosine), `tail` (extended low-LR floor in the final 20%). Throughput-free scalar change. See `proposals/idea-02.md`.
**What it targets**: The generalization ceiling at the **anneal tail** — EXP-001 showed most accuracy lands in the low-LR tail, so the decay shape plausibly selects a different (flatter/better) minimum. EXP-012 explicitly flagged schedule-shape as an untried lever with ceiling above noise.
**Reasoning**: Throughput-free (no under-anneal risk — the failure mode behind 5+ nulls); the one validated-as-promising lever never pulled; cosine/SGDR is standard and changes minimum selection.
**Sources**: `proposals/idea-02.md`; EXP-012 04-analysis.md (flag); project-insights Medium (tail dominates); arXiv:1608.03983; train.py:282-290.
**Estimated Effort**: low.
**Risk Assessment**: a well-tuned triangular one-cycle is already strong → most-likely-to-tie of the finalists, but genuinely untried and zero-cost; tail floor could mildly under-anneal the end (mitigated: low floor + EMA).

### 3. AdaptiveConcatPool head (avg ⊕ max)
**Summary**: Replace MaxPool(4)+fc(512→10) with avg⊕max concat→fc(1024→10). `HEAD_POOL` env (max/avgmax/avg). Throughput-free; +5,120 params. See `proposals/idea-03.md`.
**What it targets**: The generalization ceiling at the **readout** — max-only pooling discards the spatial-average; concat gives the classifier complementary, lower-variance statistics. Genuinely throughput-free.
**Reasoning**: fastai/NiN — avg and max are complementary; the DavidNet max-only head is convention, not tuned. Cheapest, safest probe of whether the head leaves accuracy on the table.
**Sources**: `proposals/idea-03.md`; EXP-018 idea-03 + idea-review (4.5/10); arXiv:1312.4400; train.py:152-178.
**Estimated Effort**: low.
**Risk Assessment**: smallest upside — a linear readout change may only shuffle ~0.1pp; SCALE_OUT may need a minor retune; likely best as a rider on a stronger change.

## Review
Cross-model (Codex) review in `01-idea-review.md`. Scored: **idea-01 SE 6.5/5.5** > idea-02 schedule 6/4.5 > idea-03 ConcatPool 4.5/3. Pick: **idea-01 SE** — "the only finalist that adds a materially new functional form while staying throughput-neutral."

Top concerns + resolutions:
1. **All three are plausible but small; the prior is sub-noise given 13 nulls** — acknowledged; SE has the best upside as the only genuinely-new modeling mechanism. The honest modest-EV framing is retained.
2. **SE init must preserve the validated recipe — use identity-preserving `2*sigmoid` gates, NOT a 0.5-gate** (the un-gated `Residual(128)`/`Residual(512)` blocks are not ReZero-protected, so a 0.5 gate scales the branch at init and disturbs the recipe). → Resolution: SE gate = `2*sigmoid(fc2(relu(fc1(GAP(x)))))` with `fc2` zero-init → gate starts at exactly 1.0 (identity) everywhere; verify ep25 not depressed.
3. **idea-02 `tail` variant under-anneals** (holds LR at 0.05·PEAK=0.02, not ~0, contradicting the complete-the-anneal lesson) — moot (not picked); noted for any future schedule experiment (prefer `cos`, finish at 0).
4. **idea-03 too shallow** for the ceiling (readout over a saturated representation) — agreed; deferred as a later rider.
5. **Same-session control + mandatory confirmation re-run** on any apparent win — adopted (low-c0-draw lesson, EXP-016/017).

## Idea Evaluation
Adopting the reviewer's pick (idea-01 SE). It is the strongest remaining genuinely-different architectural mechanism — content-adaptive channel attention, orthogonal to every saturated axis (width/depth/optimizer/regularization/BN-noise/downsampling) — and stays throughput-neutral, avoiding the under-anneal trap. Schedule-shape (idea-02) and ConcatPool (idea-03) are kept as documented next-loop options (idea-02 needs the `tail`→0 fix). No override. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: idea-01 — Squeeze-Excitation channel attention at layer2+layer3 residual branches.

**Why this idea**:
The diagnosis is a robust generalization ceiling (~96.3–96.5; 13 straight nulls, EXP-014 disproved the epoch/throughput/capacity framing, and EXP-016/017/018 closed BN-noise and the downsampling inductive bias). SE is the only remaining lever that adds a *new functional form* — content-adaptive per-channel recalibration conditioned on global image content — rather than re-tuning capacity, the optimizer, regularization strength, or the readout. It costs <1% params and near-zero compute (a GAP + two 1×1 convs on a [N,C,1,1] tensor), so it stays throughput-neutral and avoids the under-anneal failure mode behind 5+ prior nulls. The Codex reviewer scored it highest (6.5/10) as the best chance at a real ≥96.48 signal vs another schedule/readout tie. Refined per review: SE at layer2+layer3 with identity-preserving `2*sigmoid` gates (fc2 zero-init → gate=1.0 at init, recipe-neutral), `num_epochs ≥ 135` hard gate, same-session c0 + mandatory confirmation re-run.

**Hypothesis**:
Adding content-adaptive channel attention (SE, r=16) to the layer2/3 residual branches lifts best_test_acc to ≥96.48 over the same-session control by a clear >0.1pp margin at near-full epochs (≥135), replicated on a confirmation re-run. If it ties at healthy epochs/ep25, channel-attention is redundant with the existing representation on this 7.8M-param net at 300s — leaving schedule-shape (idea-02, cos) and the readout (idea-03) as the remaining cheap probes before the ceiling is declared a genuine architecture+data limit.

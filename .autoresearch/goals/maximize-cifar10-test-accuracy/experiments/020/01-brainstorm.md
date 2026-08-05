# Brainstorm EXP-020
**Created**: 2026-06-30

## Web Search & Literature Review
- **MosaicML LR-schedule benchmark** (https://cameronrwolfe.substack.com/p/the-best-learning-rate-schedules): cyclic/one-cycle schedules "do not necessarily lead to improved accuracy when compared to cosine decay … in many instances the cyclic tradeoff curve underestimated the standard [cosine] tradeoff curve by a margin of 0.5% validation accuracy," and "results held for CIFAR-10 as well." Our schedule is the cyclic/LINEAR-decay kind → cosine could plausibly be ~0.5% better. **Strongest external evidence for any cheap lever on this goal.** → idea-01.
- **fastai one-cycle uses COSINE annealing** for the curve, not linear — our EXP-001 linear-triangular decay is the less-standard variant. SGDR (Loshchilov, arXiv:1608.03983) + Bag-of-Tricks (He et al. CVPR 2019, arXiv:1812.01187): cosine is the standard research-grade CIFAR schedule. → idea-01.
- **hlb-CIFAR10 (tysam) and airbench (Keller Jordan, arXiv:2404.00498)** — the two fastest published CIFAR recipes — BOTH use **GELU** activations; our net descends from this lineage but kept DavidNet's ReLU. → idea-02. (knowledge/references/fast-cifar10-recipes.md)
- **NormFace / cosine-softmax (Wang 2017), CosFace (Wang 2018)**: normalized-feature classification regularizes logit geometry (angular separation), distinct from the untuned fixed `SCALE_OUT` logit scale. → idea-03.

## Experimental History Review
- **Current best: 96.38 (EXP-008)**, commit 07c3760. 19 experiments; **14 straight no-improvements (EXP-005, 006, 007, 009–019)** — only EXP-008 (stronger aug) improved since EXP-004.
- **Saturated / closed axes (approach-specific nulls)**: capacity width/depth (005/007/014), optimizer Muon (009/010), input-aug ALL 3 mechanisms occlusion/mixing/transform (008/011/015), reg-scalars wd+LS (012), loss-geometry SAM (013), epoch/throughput via torch.compile (014 — flat), BN/activation-statistic noise (016/017 — closed), downsampling inductive bias BlurPool (018 — closed), and **channel attention SE (019 — closed; +0.28pp same-session did NOT replicate, +0.02pp confirm, never cleared 96.48; adding layer1 SE net-negative)**.
- **Decisive diagnosis (EXP-014, project-insights High; reinforced 016/017/018/019)**: buying +12% epochs AND +capacity gave flat accuracy → the ~96.3–96.5 plateau is a **budget-limited generalization ceiling** for this whitened ResNet-9 at 300s, robust to EVERY within-DavidNet structural lever incl. attention. Mandate: throughput-free (under-anneal is the #1 failure mode behind 5+ nulls) OR a wholesale different backbone.
- **Untried gaps**: (a) **schedule SHAPE** (cosine vs the EXP-001 linear triangular) — EXP-012 EXPLICITLY flagged "schedule-shape (throughput-free)" as an untried lever with ceiling above noise; NOW also has external evidence (MosaicML ~0.5% on CIFAR-10); throughput-free. (b) **activation function** (GELU vs ReLU) — the fastest CIFAR recipes use GELU; never tried here; ~throughput-free. (c) **output-layer geometry** (normalized cosine classifier vs fixed `SCALE_OUT`) — EXP-012 retuned loss SCALARS but never the logit GEOMETRY; throughput-free.
- **Protocol carried in**: same-session control mandatory (~0.1–0.2pp noise floor, stored 96.38 too weak); hard num_epochs≥135 under-anneal gate; mandatory confirmation re-run on any apparent win (low-c0-draw lesson, EXP-016/017/019); anti-gaming integrity (summary==per-epoch max); background nvidia-smi contention sampling; all runs `CUDA_VISIBLE_DEVICES=1`.

## Diagnosis — what limits the objective
The metric is at a **budget-limited generalization ceiling**, not an epoch/throughput/capacity limit (EXP-014 directly disproved the latter by buying epochs+capacity for flat accuracy). 14 straight nulls span every within-architecture axis. Two classes of move remain:
1. **Throughput-free levers that change the LEARNED FUNCTION or its OPTIMIZATION without per-step cost** — these sidestep the under-anneal trap entirely (the failure behind 5+ nulls) and are the rational next probes. The three genuinely-untried ones are the anneal-tail SHAPE (cosine), the ACTIVATION (GELU), and the output GEOMETRY (cosine classifier).
2. **A wholesale different backbone** — flagged as high-EV, BUT the 300s/~150-epoch budget is binding: every standard higher-accuracy CIFAR net (WRN-28-10, DenseNet-BC, Pyramid+ShakeDrop) reaches its accuracy at 200–1800 epochs and would massively UNDER-ANNEAL at ~20–40 epochs in 300s, while DavidNet is already near-optimal for the epoch-efficient regime. So a backbone rewrite is high-risk/poorly-specified THIS loop; the throughput-free levers should be exhausted first (cheap, high information-per-risk).
The schedule-shape lever (idea-01) is the standout: it is the ONE cheap lever with BOTH internal flagging (EXP-012) AND external evidence (MosaicML ~0.5% on CIFAR-10), targets the anneal-tail minimum-selection that EXP-001 showed dominates final accuracy, and cannot under-anneal.

## Collected Ideas
- (Optimization/schedule) **Cosine one-cycle decay** vs the linear triangular shape — throughput-free, MosaicML/EXP-012-evidenced, anneal-tail minimum selection.
- (Activation/representation) **GELU instead of ReLU** — the fastest CIFAR recipes' activation; smoother landscape; ~throughput-free.
- (Output geometry) **Normalized cosine classifier + learnable temperature** vs fixed `SCALE_OUT` linear head — angular class separation, throughput-free.
- (Schedule variant) **Cosine with one warm restart (SGDR)** — snapshot-ensemble flavor; rejected for the lead: a restart risks not fully annealing within 300s and needs tuning (we can eval only 1×/epoch, one model).
- (Capacity) **Second ReZero GatedResidual(256) at the proven 8×8 layer2 stage, compile-funded** — extends EXP-004's one capacity win at the cheap location; rejected as a finalist: EXP-014 found layer2 capacity SATURATED (320 at healthy 143ep still lost), and a 2nd block risks under-anneal even with compile → poor prior.
- (Moonshot/backbone) **Wholesale different backbone** (WRN-lite / ConvMixer / extra full-res stage, compile-funded) — high-EV per the standing diagnosis but high-risk/poorly-specified at the 300s epoch budget (standard high-acc nets under-anneal); deferred until the cheap throughput-free levers are exhausted.
- (Regularization) **Mixup as a REPLACEMENT for occlusion aug** — input-aug lane declared exhausted across 3 mechanisms (008/011/015); low EV.
- (Optimizer) **Per-layer LR / LARS-style scaling** — optimizer axis saturated (009/010); low EV.

## Combinations
- **Cosine schedule + GELU (idea-01 + idea-02)**: both throughput-free, orthogonal (optimization-path × activation); a smoother activation may pair naturally with a smoother (cosine) anneal — reserve as a follow-up rider if either wins alone (avoid confounding two new mechanisms in one verdict cell).
- **Cosine schedule + cosine classifier (idea-01 + idea-03)**: schedule selects the minimum, the normalized head shapes its geometry — composable free riders in a later loop.
- **Winner + a wholesale backbone (any cheap winner → idea moonshot)**: a throughput-free schedule/activation/head win transfers as a free rider onto a future different backbone.

## Candidate Ideas

### 1. Cosine one-cycle decay (vs linear triangular)
**Summary**: Add a `SCHEDULE` env changing ONLY the post-warmup LR decay shape from the current linear (`lr=PEAK·(1−q)`) to cosine (`lr=PEAK·0.5·(1+cos(πq))`), both finishing at exactly 0; warmup/peak/EMA/TTA gates unchanged. Same-session c0(tri)/cA(cos, primary)/cB(cos, shorter-warmup diagnostic). See `proposals/idea-01.md`.
**What it targets**: The budget-limited generalization ceiling at the **anneal tail** — EXP-001 showed most accuracy lands in the low-LR tail, so the decay SHAPE plausibly selects a different (flatter/better) minimum. The single cheap lever EXP-012 explicitly flagged AND that now has external CIFAR-10 evidence.
**Reasoning**: MosaicML benchmark — cyclic/linear schedules underperform cosine decay by up to ~0.5% val accuracy on CIFAR-class CNNs; fastai one-cycle uses cosine, ours uses linear (the less-standard variant); throughput-free → zero under-anneal risk (the #1 failure mode). Strongest cheap-lever evidence on this goal.
**Sources**: `proposals/idea-01.md`; MosaicML LR benchmark (cameronrwolfe substack); EXP-012 04-analysis.md; project-insights Medium (anneal-tail); SGDR arXiv:1608.03983; train.py:286-290.
**Estimated Effort**: low.
**Risk Assessment**: a well-tuned triangular one-cycle is already strong; the ~0.5% MosaicML figure is on ResNet-50/longer schedules and may shrink on this small heavily-augmented net at 150ep — could tie. Throughput-free so num_epochs identical; finishes at 0 so no under-anneal; EMA denoises the steep cosine tail.

### 2. GELU activation (replace ReLU)
**Summary**: Replace `nn.ReLU` in `conv_bn` with `nn.GELU` behind an `ACT` env (`relu`/`gelu`) — the activation used by the fastest published CIFAR recipes (hlb, airbench). Same-session c0(relu)/cA(gelu, primary). See `proposals/idea-02.md`.
**What it targets**: The ceiling at the **activation/representation** — a smoother function class and non-zero-everywhere gradient, a lever the leading fast-CIFAR nets adopted but this net (inherited DavidNet ReLU) never tried. ~throughput-free.
**Reasoning**: hlb-CIFAR10 and airbench (arXiv:2404.00498) both use GELU; GELU (arXiv:1606.08415) is the modern vision default; smoothness aids optimization/generalization. ~throughput-free (cheap elementwise op) → avoids the under-anneal trap.
**Sources**: `proposals/idea-02.md`; knowledge/references/fast-cifar10-recipes.md; arXiv:2404.00498; arXiv:1606.08415; train.py:101-106.
**Estimated Effort**: low.
**Risk Assessment**: BN after each conv largely normalizes activation-scale differences, so ReLU≈GELU often ties at convergence on small CIFAR nets (the GELU choice in airbench/hlb may be for speed/stability, not accuracy) — modest EV; GELU kernel could under-anneal if slow under our bf16/channels_last (mitigated by a num_epochs probe; airbench runs GELU at full speed); possible mild LR-sensitivity (judged via ep25/anneal).

### 3. Normalized cosine classifier + learnable temperature
**Summary**: Replace the bias-free linear head + fixed `SCALE_OUT=0.125` with a cosine classifier — L2-normalize pooled features and fc-weight rows, scale by a learnable temperature τ (logits=τ·cosθ). `HEAD` env (`linear`/`cosine`), learnable `logit_scale`. Same-session c0(linear)/cA(cosine, primary)/cB(fixed-τ diagnostic). See `proposals/idea-03.md`.
**What it targets**: The ceiling at the **output geometry** — the current head couples class scores to feature MAGNITUDE (the optimizer can cut loss by inflating norm); normalization forces angular separation on a hypersphere and decouples confidence (τ). EXP-012 retuned loss SCALARS but never the logit GEOMETRY. Throughput-free.
**Reasoning**: NormFace/cosine-softmax/CosFace report improved generalization+calibration from normalized-feature classification; `SCALE_OUT` is an untuned DavidNet convention never reformulated as a normalized/learnable temperature; two cheap normalizations → throughput-free.
**Sources**: `proposals/idea-03.md`; EXP-012 04-analysis.md (geometry untouched); NormFace (Wang 2017), CosFace (Wang 2018); train.py:141-178.
**Estimated Effort**: low-medium.
**Risk Assessment**: BN before the head already controls feature scale, so normalization may be largely redundant — likely smallest-upside; τ/LS=0.2 interaction is delicate (mitigated by learnable τ + sane init ~14); bounded logits can slow early convergence (judged via ep25/anneal); cosine heads help most on open-set/imbalanced tasks, less on balanced closed-set CIFAR.

## Review
Cross-model (Codex) review in `01-idea-review.md`. Scored: **idea-01 cosine 8/7** > idea-02 GELU 5/4 > idea-03 cosine-classifier 4.5/4. Pick: **idea-01** — "strongest combination of internal history, external schedule evidence, low implementation risk, and direct relevance to the diagnosed limiter."

Top concerns + resolutions:
1. **Don't oversell the MosaicML ~0.5%** (it's ResNet-50/longer-schedule, not this DavidNet/aug/150ep/EMA-TTA setup) → honest-prior framing retained; the verdict requires a same-session >0.1pp margin, not the external number.
2. **Pre-register ONLY `SCHEDULE=cos` with unchanged `PCT_START=0.15` as the verdict cell** → adopted: cA(cos, 0.15 warmup) is the single hypothesis-bearing cell; the shorter-warmup cB is DIAGNOSTIC-ONLY (EXP-021 material if it wins) and cannot trigger an improvement verdict (no schedule-search on the test metric — the EXP-019 cB rule).
3. **`SCHEDULE=tri` must reproduce the baseline LR trace bit-for-bit** → adopted as a regression smoke (sampled `progress, lr_tri, lr_cos` + fraction of steps below LR thresholds).
4. **Correct the "cosine decays faster initially" wording** → cosine HOLDS LR HIGHER early after warmup, crosses linear near mid-decay, then spends MORE time at very low LR (the corrected mechanism; the low-LR-tail emphasis is what plausibly helps).
5. **Same-session control + mandatory confirmation re-run** (cA≥96.48 AND cA−c0>0.1pp, replicated) → adopted (EXP-019 low-c0-draw lesson: +0.28 collapsed to +0.02 on confirmation).

## Idea Evaluation
Adopting the reviewer's pick (idea-01, cosine schedule). It is the strongest remaining cheap throughput-free lever — the ONLY one with both internal flagging (EXP-012) and external CIFAR-10 evidence (MosaicML), targeting the anneal-tail minimum-selection EXP-001 showed dominates final accuracy, with zero under-anneal risk. GELU (idea-02) and the cosine classifier (idea-03) are kept as documented next-loop probes (both throughput-free, lower-EV). No override. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: idea-01 — Cosine one-cycle decay (vs the linear triangular shape).

**Why this idea**:
After 14 straight nulls the net sits at a budget-limited generalization ceiling robust to every within-DavidNet structural lever (capacity/optimizer/regularization/BN-noise/downsampling/attention). The remaining rational moves are throughput-free levers (sidestepping the under-anneal trap behind 5+ nulls) before a costly backbone rewrite. Cosine decay is the standout: it is the one cheap lever with BOTH internal flagging (EXP-012 named schedule-shape "an untried lever with ceiling above noise") AND fresh external evidence (MosaicML: cyclic/linear schedules underperform cosine by up to ~0.5% on CIFAR-class CNNs; fastai's own one-cycle uses cosine, ours uses the less-standard linear). It directly governs the low-LR anneal tail that EXP-001 showed dominates final accuracy, costs zero throughput (num_epochs identical), and finishes at exactly 0 (no under-anneal). The Codex reviewer scored it highest (8/7) as the best cheap path to ≥96.48. Refined per review: verdict keyed on `SCHEDULE=cos` with unchanged `PCT_START=0.15`; `tri` is an exact bit-for-bit regression control; the shorter-warmup variant is diagnostic-only; same-session c0 + mandatory confirmation re-run.

**Hypothesis**:
Changing only the post-warmup LR decay shape from linear to cosine selects a better-generalizing minimum in the low-LR tail and lifts best_test_acc to ≥96.48 over the same-session control by a clear >0.1pp margin at ~150 epochs (throughput-free), replicated on a confirmation re-run. If it ties at full epochs (best≈final), the linear one-cycle is already near-optimal for this net/budget and the anneal-shape lever is exhausted — leaving GELU (idea-02) and the cosine classifier (idea-03) as the remaining cheap probes before the ceiling is declared an architecture+data limit requiring a wholesale different backbone.

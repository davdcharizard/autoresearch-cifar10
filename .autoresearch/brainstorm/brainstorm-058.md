# Brainstorm EXP-058
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **He et al. 2015/2016 (ResNet / pre-act ResNet)** (knowledge: ResNet): depth-vs-width trade on CIFAR — wider-shallower (WRN, Zagoruyko 2016) often matches deeper-thinner at equal params and trains faster (fewer sequential layers). Relevant because our binding constraint is per-step wall-time (launch/bandwidth-bound), not FLOPs.
- **Zagoruyko & Komodakis 2016 (Wide ResNet)**: on CIFAR, a shallow-wide WRN (e.g. WRN-16) reaches near-SOTA with far fewer layers than a deep-thin ResNet, because GPU utilization is higher per layer. Motivates testing FEWER-but-WIDER blocks to reduce sequential-layer launch cost.
- No new external sources required — the project knowledge base + experiment history are the binding evidence here; this loop is a within-known-space pivot, not a literature-driven new technique.

## Experimental History Review
- **Current best 96.45** (EXP-054): `RandomApply([AugMix() w3,d-1], p=0.5)` + GPU Cutout(16) on k=4 ResNet-20. The augmentation-diversity lever (EXP-012/052/054) is the ONLY lever that has ever lifted top-1 (96.00→96.45).
- **The augmentation axis is now FULLY EXHAUSTED** (this loop's key update): magnitude (EXP-053 ✗), width>3 (EXP-055 ✗), coverage<50% (EXP-055 ✗), coverage=100% GPU faithful (EXP-057 ✗, −0.81pp), naive-harsh-GPU (EXP-056 ✗). The winner already uses chain_depth=-1 (random 1–3), so depth is maxed. ~50% coverage is a TRUE interior optimum (bracketed by EXP-055 below / EXP-057 above). Only GPU-AugMix-at-50% remains untried and ≈replicates 96.45 by construction.
- **Capacity axis (claimed closed, EXP-004/009/038/044)**: uniform widening k=5/6 (EXP-004/009 → compute-bound, epoch wall), FLOP-neutral width realloc (EXP-038 → +31% dt, memory-bound, −0.75pp), depth↔width realloc DEEPER-narrower ResNet-32 (EXP-044 → +50% dt, 60 ep, −3.64pp). **CRUCIAL GAP: every capacity experiment INCREASED dt (added FLOPs or sequential layers). The dt-REDUCING quadrant — FEWER blocks (shallower) to LOWER per-step wall-time and reinvest the budget into width-capacity and/or epochs — is the one untested direction.** EXP-044's "don't retry depth variants" was specifically about the unfavorable dt-vs-MORE-depth curve; shallower is its inverse.
- **Throughput→epochs is epoch-saturated at ~91** (EXP-007/045/046): more epochs of the SAME recipe don't help. So a dt reduction only helps if it ALSO changes capacity (more width per block) — which shallower-wider does.
- **Polish/optimizer/schedule/normalization/head/residual/batch axes all CLOSED** (EXP-005/006/010/015/019/020/022/023/025/026/028/029/030/031/032/036/039/040/041/042/043/047/048/050/051). Combining near-misses also failed (EXP-049, cooldown+GC anti-combined).
- **Codebase note**: `train.py` uses `std=(1,1,1)` (mean-centering only, no per-channel std divide) — a non-standard normalization, but BN's scale-invariance (forward standardization + weight-scale-invariant effective LR, Hoffer "Norm matters") makes this a near-certain null. Low-value to test.

## Candidate Ideas

### 1. Shallower-but-wider ResNet-14 (6 blocks, k=5) — the untested dt-REDUCING quadrant of the capacity surface
**Summary**: Reduce `NUM_BLOCKS` 3→2 (ResNet-20 9-block → ResNet-14 6-block; 2 blocks per stage) and widen `WIDTH_MULT` 4→5 (stages {64,128,256}→{80,160,320}). Net effect: ~1/3 fewer sequential conv+BN blocks → fewer kernel launches → LOWER per-step dt (the binding constraint; EXP-044 showed +6 blocks cost +4ms, so −3 blocks should save ~2ms), while the +25% width preserves/raises per-layer capacity (~5M params vs baseline 4.3M). Tests whether a fewer-wider-blocks point on the depth×width×dt surface reaches a better accuracy-per-budget than k=4/9-blocks. Gated hard on dt/epochs: if dt≫8.5ms (epochs <~80) the wide-conv memory wall dominates → abort (W=fewer-blocks-only fallback or k=4 ResNet-14). Single change to NUM_BLOCKS + WIDTH_MULT in train.py; CPU/aug/optimizer/schedule untouched.
**Reasoning**: WRN evidence (Zagoruyko 2016) shows shallow-wide CIFAR nets train faster per-accuracy due to higher GPU utilization per layer — directly relevant to our launch/bandwidth-bound dt floor. Every prior capacity experiment moved dt UP and hit the epoch wall; NONE reduced block count to move dt DOWN. The capacity-closed claim is empirically supported only in the dt-increasing direction. This is the genuinely-untested inverse and the directive-sanctioned "radical architectural change" now that augmentation and all polish axes are exhausted.
**Sources**: WRN (Zagoruyko 2016); ResNet (He 2015); goal-learnings § capacity entries (EXP-004/009/038/044); project-insights § capacity-closed; the dt-floor / launch-bound insight (EXP-040/050).
**Estimated Effort**: low (two-constant change + dt/epoch feasibility gate + idle-GPU launch).
**Risk Assessment**: (a) **wide-conv memory-bandwidth wall** — k=5 convs at 6 blocks may still raise dt past the ~2ms headroom that shallower buys → epoch wall (EXP-004 k=6@9blk=22ms) → mild-to-severe underfit. Gated. (b) **capacity loss from fewer blocks** — 6 blocks give fewer residual-refinement stages; even with more width and more epochs, the net may plateau lower (epoch-saturation means extra epochs on a fully-converged smaller net don't help). (c) Worst case: a regression that confirms k=4/9-blocks is the true compute-optimal frontier (informative — closes the dt-reducing quadrant). Justification for revisiting a High-importance "capacity closed" axis: all prior tests increased dt; this is the untested dt-reducing direction (explicit per-skill justification requirement met).

### 2. GPU faithful AugMix at the proven p=0.5 coverage (W=3) — the one remaining augmentation variant
**Summary**: Reuse the EXP-057 `gpu_augmix` primitives (Dirichlet multi-chain + Beta clean-mix, GPU-side) but apply at the PROVEN-optimal ~50% coverage (per-batch random mask) instead of 100%, W=3. Delivers the EXP-054 winner's coverage/structure via the validated cheap GPU path, with potentially richer continuous-magnitude affine chains than CPU torchvision AugMix.
**Reasoning**: 50% coverage is the proven interior optimum (EXP-054); GPU infra is validated cheap (EXP-056); at 50% coverage the grid_sample cost roughly halves → comfortable epoch budget. The one same-family variant flagged as untried in EXP-057's analysis.
**Sources**: EXP-054 (w3/50% = 96.45), EXP-056 (GPU infra), EXP-057 (clean-mix), goal-learnings § augmentation.
**Estimated Effort**: low-medium (port gpu_augmix + per-batch 50% mask).
**Risk Assessment**: low risk, LOW CEILING — by construction ≈replicates 96.45 (analysis: "≈matches by construction"). Unlikely to clear +0.1 reliably given run-to-run jitter ±0.25pp. Diagnostic value (GPU path matches CPU) > headroom value.

### 3. Per-channel std normalization (std=(1,1,1) → CIFAR-10 std) — clean baseline-gap probe
**Summary**: Change `std` in `Normalize` from (1,1,1) to the standard CIFAR-10 per-channel std (~0.247,0.243,0.261) so inputs have unit variance. Single-line, compute-neutral, cannot epoch-wall.
**Reasoning**: The only clear deviation from standard practice left in the baseline.
**Sources**: codebase (train.py L152-155); Hoffer "Norm matters" (BN scale-invariance).
**Estimated Effort**: trivial.
**Risk Assessment**: NEAR-CERTAIN NULL by theory — BN standardizes conv1 output (forward scale-invariant) and BN-preceded weights have an effective LR governed by weight-norm not input-scale (backward scale-invariant). Low information value; included only for completeness.

## Idea Evaluation
All three respect the hard constraints (train.py only, no new deps, single GPU, eval untouched, ≤1 eval/epoch, no seed hacking). Candidate 3 is near-certain null by well-understood BN scale-invariance theory — minimal information, not worth a loop. Candidate 2 has the strongest evidence base but, by my own EXP-057 analysis, ≈replicates 96.45 by construction (low ceiling, unlikely to clear the +0.1 bar above the ±0.25pp jitter); its value is diagnostic, not headroom.

Candidate 1 (shallower-wider) is the highest-EV remaining option despite real risk. It is the ONLY genuinely-untested quadrant of the capacity surface (every prior capacity test increased dt; none reduced block-count to lower it), it is the directive-sanctioned radical architectural change now that augmentation and all polish/optimizer/schedule axes are exhausted and near-miss-combination already failed, and it has the highest variance → the best chance among available options of actually moving the metric (in either direction), making it the most informative. The epoch-wall risk is real but caught cheaply by an early dt/epoch gate. Choosing it over the "capacity closed" High-importance insight is justified because that insight is empirically grounded only in the dt-INCREASING direction; the dt-reducing inverse is untested.

## Chosen Idea
**Selected**: Candidate 1 — Shallower-but-wider ResNet-14 (6 blocks, k=5).

**Why this idea**: With augmentation fully mapped (this loop) and all polish/optimizer/schedule/normalization/head/batch axes closed, the only principled, directive-sanctioned move left is a radical architectural test of the one untested capacity quadrant: fewer (shallower) blocks to LOWER the binding per-step dt, reinvested into width-capacity. It has the highest variance and information value of the available options, and a strict early dt/epoch gate bounds the downside.

**Hypothesis**: A ResNet-14 (6-block) k=5 net will run at lower block-launch cost than the 9-block k=4 baseline; if the wider convs keep dt within ~2ms of the 8ms floor (epochs ≥ ~80), the preserved/increased capacity at full convergence yields best_test_acc ≥ 96.55 (baseline 96.45 + 0.1). A dt blow-up past ~11ms (epochs <~76) would confirm the wide-conv memory wall dominates even with fewer blocks (epoch-wall regression); a converged result near/below 96.45 at adequate epochs would confirm fewer residual stages cap capacity below k=4/9-blocks — either way closing the dt-reducing capacity quadrant.

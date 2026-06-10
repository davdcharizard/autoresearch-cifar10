# Brainstorm EXP-070
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Global pooling for classification readout** (general CNN practice; CBAM Woo et al. ECCV 2018; image-retrieval GeM Radenović et al. PAMI 2018): the final-stage spatial readout matters. Global Average Pooling (current) captures the *prevalence* of each channel's activation; Global Max Pooling captures its *peak* (the single most discriminative spatial location). For a ReLU network (final features ≥ 0), avg washes out a strong-but-localized response that max preserves. Concatenating avg+max ("dual pooling") gives the classifier both statistics — a strictly richer global descriptor at ~zero FLOP cost (pooling is cheap; fc input doubles 256→512, +2,560 params, negligible). CBAM uses avg+max pooled descriptors precisely because they are complementary.
- **Input normalization (frozen by eval)** — NOTE for the record: train.py L152-154 uses `std=(1,1,1)` (mean-subtraction only, non-standard). I checked `prepare.py` — the **frozen** `Eval` harness ALSO normalizes the test set with `std=(1,1,1)`. So the train normalization is LOCKED to match eval; changing train std to the standard CIFAR per-channel std (≈0.247,0.243,0.262) would create a train/test normalization mismatch → accuracy collapse. **The input-normalization axis is CLOSED by the frozen eval — do NOT touch the Normalize constants.**

## Experimental History Review

Current best = **EXP-054 = 96.45** (`RandomApply([AugMix()], p=0.5)` + GPU Cutout16). Bar = 96.55. **15 consecutive no-improvements since EXP-054** (EXP-055..069).

The augmentation lever — the ONLY one ever to lift top-1 (EXP-012/052/054) — is now FULLY mapped and closed on every sub-axis: chain-COUNT/width (EXP-055), magnitude/severity (EXP-053), coverage (EXP-055/057), GPU delivery (EXP-056/057/059), policy family (EXP-060), label-mixing (EXP-011/018), cooldown (EXP-033/034/035/063), border/occlusion (EXP-037/048), **AND the internal mixing-distribution alpha (EXP-069, this loop's predecessor — −0.20pp)**.

Every non-augmentation axis is also closed: optimizer-family (AdamW EXP-043), grad-dynamics (GC EXP-031, clip EXP-064), objective (SAM EXP-036, PolyLoss EXP-041), weight-averaging (EMA/SWA/Lookahead EXP-006/019/020/068), schedule (peak-LR EXP-016/017, warmup EXP-062, SGDR EXP-029), capacity (k5/k6 EXP-004/009, depth EXP-044, shallow-wide EXP-058, fat-head EXP-038), normalization (GhostBN EXP-047, BN-momentum EXP-067, clean-BN EXP-061), regularizers (WD EXP-005, LS EXP-023/065, dropout EXP-022, LayerScale EXP-051), batch (EXP-050/025), init/micro-arch (pre-act EXP-015, zero-gamma EXP-026, ResNet-D EXP-027, SE EXP-008, deep-sup EXP-042, BlurPool EXP-024).

**Crucial recurring constraint**: EVERY capacity/architecture add that raised dt (k5, k6, depth, fat-head, BlurPool, GhostBN) → fewer epochs → underfit. At this budget the net is epoch-saturated at ~91 (EXP-007/045/046) but ANY dt increase drops below that and underfits. So a new architectural idea MUST be **dt-NEUTRAL** to have a chance.

**Head-change history**: multi-scale head (EXP-032, concat layer2+layer3 → −1.5pp, "disrupts the tuned feature HIERARCHY") and cosine/normalized head (EXP-039, −0.33pp) both failed. KEY DISTINCTION: both fed *different* features (mid-level layer2) or changed the *geometry* (angular). A pooling-statistic change that keeps the SAME final-stage (layer3) features and only adds a complementary global statistic does NOT disrupt the hierarchy — EXP-032's failure mode does not apply.

**Genuinely UNTESTED, dt-neutral gaps**: the global-pooling readout (avg → avg+max or GeM); dilated convs in layer3 (free FLOPs, larger receptive field); SGD momentum coefficient. The pooling readout has never been touched in 70 experiments.

## Candidate Ideas

### 1. Dual (avg + max) global pooling readout
**Summary**: Replace the single `F.adaptive_avg_pool2d(out, 1)` in `ResNet.forward` with a concatenation of average-pooled and max-pooled global descriptors: `a = avgpool(out); m = maxpool(out); out = cat([a, m], 1)`, and widen the classifier `self.fc = nn.Linear(w3, 10)` → `nn.Linear(2*w3, 10)` (512→10). Everything else byte-identical to EXP-054. The model now reads out BOTH the mean and the peak of each channel's 8×8 final feature map.

**Reasoning**: The single biggest unexamined structural choice on this net is the global-pooling readout (untested across 70 exps). For a ReLU net, GAP discards a strong-but-spatially-localized channel response that GMP preserves — these are complementary statistics (the CBAM rationale). Concatenating them gives the linear classifier a strictly richer descriptor at **~zero FLOP cost** (pooling is trivial; fc input 256→512 adds only 2,560 params and a negligible matmul) → **dt stays 8ms → 91 epochs preserved**, avoiding the underfit trap that killed every prior capacity add. Unlike the failed multi-scale (EXP-032) and cosine (EXP-039) heads, this keeps the SAME tuned layer3 features and the SAME logit geometry — it only augments the spatial-aggregation statistic, so EXP-032's "hierarchy disruption" failure mode does not apply.

**Sources**: CBAM (Woo et al. ECCV 2018, avg+max complementary descriptors); project-insights line on head changes; EXP-032/039 reports (head-change failures, distinguished above); goal-learnings epoch-saturation constraint.

**Estimated Effort**: low (3-line forward change + fc width; single-variable; ~590s run).

**Risk Assessment**: Most-likely a within-noise null or small regression — head changes have leaned negative here, and max-pooling can be noisier than avg on small 8×8 maps (a single outlier activation dominates). Worst case ≈ −0.3pp (scalar/head-knob band). dt-safe: adaptive_max_pool2d is a static-shape op (cudagraph-safe under reduce-overhead, no graph break); the 2× fc input is trivial. Params +2,560 (4,302,426 — reported count changes slightly, expected and within the no-fixed-param-count constraint). Clean no-caveat run expected.

### 2. SGD momentum coefficient 0.9 → 0.95
**Summary**: Single hyperparameter change `MOMENTUM = 0.9` → `0.95` (Nesterov SGD). The one untested optimizer scalar.

**Reasoning**: Higher momentum averages more gradient history → smoother trajectory, sometimes a better minimum. Trivial, wall-neutral, single-variable.

**Sources**: train.py L25; goal-learnings optimizer closures (family/grad-dynamics/objective all closed, but the momentum COEFFICIENT itself was never swept).

**Estimated Effort**: low (one constant).

**Risk Assessment**: Near-certain null/slight-regression. m=0.9 is the robust near-universal default; with the budget-matched cosine-to-0 LR (peak 0.2) already tuned, raising momentum effectively changes the step-size/averaging balance the LR schedule was tuned around → likely overshoots the late-anneal. Confounded by the tuned LR (EXP-016/017 showed the schedule is finely balanced). Low expected value but a clean axis-closer.

### 3. Dilated convolutions in layer3 (dilation=2)
**Summary**: Set dilation=2 (padding=2) on the 3×3 convs in layer3's BasicBlocks — a larger effective receptive field over the 8×8 final-stage maps at IDENTICAL FLOPs (dilation does not change kernel size or output size → dt-neutral).

**Reasoning**: Larger receptive field in the final stage → more global context aggregation before pooling, at zero compute cost (91 epochs preserved).

**Sources**: dilated/atrous conv literature (Yu & Koltun ICLR 2016); train.py BasicBlock.

**Estimated Effort**: low-medium (thread a dilation arg through `_make_layer`/`BasicBlock` for layer3 only).

**Risk Assessment**: Weak mechanism — on already-downsampled 8×8 maps a 3-block stack + global pool already aggregates near-globally, so a larger receptive field adds little; dilation can introduce gridding artifacts on small maps. Classification (vs dense prediction) rarely benefits from dilation. Likely null-to-negative. More code surface (risk of a subtle stride/padding bug) than Ideas 1-2.

## Idea Evaluation

All three are dt-neutral (respecting the hard epoch-saturation constraint) and genuinely untested. They differ on mechanism strength and risk:

- **Idea 2 (momentum)** is the safest/cheapest but has the weakest upside — m=0.9 is a tuned-around default and the cosine LR closures (EXP-016/017) make a momentum change likely to just perturb the finely-balanced schedule. Near-certain null. It is a pure axis-closer, not a real bid for +0.1pp.
- **Idea 3 (dilation)** has a weak mechanism on 8×8 maps (classification rarely benefits; gridding risk) and the most code surface / bug risk. Lowest EV.
- **Idea 1 (avg+max pooling)** has the clearest mechanism with literature support (CBAM's complementary-descriptor rationale), targets the ONE genuinely-unexamined structural choice (the readout), is dt-neutral (preserves the saturated 91-epoch budget), and is cleanly distinguished from the failed head changes (EXP-032/039) — it keeps the tuned layer3 features and logit geometry, only enriching the spatial statistic. It is the only candidate that both targets a real representational lever AND avoids every known failure mode (underfit trap, hierarchy disruption, geometry change).

Evidence strength and mechanism clarity favor Idea 1; expected impact is highest there (a richer readout could capture discriminative peak responses the GAP discards); risk profile is comparable across all three (all dt-safe, all single-change). Honest prior for ALL three remains modest given 15 straight misses and the "polish-vs-top1 / 96.45 ceiling" pattern — but Idea 1 is the best-justified real bid on the only structurally-unexamined part of the net.

## Chosen Idea
**Selected**: Idea 1 — Dual (avg + max) global pooling readout

**Why this idea**:
It is the single genuinely-untested *structural* lever that is also dt-neutral (preserving the epoch-saturated 91-epoch budget that every prior capacity add sacrificed → underfit). The global-pooling readout has never been examined in 70 experiments, yet it is where a free change can capture discriminative information the current GAP throws away: for a ReLU net, GAP records only the mean channel activation and discards the peak spatial response, which max-pooling preserves — the two are complementary (the CBAM rationale). Concatenating them enriches the classifier's descriptor at ~zero FLOP cost and only +2,560 params. It is cleanly distinguished from the failed head experiments: EXP-032 disrupted the feature hierarchy by feeding mid-level features, and EXP-039 changed the logit geometry — Idea 1 keeps the tuned final-stage features and standard linear logits, changing ONLY the spatial-aggregation statistic.

**Hypothesis**:
Giving the classifier both the average AND the peak global response of each final-stage channel (avg+max concat pooling, fc 256→512) will let it exploit strong-but-localized discriminative activations that global average pooling currently washes out, raising best_test_acc to ≥ 96.55 at an unchanged ~91-epoch/8ms budget. Stated most-likely alternative: a within-noise null (96.2–96.45), since head changes have leaned null-to-negative on this saturated net and max-pooling can be noisy on 8×8 maps — in which case the pooling-readout axis is closed and the architecture is confirmed exhausted on its last dt-neutral structural lever.

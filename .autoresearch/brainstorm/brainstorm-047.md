# Brainstorm EXP-047
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Ghost BatchNorm** (Hoffer, Hubara & Soudry, "Train longer, generalize better", NeurIPS 2017): normalizing over small "ghost" sub-batches instead of the full batch injects controlled statistical noise into the BN estimates, acting as an implicit regularizer that narrows the large-batch generalization gap. The benefit is strongest at large batch; at small batch the added noise is milder but the mechanism (noisier normalization statistics → flatter, better-generalizing minima) is qualitatively the same. Eval is unchanged — population running stats are used identically to standard BN. Cited as candidate #2 in brainstorm-046 (deferred there only for a torch.compile dt-confound concern, which this experiment specifically engineers around).
- **"How Does BatchNorm Help Optimization?"** (Santurkar et al., NeurIPS 2018) and **Precise BN** (recompute running statistics with frozen weights, used in pycls / "Rethinking the BN" practice): the EMA running stats BN uses at eval can be a biased estimate of the true population statistics; recomputing them over more data with fixed final weights can recover a small top-1 gain. Magnitude shrinks toward zero once the EMA has converged over many updates.
- **project-insights.md / goal-learnings.md** (local): the only intervention that ever broke this plateau was a STRONG, mechanistically-DISTINCT regularizer (TrivialAugment, EXP-012, +0.22pp over the then-best) — weak/redundant regularizers saturate, but a distinct strong one need not. Normalization-induced noise (GhostBN) is mechanistically distinct from the input-space aug (TA/Cutout) and penalty regularizers (WD/dropout) already in the recipe. Separately: the torch.compile reduce-overhead CUDA-graph break (EXP-042) is triggered by DATA-DEPENDENT control flow / variable output structure — NOT by static-shape reshapes; with `BATCH_SIZE=128` + `drop_last=True` every training batch is exactly 128, so a fixed ghost reshape keeps shapes static and the graph intact.

## Experimental History Review

- **Current best / baseline**: 96.22% (EXP-012, 6c417a4), k=4 ResNet-20, ~91 ep @ 8ms. **37 consecutive no-improvements** (EXP-013..046, the last being a no-improvement that closed the throughput→epochs question).
- **ALL conventional accuracy axes are CLOSED**:
  - capacity (all 3 directions — width EXP-004/009, FLOP-realloc EXP-038, depth EXP-044) → epoch wall
  - augmentation — strength (EXP-013/021), policy (EXP-014), mixing (EXP-011/018), cooldown (EXP-033/034/035), border-quality (EXP-037) → saturated
  - LR schedule — peak/floor/shape (EXP-016/017/019/020/029) → closed
  - regularizer-ADDS — dropout (EXP-022) → epoch wall / saturated
  - classifier-head — aggregation (EXP-032), scoring-geometry (EXP-039) → null
  - intermediate-feature-routing (EXP-032/042) → regressive
  - activations (EXP-010/028) → null, doubly-closed
  - weight-averaging (EMA/SWA, EXP-006/019/020) → loss-only polish
  - optimizer family (AdamW EXP-043) + grad-dynamics (GC EXP-030/031) + objective (SAM EXP-036, PolyLoss EXP-041) → null/polish
  - bag-of-tricks (EXP-026) → polish
  - large-batch (EXP-025) → regressive
  - cheap-throughput (cudnn.benchmark EXP-040) → no-op; throughput→buy-epochs (max-autotune+warmup EXP-045, reduce-overhead+warmup EXP-046) → epoch-saturated kernel-independently
- **Two governing walls**: (a) EPOCH WALL — any compute- or sequential-layer-adding change underfits at 300s; (b) POLISH-vs-TOP1 WALL — compute-neutral changes lower loss/calibration but do not raise top-1 (the net is generalization-bound, not boundary/loss-conditioning-bound).
- **The ONE genuinely-untouched accuracy axis**: NORMALIZATION. No experiment has ever altered the BatchNorm computation. GhostBN (candidate #2 in brainstorm-046) and Precise-BN are both untried. This is the only axis with an untested *mechanism* rather than a re-test of a closed one.

## Candidate Ideas

### 1. Ghost BatchNorm — implicit regularization via small-sub-batch statistics, implemented dt-SAFE (static reshape)
**Summary**: Replace every `nn.BatchNorm2d` with a custom `GhostBatchNorm2d` that, in training mode, splits the fixed 128-sample batch into G equal ghost groups (start with G=4 → ghost size 32) via a STATIC reshape `(128,C,H,W)→(G,128//G,C,H,W)`, computes mean/var per (group,channel) over the sub-batch + spatial dims, normalizes each group with its own statistics, reshapes back, and updates the running buffers from the group-averaged statistics. In eval mode it is byte-identical to standard BN (single normalization with the population running stats) — so the frozen `Eval.evaluate()` path is unchanged. The noisier per-ghost statistics inject a regularizing perturbation distinct from input-space aug. All shapes are static (batch fixed at 128, `drop_last=True`), so the reduce-overhead CUDA graph is preserved — the single most important implementation requirement, verified by checking dt stays ~8ms.
**Reasoning**: Normalization is the only untouched axis after 37 no-improvements. The plateau's one breaker was a strong, mechanistically-distinct regularizer (TrivialAugment, EXP-012); GhostBN is exactly that class — a generalization lever orthogonal to the closed input-space-aug and penalty-regularizer families. It is throughput-neutral by construction (one extra cheap reduction per BN, static shapes) so it sidesteps the epoch wall that killed every capacity/compute idea, and it targets *generalization* (the actual bound) rather than loss/calibration (the polish wall). The deferral reason in brainstorm-046 (dt-confound) is removed by the static-reshape implementation + an explicit dt gate.
**Sources**: Hoffer et al. 2017 (GhostBN); goal-learnings EXP-012 (distinct-strong-regularizer lesson); project-insights polish-vs-top1 + epoch-wall entries; EXP-042 CUDA-graph gotcha (the thing to engineer around); brainstorm-046 candidate #2.
**Estimated Effort**: medium — a ~25-line `GhostBatchNorm2d` module + swapping the 4 BN construction sites (stem bn1, BasicBlock bn1/bn2, downsample BN). Recipe/optimizer/schedule/aug/seed all untouched.
**Risk Assessment**: (a) MAIN RISK — at batch 128 GhostBN's benefit is weaker than at the >256 regime it was designed for; ghost-32 might add slightly-too-much noise and land within ±0.25pp (no-improvement) or marginally regress. Mitigated by choosing a mild split (G=4) and being ready to read the result as "normalization-noise is/ isn't a live lever." (b) dt-confound (the EXP-042 failure mode) — mitigated by static shapes + a hard dt-verification gate (must stay ~8ms; if dt rises, the result is confounded and discarded). (c) running-stat update semantics must match BN closely so eval stays clean — use group-averaged stats for the buffer. Worst case: a clean throughput-neutral no-improvement that definitively closes the normalization axis.

### 2. Precise-BN — recompute BN running statistics with frozen final weights before eval
**Summary**: After the timed training loop, freeze the weights, set BN to accumulate, and run a few hundred forward-only passes over training batches to recompute `running_mean`/`running_var` as a precise population estimate (replacing the EMA estimate), then eval. No per-step training-budget cost (it runs after the 300s loop); modest wall-clock (forward-only, well under the 600s limit).
**Reasoning**: A standard, legitimate top-1 lever in modern recipes; the EMA stats can be a biased population estimate. Cheap and untouched. Distinct from weight-averaging (touches only BN buffers, not weights).
**Sources**: Precise-BN (pycls / "Rethinking BN" practice); Santurkar et al. 2018.
**Estimated Effort**: low — a post-training recompute loop in train.py before the eval call.
**Risk Assessment**: With ~91 ep × ~390 steps ≈ 35k BN updates at momentum 0.1, the EMA running stats are already very well converged → the EMA-vs-population gap is tiny → expected gain near-zero (likely within noise). Low-risk, low-reward; a near-certain no-improvement on this long-trained recipe. Better held as a cheap composable add-on than a standalone lead.

### 3. DropBlock — structured spatial dropout in the conv feature maps
**Summary**: Replace the implicit "no structured regularizer" with DropBlock (drop contiguous spatial regions of conv activations, block_size≈3, small drop rate ramped 0→~0.1) in the later stage(s), training-only.
**Reasoning**: Structured dropout is more effective than plain dropout for conv nets (the EXP-022 plain-dropout failure does not directly transfer); it is a distinct regularizer.
**Sources**: DropBlock (Ghiasi et al. 2018); contrast EXP-022 (plain dropout −1.37pp).
**Estimated Effort**: medium.
**Risk Assessment**: HIGH overlap with the CLOSED regularizer-add family — the net is regularization-saturated (Cutout+TA+LS+WD) and EXP-022 showed even mild dropout regressed −1.37pp by under-fitting at the 92-ep budget; DropBlock adds both a regularizer the budget can't absorb AND per-step compute (mask generation) → epoch-wall risk. Lower prior than #1.

## Idea Evaluation

The strategic reality after 37 no-improvements: every conventional axis is closed, and the two walls (epoch wall for compute-adds, polish wall for compute-neutral tweaks) mean the only viable move is a change that is BOTH throughput-neutral AND targets generalization through a mechanism not already saturated.

- **Evidence strength**: #1 sits on the single clearest strategic signal in the project — the only plateau-breaker (EXP-012) was a strong, mechanistically-distinct regularizer, and GhostBN is the one such regularizer on the one untouched axis (normalization). #2 rests on a real but tiny effect (stats already converged). #3 re-enters a closed family (regularizer-adds) against a direct −1.37pp precedent.
- **Mechanism clarity**: #1 is crisp — noisier per-ghost normalization statistics = implicit regularization, throughput-neutral via static reshape, eval unchanged. #2 is crisp but the effect is near-zero here. #3's mechanism is sound but collides with regularization-saturation + epoch wall.
- **Expected impact / risk**: #1 has the best risk-adjusted upside — a genuine (if modest) generalization path with a graceful failure mode (clean no-improvement that closes the normalization axis), and the historical dt-confound blocker is specifically engineered away. #2 is near-certain no-improvement. #3 risks a confounded regression.
- **Feasibility**: all three are implementable in train.py within one loop. #1's only real hazard is the dt-confound, neutralized by static shapes + a dt gate.

#1 wins decisively: it is the only candidate that (a) targets the actual bound (generalization) through an UNTOUCHED mechanism (normalization noise), (b) is throughput-neutral by construction so it dodges the epoch wall, (c) belongs to the exact regularizer class that produced the last real gain, and (d) has its sole historical blocker (CUDA-graph dt-confound) removed by the static-reshape design. #2 (Precise-BN) is noted as a cheap future composable add-on; #3 (DropBlock) is deprioritized as a closed-family re-test.

## Chosen Idea
**Selected**: Ghost BatchNorm — implicit regularization via small-sub-batch statistics, dt-safe static-reshape implementation

**Why this idea**:
Normalization is the only accuracy axis never touched in 47 experiments, and GhostBN is the one regularizer on it that matches the profile of this project's sole plateau-breaker (a strong, mechanistically-distinct regularizer, EXP-012): it is orthogonal to the saturated input-space-aug and penalty-regularizer families, it targets generalization rather than the loss/calibration that the polish wall blocks, and — implemented with a static `(128→G×(128/G))` reshape under `drop_last=True` — it is throughput-neutral and CUDA-graph-safe, sidestepping the epoch wall and the EXP-042 dt-confound that deferred it before. It fails gracefully: a clean throughput-neutral no-improvement would definitively close the last open axis.

**Hypothesis**:
Swapping all BatchNorm2d for a dt-safe GhostBatchNorm2d (G=4, ghost size 32) keeps dt steady at ~8ms and epochs at ~91 (verified), with eval behavior identical to BN. IF normalization-noise regularization is a live lever on this regularization-saturated-but-generalization-bound net, best_test_acc rises ≥0.1pp over 96.22 (≥96.32). The more likely outcome, given the batch is already small (128) so the large-batch gap GhostBN was designed to close is narrow, is a landing within ±0.25pp of 96.22 (no-improvement) — which, at verified throughput-neutrality, would close the normalization axis and confirm the 96.22 k=4/300s ceiling is fully mapped across every accuracy axis.

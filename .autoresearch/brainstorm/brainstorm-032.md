# Brainstorm EXP-032
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **GeM (Generalized Mean) pooling — Radenović et al., "Fine-tuning CNN Image Retrieval with No Human Annotation" (TPAMI 2018)** (web: amaarora.github.io/posts/2020-08-30-gempool.html; researchgate GeM-for-classification notes)
  GeM generalizes avg (p=1) and max (p→∞) pooling: `f = (mean(x^p))^(1/p)`, with p a single learnable scalar. **Key finding from the search: GeM's documented gains are in image RETRIEVAL, not classification — on classification it is reported "comparable to average pooling."** So GeM is a clean compute-neutral probe but with a weak (likely-null) classification prior.
- **Sharpness-Aware Minimization (SAM) — Foret et al., ICLR 2021 (arXiv:2010.01412)** (well-known; not yet in knowledge base)
  SAM seeks FLAT minima by a min-max: perturb weights to the worst-case point in an ρ-ball (ascent on the grad direction), then descend there. ~2× compute (2 forward+backward/step). Documented +0.5–1.5% top-1 on CIFAR ResNets AT FULL epochs. It targets GENERALIZATION (the project's identified binding constraint) directly — but at a fixed compute budget the 2× cost halves epochs.
- **Multi-scale / feature-pyramid classifier heads** (general precedent — FPN, deep-supervision/GoogLeNet aux heads): aggregating features from multiple depths gives the classifier both mid-level and high-level semantics; compute-near-neutral if only an extra global pool + a slightly wider fc.

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). **32 experiments; ~23 axes now closed.** The plateau is exhaustively mapped:
- Scalar knobs bracketed (LR-peak/floor/shape, Cutout, LS, WD, batch).
- Augmentation family closed (TA+Cutout ceiling; policy saturated RA≈TA; label-mixing Mixup/CutMix underfit; occlusion-strength optimal).
- Adding regularizers fails (convergence-bound, not overfit-bound: dropout/Mixup/CutMix/WD↑ all regress).
- Adding compute/capacity hits the epoch wall (k≥5, pre-act, BlurPool, batch-256 — all under-train).
- **Weight-averaging / convergence-POLISH moves LOSS not top-1** (EMA/SWA EXP-006/019/020, Bag-of-Tricks EXP-026, LS-down EXP-023, **and now GC EXP-030/031**).
- Downsampling/anti-aliasing closed both sides (BlurPool + ResNet-D). Activation closed (SiLU ×2). LR-schedule fully closed.
- **Optimizer/gradient-dynamics (entered via GC) behaves polish-like** — GC at a fair throughput-neutral test (EXP-031: 91 ep/8ms) gave 96.14 (top-1 within noise) + loss 0.1894 (better). The EXP-030 "near-miss" was the noise-favorable tail of a top-1 null.

**Crucial synthesis for this loop** (project-insights): the net is **GENERALIZATION-bound at fixed k=4 capacity**, NOT optimization-bound — which is WHY every optimization aid (init, activation, GC, zero-γ) nulls (the net already trains cleanly), and why every compute-adder regresses (epoch wall). Top-1 gains require either more effective capacity (closed) or a genuinely different generalization mechanism. **Flat-minima — the textbook generalization lever — was effectively tested via SWA (EXP-019/020) and gave LOSS-not-top1**, which weakly forecasts the same for SAM (a different route to the same flat-minima goal).

**Genuinely-untested gaps**: (a) the **feature-aggregation / classifier-head** axis — how spatial features are pooled and which depths feed the classifier — never touched (always single-layer global-avg-pool → fc). (b) explicit sharpness-minimization (SAM) — untried, but compute-confounded + flat-minima-forecast-null. (c) inference-side test-time augmentation — high-EV but integrity-gray (see Idea Evaluation).

## Candidate Ideas

### 1. Multi-scale feature-aggregation classifier head
**Summary**: Change `ResNet.forward` so the classifier sees BOTH mid-level (layer2, 128ch @16×16) and high-level (layer3, 256ch @8×8) features: global-avg-pool each, concatenate → 384-vec → `fc(384→10)` (currently `fc(256→10)`). Compute-near-neutral (one extra cheap global pool + a marginally wider fc matmul); params 4,299,866 → ~4,301,146 (+1280 fc weights, +0.03%). The only structural change to a heavily-tuned net is the head's input.

**Reasoning**: The classifier-head / feature-aggregation axis is genuinely UNTOUCHED in 32 experiments — every run used a single layer3 global-avg-pool → fc. On a shallow 9-block net, layer2's 16×16 mid-level features are still discriminative and are currently summarized only after layer3's further downsampling + abstraction. Feeding them directly to the classifier gives a multi-scale inductive bias (mid- + high-level semantics) and adds a direct gradient path to layer2 — a generalization-side change (different feature USE), not an optimization aid (which the net doesn't need) and not a compute-adder (so it sidesteps the epoch wall). It is the cleanest compute-neutral, integrity-clean swing at the one structural axis left open.

**Sources**: train.py L126-133 (forward: single `F.adaptive_avg_pool2d(out,1)` → fc), L107 (`self.fc = nn.Linear(w3, num_classes)`); FPN/deep-supervision multi-scale precedent; project-insights (generalization-bound at fixed capacity; compute-neutral changes only).

**Estimated Effort**: low (edit `__init__` fc in-dim + `forward` to pool layer2/layer3 and concat; one 300s run).

**Risk Assessment**: (a) Directly supervising layer2 via the head could disrupt the tuned feature hierarchy → mild regression (moderate risk). (b) Magnitude likely small on an already-good net → no-improvement within the ±0.2pp noise floor (the dominant outcome across recent experiments). (c) torch.compile recompiles for the new graph (one-time, negligible). (d) Compute-neutral so no epoch-wall confound — verify epochs ~91. Fails gracefully to no-improvement; closes the feature-aggregation axis either way.

### 2. GeM (generalized-mean) pooling with a learnable exponent
**Summary**: Replace `F.adaptive_avg_pool2d(out,1)` with GeM: `out = avg_pool(out.clamp(min=eps).pow(p)).pow(1/p)` where `p` is a single learnable `nn.Parameter` (init p=1 → starts identical to avg pool, recoverable). Compute-neutral, +1 param, no fc change.

**Reasoning**: GeM smoothly interpolates avg↔max pooling, letting the net learn how peaky the spatial aggregation should be — a principled, compute-neutral generalization of the current avg pool with a graceful (p=1) init. Cleanest possible pooling-axis probe.

**Sources**: Radenović et al. TPAMI 2018; web search (amaarora GeM explainer); train.py L131.

**Estimated Effort**: low.

**Risk Assessment**: **Weak prior — the web search reports GeM is "comparable to average pooling" for CLASSIFICATION (its gains are retrieval-specific)** → likely null. Lower ceiling than Idea 1 (single learnable scalar vs a multi-scale inductive-bias change). Safe (p=1 init = baseline) but probably uninformative beyond closing the pooling-exponent sub-axis.

### 3. Periodic Sharpness-Aware Minimization (SAM)
**Summary**: Add SAM's flat-minima min-max to the SGD step, applied every k-th step (e.g., k=8) to bound the compute cost: on a SAM step, forward+backward → g; perturb `w += ρ·g/‖g‖`; forward+backward → g_sam; restore w; step with g_sam. Other steps are plain SGD. ρ≈0.05.

**Reasoning**: SAM directly targets the project's identified binding constraint (generalization at fixed capacity) — the one mechanism class the literature most associates with TOP-1 (not just loss) gains, and it is genuinely untried.

**Sources**: Foret et al. ICLR 2021 (arXiv:2010.01412); project-insights (generalization-bound); goal-learnings (SWA flat-minima = loss-not-top1).

**Estimated Effort**: medium (global grad-norm, perturb/restore around a compiled-model forward, ρ + period to tune).

**Risk Assessment**: TWO strong negatives stacked: (a) **flat-minima was already weakly falsified for top-1 here** — SWA (a different route to flat minima) gave loss-not-top1 (EXP-019/020), forecasting SAM likely does the same; (b) even periodic SAM ADDS compute (~+12–25% → ~73–81 ep) → epoch-wall confound (cf. EXP-024 −0.56pp at 77 ep), so a regression couldn't be cleanly attributed to SAM's merit. Combines two of the project's worst-EV patterns (optimizer-axis polish + compute-adding). Implementation risk with the compiled model (in-place param perturb between graph replays). Deprioritized.

### (Considered and consciously REJECTED) Test-time augmentation (TTA)
Horizontal-flip TTA — wrap the model so eval averages logits over `x` and `flip(x)` (the frozen `evaluate()` does `model.eval(); model(inputs)`, so a wrapper is technically compatible) — is the highest-EV move (reliable +0.1–0.3% top-1, FREE under the budget since eval is untimed). It passes the skill's literal reward-hacking test (the flip-invariance benefit survives any test-set change). **But it is consciously rejected**: it is an INFERENCE-side lever orthogonal to the loop's actual research question (a better model TRAINED in 300s) — it would stack on any model and exploits that inference is unconstrained while training is budgeted. Under this benchmark's strong emphasis on frozen-eval fairness and anti-gaming, running it risks an `invalid`/reward-hacking verdict and does not advance the training question. Documented here so the option (and its rejection rationale) is on record.

## Idea Evaluation

All clean candidates respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking).

- **Evidence strength**: All three have modest priors — the honest state is that ~23 axes are closed and the plateau is likely real. Idea 1 (multi-scale head) targets the one genuinely-UNTOUCHED structural axis (feature aggregation) and is a real inductive-bias change; Idea 2 (GeM) has an explicitly weak classification prior (≈avg pool per the search); Idea 3 (SAM) targets the right mechanism but is doubly-handicapped (flat-minima forecast-null via SWA + compute-confounded).
- **Mechanism clarity**: Idea 1 is crisp and clean — give the classifier multi-scale features, compute-neutral, no epoch wall, no integrity question. Idea 3's mechanism is clear but its forecast (via SWA) is loss-not-top1, and the compute confound muddies attribution. Idea 2's mechanism is clear but evidence says it ≈ no-op for classification.
- **Expected impact**: All low-probability for clearing +0.1pp (plateau is real), but Idea 1 has the best combination of a real-change ceiling AND a CLEAN, attributable test (compute-neutral, integrity-clean). Idea 3's possible signal would be confounded; Idea 2 is likely a flat null.
- **Risk profile**: Idea 1 and 2 fail gracefully (compute-neutral, can't badly regress); Idea 3 risks an ambiguous compute-confounded regression and carries implementation risk.
- **Feasibility**: 1 and 2 trivial; 3 medium.

Idea 1 (multi-scale head) leads: among clean compute-neutral options it targets the one structural axis never tested, gives an unambiguous attributable result, and has a higher real-change ceiling than the pooling-exponent tweak (Idea 2, weak classification prior) without SAM's (Idea 3) compute confound + flat-minima-forecast-null. Even a null cleanly closes the feature-aggregation axis — genuine progress in mapping the plateau.

## Chosen Idea
**Selected**: Multi-scale feature-aggregation classifier head

**Why this idea**:
After 32 experiments closing ~23 axes, the net is generalization-bound at fixed k=4 capacity, so optimization aids null and compute-adders hit the epoch wall. The classifier-head / feature-aggregation axis is the one STRUCTURAL lever never touched — every run pooled only layer3 → fc. Feeding the classifier multi-scale features (global-avg-pooled layer2 ⊕ layer3 → fc) is a compute-neutral, integrity-clean, genuinely-new inductive-bias change: it can move top-1 (different feature USE, the generalization side) without adding compute (no epoch wall) and without the polish-vs-top1 trap (it changes WHAT the classifier sees, not how the optimizer converges). It is preferred over GeM (weak classification prior — ≈avg pool) and SAM (flat-minima already forecast loss-not-top1 via SWA, plus compute-confounded). TTA, though higher-EV, is consciously rejected as an inference-side lever orthogonal to the training research question.

**Hypothesis**:
Concatenating global-avg-pooled layer2 (128ch) + layer3 (256ch) features into `fc(384→10)` lifts `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / dt~8ms / ~4.30M params / <600s, by giving the classifier multi-scale semantics. Falsifiable: if epochs hold ~91 (confirming compute-neutrality) but accuracy lands within ±0.2pp of 96.22, the feature-aggregation axis is also closed and the 96.22 plateau stands — strengthening the conclusion that k=4/300s is the ceiling, and pointing the next loop toward either accepting the plateau or the integrity-deferred inference-side levers.

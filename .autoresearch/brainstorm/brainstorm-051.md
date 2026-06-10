# Brainstorm EXP-051
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Goal, metric, direction, constraints, verification live in the goal file.
     Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.22, EXP-012, 6c417a4). Bar = 96.32. -->

## Web Search & Literature Review

No new external search — the candidate mechanisms are standard and already partly probed in-project:
- **Weight Standardization** (Qiao et al. 2019, "Micro-Batch Training with BN & WS"): standardize each conv filter's weights to zero-mean/unit-variance in the forward pass; smooths the loss landscape and is the forward-weight analog of Gradient Centralization (EXP-031, which acts on grads). Its documented benefit is largest when normalization is WEAK (GroupNorm / micro-batch); with strong BN at batch 128 it is often redundant.
- **LayerScale** (Touvron et al. 2021, CaiT): a learnable per-channel scale on each residual branch (small init) — stabilizes training of VERY DEEP nets. Its benefit is depth-driven.
- **PReLU** (He et al. 2015): learnable per-channel negative slope; a piecewise-linear learnable activation (distinct from the smooth SiLU already tested).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4). Bar = 96.32. **43 consecutive no-improvements (EXP-013..050)**; 6 lifetime improvements (EXP-000..012 era).

**The two governing walls (project-insights, High Importance)**: (1) the **epoch wall** — ANY compute- or sequential-layer-adding change under-trains at the 300s budget (capacity closed all 3 directions: width EXP-004/009, FLOP-neutral realloc EXP-038, depth EXP-044; also BlurPool/SAM/pre-act); (2) the **polish-vs-top1 wall** — compute-neutral changes move loss but not top-1 (the net is generalization-bound at fixed capacity). EXP-050 just closed the **batch-size axis both directions** (128 optimal).

**Direct priors for this loop's candidates**:
- **EXP-026** (zero-init residual γ + no-bias-decay): 96.18, null — "zero-gamma NEEDS DEPTH" on this shallow 9-block net. This directly predicts **LayerScale** (also residual-branch scaling) is likely null here.
- **EXP-028** (SiLU/Swish): 95.98, null AND cost ~1ms/step (8→9ms, 91→88 ep). Predicts **PReLU** is likely null and at risk of a similar non-fusing dt cost.
- **EXP-031** (Gradient Centralization): loss-polish, top-1 null — the grad analog of **Weight Standardization**.

**Honest framing**: after 43 no-improvements with both walls firm and capacity/augmentation/schedule/optimizer/normalization/batch all closed, no remaining lever has meaningful EV for +0.1. The remaining moves are clean throughput-neutral sub-levers that each CLOSE one more cell of the map. Per NEVER STOP, run the cleanest available, document, continue.

## Candidate Ideas

### 1. LayerScale — learnable per-channel residual-branch scaling (small init)
**Summary**: Add a learnable per-channel vector `γ_ls` (init 1e-1) multiplying each BasicBlock's residual-branch output immediately before `+= shortcut(x)`. ~9 vectors of size {64,128,256} → negligible params. Throughput-free (one fused elementwise multiply per block; no graph break, no dt cost).

**Reasoning**: Gives the residual branches a learnable magnitude DOF at small init, which down-weights them early (cleaner identity signal) and lets the net learn per-channel residual contribution — the modern CaiT/ConvNeXt best-practice version of residual scaling. Throughput-free → no epoch-wall confound (the cleanest possible test).

**Sources**: Touvron et al. 2021 (LayerScale); `reports/exp-report-026.md` (zero-init-γ null, "needs depth"); `train.py` BasicBlock L88-92.

**Estimated Effort**: low — one `nn.Parameter` per block + one multiply in `forward`.

**Risk Assessment**: Strong null prior — EXP-026 (the closely-related residual-scaling DOF) was null with the explicit learning "zero-gamma needs DEPTH"; LayerScale's benefit is likewise depth-driven and this is a shallow 9-block net. Most likely a clean within-noise null. No crash/epoch-wall risk (throughput-free).

### 2. PReLU — learnable-slope activation replacing ReLU
**Summary**: Replace `F.relu` at the three sites (block pre/post-residual + stem) with `nn.PReLU` (learnable per-channel negative slope, init 0.25). Distinct from the smooth SiLU already tested — piecewise-linear, learnable.

**Reasoning**: A learnable activation can fit a marginally better nonlinearity; PReLU is piecewise-linear (unlike SiLU), so the smooth-activation null (EXP-028) doesn't fully cover it.

**Sources**: He et al. 2015 (PReLU); `reports/exp-report-028.md`, `-010.md` (SiLU null + ~1ms cost).

**Estimated Effort**: low.

**Risk Assessment**: Activation axis closed (SiLU null ×2); PReLU likely null for the same generalization-bound reason. Per-channel PReLU may not fuse → ~1ms dt cost → mild epoch wall (the EXP-028 failure mode), which would confound a clean test.

### 3. Weight Standardization on conv weights
**Summary**: Wrap the conv weights to standardize each output filter (zero-mean, unit-variance over fan-in) in the forward pass, before `F.conv2d`. The forward-weight analog of Gradient Centralization (EXP-031).

**Reasoning**: Smooths the loss landscape / better-conditions optimization; genuinely untested (GC acted on grads; WS acts on weights in the forward).

**Sources**: Qiao et al. 2019 (WS); `reports/exp-report-031.md` (GC, grad analog).

**Estimated Effort**: medium — custom conv-weight standardization on all conv weights, compiled to limit dt cost; must verify no CUDA-graph break / dt rise.

**Risk Assessment**: WS's benefit is largest when normalization is WEAK; with strong BN at batch 128 it is likely redundant (BN already normalizes activations) → null or mild regression. The per-forward weight reduction risks a dt cost → epoch wall (the EXP-028/036 failure mode). Higher complexity + dt risk than 1/2.

## Idea Evaluation

All three are low-EV by the firm generalization-bound diagnosis; the selection criterion is cleanest-test + most-distinct-from-priors + lowest-risk, optimizing for INFORMATION (closing another map cell) rather than an expected gain.

**Cleanliness / risk**: Candidate 1 (LayerScale) is the ONLY throughput-free option — no dt cost, no epoch-wall confound, so its result is an uncontaminated test of the residual-scaling DOF. Candidates 2 (PReLU) and 3 (WS) both carry a real dt-cost / epoch-wall risk (non-fusing activation; per-forward weight reduction), which — per EXP-028/036 — would confound the result and waste the loop.

**Distinctness from priors**: All three have priors, but candidate 1's prior (EXP-026 zero-init-γ) is itself a slightly different formulation (γ re-init to 0 vs a separate small-init learnable vector), so LayerScale still adds marginal information about the modern formulation. Candidate 3 (WS) is the most mechanistically novel but the highest-risk to execute cleanly.

**Decision**: Lead with **LayerScale** — the cleanest, lowest-risk, throughput-free test; even a near-certain null cleanly extends the residual-scaling closure to the modern formulation without any confound. PReLU and WS are alternates for later loops only if their dt risk can be controlled.

## Chosen Idea
**Selected**: Candidate 1 — LayerScale (learnable per-channel residual-branch scaling, init 1e-1).

**Why this idea**:
It is the cleanest available experiment — throughput-free (no epoch-wall confound, unlike PReLU/WS), a one-line-per-block change, and the modern best-practice formulation of residual scaling. While EXP-026 (zero-init-γ) predicts it is likely null on this shallow net, LayerScale's distinct small-init separate-vector formulation makes it worth one clean confounder-free test, and a null cleanly closes the residual-scaling sub-lever in its strongest form. Lowest execution risk of the remaining candidates.

**Hypothesis**:
Adding a learnable per-channel residual scale (init 0.1) is throughput-free (dt 8ms, ~91 ep, params ≈ unchanged + ~900 scalars). IF a learnable residual-magnitude DOF helps generalization on this net, best_test_acc ≥ 96.32. Falsified (expected, per EXP-026's "needs depth") if within ±0.25pp of baseline — closing residual-scaling in its modern form and pointing the next loop to the remaining alternates (PReLU/WS, dt-risk permitting) or a documentation-of-ceiling confirmation run.

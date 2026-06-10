# Brainstorm EXP-038
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external search. Grounding from project knowledge base + standard ResNet/WRN theory:
- **WRN channel design (Zagoruyko & Komodakis 2016; He 2015 CIFAR ResNet)**: the {16k,32k,64k}
  width schedule with /2 spatial downsampling at each stage transition keeps **per-stage FLOPs
  ~equal** — channel count doubles (4× the conv-weight FLOPs) exactly as spatial area quarters.
  Corollary: a channel in stage3 (8×8, area 64) costs ~16× FEWER FLOPs than a channel in stage1
  (32×32, area 1024). So capacity is far cheaper to add in late stages.
- knowledge/papers: sam.md, gradient-centralization.md, swa.md, smooth-activations.md, blurpool.md,
  bag-of-tricks.md — the optimizer/flat-minima/polish levers, ALL CLOSED here.

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4); bar 96.32 (+0.1). 38 experiments; plateau
extremely well-confirmed (~30 axes closed).

**Two HIGH-importance walls (project-insights § Experimental > High):**
1. **Compute/epoch wall**: ANY non-trivial FLOP add → fewer epochs → under-train → regress
   regardless of merit (EXP-004 k=6 +125% FLOPs→35ep→95.26; EXP-009 k=5 +56%→41ep→94.21;
   EXP-015 preact; EXP-024 BlurPool; EXP-036 SAM). torch.compile cannot lift it.
2. **Polish-vs-top1** (Medium, axis-independent): compute-neutral OPTIMIZATION polish lowers
   loss/flatness but NOT top-1 — EMA/SWA (EXP-006/019/020), LS (EXP-023), bag-of-tricks
   (EXP-026), Gradient Centralization (EXP-030/031, fair throughput-neutral test: loss
   0.1894<0.195 but top-1 96.14 within noise). SAM objective (EXP-036) closed.

**project-insights verbatim implication**: *"Top-1 gains here require CAPACITY or fundamentally
different generalization, not optimization polish."*

**CLOSED**: uniform capacity k>4 (EXP-004/009); ALL augmentation incl. border-quality (EXP-037);
ENTIRE LR schedule (EXP-016/017/019/020/029); all regularizer-adds (dropout EXP-022); architecture
(SiLU/preact/ResNet-D/BlurPool/multi-scale-head, EXP-010/015/024/027/028/032); optimizer
gradient-dynamics (GC EXP-030/031) AND objective (SAM EXP-036); weight-averaging (EXP-006/019/020);
input std-norm INFEASIBLE (frozen eval pins std=1, goal-learnings Protocol Findings).

**The untried gap**: every capacity experiment scaled width **UNIFORMLY** (k applied to all 3
stages), which adds large FLOPs and trips wall #1. **Non-uniform, FLOP-NEUTRAL capacity
reallocation has never been tested** — and it is the ONLY route that targets the actual bound
(capacity, wall #2's stated requirement) WITHOUT tripping wall #1.

## Candidate Ideas

### 1. Compute-neutral "fat-head" width reallocation (narrow stage1, widen stage3 at fixed FLOPs)
**Summary**: Change the per-stage widths from the uniform `{w1,w2,w3} = {64,128,256}` (k=4) to a
**fat-head** schedule that moves capacity from the spatially-expensive early stage to the
spatially-cheap late stage while holding **total FLOPs ≈ constant**. Concretely, edit train.py
L101 `w1,w2,w3 = 16*k,32*k,64*k` to explicit widths, primary config **`{44,128,320}`**. FLOP check
(per-stage ∝ w²·area, area {1024,256,64}): baseline 64²·1024+128²·256+256²·64 = 4.19+4.19+4.19 =
12.58e6 (perfectly balanced); fat-head 44²·1024+128²·256+320²·64 = 1.98+4.19+6.55 = 12.72e6 →
**+1.1% FLOPs** (compute-neutral → expect ~91 ep, no epoch wall). Net channels 448→492 (+44, all in
the discriminative stage3); params increase (allowed — only train.py-edit + budget are constrained).
The `_make_layer`/`BasicBlock` projection shortcuts and `fc = Linear(w3,10)` already adapt to any
widths (L104-107) — minimal edit.
**Reasoning**: This is the one move that threads BOTH High walls. Wall #2 says top-1 needs CAPACITY;
wall #1 says don't add FLOPs. Because a stage3 channel costs ~16× fewer FLOPs than a stage1 channel,
trading a few stage1 channels buys many more stage3 channels at constant compute → a NET capacity
(parameter) increase, placed in the most abstract features that feed the classifier, with NO epoch
penalty. It is categorically distinct from the CLOSED uniform-widening axis (EXP-004/009 added
+56–125% FLOPs → severe under-train; this adds ~+1%). It directly tests project-insights' own
prescription via the only un-walled path.
**Sources**: project-insights High (compute wall) + Medium (polish-vs-top1, "top-1 needs capacity");
WRN equal-per-stage-FLOP design; train.py L100-107, L131-133 (adaptive pool handles any width).
**Estimated Effort**: low — replace one width-assignment line with explicit FLOP-matched widths.
**Risk Assessment**: MEDIUM. Upside: genuine compute-neutral capacity add → the most on-mechanism
attack on the plateau; if capacity is the true bound, this is where it shows. Downside risks: (a)
narrowing stage1 64→44 (−31% early-feature width, though still 2.75× the original ResNet-20's 16ch)
could starve early feature extraction → within-noise null or mild regression; (b) the capacity bound
may be global, not stage3-local, so reallocation just shuffles without net gain. Safe failure mode
(compute-neutral → clean null, no crash/under-train confound). Params change is permitted.

### 2. Additive last-stage widening (spend the ~15% epoch slack: {64,128,320})
**Summary**: Widen ONLY stage3 256→320 additively (no compensating narrowing): `{64,128,320}`. FLOP
add ≈ stage3 +56% = +1.0e6/12.58e6 ≈ **+8% total** (~84-87 ep, within the 77-ep-converges / 91-ep
slack). Pure capacity ADD (nothing removed) at the cheapest stage.
**Reasoning**: Avoids idea #1's stage1-narrowing risk; tests whether the measured epoch slack (77
converge vs 91 available) can absorb a small targeted FLOP add to convert capacity into top-1.
**Sources**: epoch-slack (goal-learnings; EXP-007 77ep converges); project-insights compute wall.
**Estimated Effort**: low.
**Risk Assessment**: MEDIUM-HIGH — directly spends FLOPs, so it leans on wall #1, which is the
dominant failure mode (the High entry says ANY FLOP add regresses, merit masked by under-train).
~84 ep is ≥ the 77-ep convergence point so the masking should be mild, but this is exactly the bet
that has failed repeatedly. Less defensible than #1's compute-neutral framing.

### 3. Stochastic Depth (Huang et al. 2016) — per-block residual drop with linear-decay survival
**Summary**: Randomly zero the residual branch f(x) of each BasicBlock with prob (1−p_l), linear
decay survival 1.0→~0.5 over the 9 blocks; scale by survival at test. Implement vectorized (multiply
f(x) by a per-batch Bernoulli mask) to stay compile-friendly.
**Reasoning**: An implicit ensemble of shallower subnetworks → generalization, the one regularizer
untried here and distinct from feature-dropout (EXP-022).
**Sources**: Deep Networks with Stochastic Depth (Huang 2016); project-insights regularizer-underfit.
**Estimated Effort**: low-medium (per-block mask in forward; depth-indexed survival).
**Risk Assessment**: HIGH-ish — (a) the High regularizer-underfit pattern (EXP-022 dropout −1.37pp)
strongly predicts under-fit at the short budget on the convergence-bound net; (b) the FLOP-saving
benefit of real block-skipping evaporates under torch.compile (data-dependent control flow → graph
breaks), so the masked impl is compute-neutral-at-best, landing it under the polish/regularizer
pattern; (c) on only 9 blocks the depth-ensemble benefit (which scales with depth; proven on
ResNet-110) is weak. Low expected value.

## Idea Evaluation

The plateau is bounded by two HIGH walls; project-insights states top-1 now requires **capacity**, not
polish — so any idea that is pure optimization/regularization polish (#3) or that trips the compute
wall (#2) is fighting the strongest established evidence.

- **Evidence/mechanism**: #1 has the strongest mechanism — it is the unique move that satisfies wall
  #2's requirement (add capacity) without violating wall #1 (constant FLOPs), exploiting the
  quantified ~16× per-stage FLOP asymmetry. #2 shares the capacity mechanism but pays FLOPs → leans
  on the dominant failure mode. #3's mechanism (depth-ensemble regularization) is weak on a 9-block
  net and collapses to the closed polish/regularizer pattern under compile.
- **Expected impact**: #1 targets the real bottleneck (capacity) head-on with no epoch penalty —
  highest ceiling among compute-neutral options. #2 similar ceiling but confounded by under-train.
  #3 most likely within-noise null or regression (regularizer-underfit).
- **Risk**: #1 safest *failure mode* (compute-neutral → clean null, no confound) with a real
  architecture-question downside (stage1 starvation). #2 risks a confounded regression (the exact
  trap of EXP-004/009/024). #3 near-certain null/underfit.
- **Novelty**: #1 is genuinely untried — ALL prior width work was uniform; non-uniform FLOP-neutral
  reallocation is a new axis. #2 is a minor variant of the closed uniform-widening axis. #3 is a new
  regularizer but in a closed class.

**#1 wins**: it is the single most defensible attack on the confirmed capacity bound — it does what
project-insights says is required (add capacity) via the only path that doesn't trip the compute
wall, is a low-effort one-line edit, and fails gracefully (compute-neutral). A clean null teaches
that the capacity bound is global (not reallocatable); a gain breaks a 38-experiment plateau.

## Chosen Idea
**Selected**: Compute-neutral "fat-head" width reallocation — `{64,128,256} → {44,128,320}`
(FLOP-matched ≈ +1%), narrowing the spatially-expensive stage1 to fund a wider spatially-cheap
discriminative stage3.

**Why this idea**:
After ~30 closed axes, project-insights is explicit that top-1 gains now require CAPACITY, not
optimization polish — yet every capacity experiment so far scaled width uniformly and tripped the
compute/epoch wall. Fat-head reallocation is the only untried move that adds net effective capacity
(more total channels/params, concentrated in the discriminative late stage) while holding FLOPs —
and therefore epochs — constant, threading BOTH High walls simultaneously. It is a one-line,
compute-neutral, gracefully-failing probe of the exact bound the whole project has converged on.

**Hypothesis**:
Reallocating capacity from the spatially-expensive stage1 (64→44 ch) to the spatially-cheap
discriminative stage3 (256→320 ch) at constant total FLOPs (≈+1%) will add net effective capacity
(448→492 channels) where it most directly serves classification, lifting best_test_acc above the bar
96.32 WITHOUT the epoch-wall under-training that killed uniform widening — tested at throughput-neutral
~91 ep (the key check: realized epoch count must stay ≈baseline; if it drops materially the FLOP
estimate was wrong). Honest most-likely outcome: within-noise (~96.0–96.3) if the capacity bound is
global rather than stage3-local, or a mild regression if narrowing stage1 starves early features — but
a genuine, on-mechanism shot at the plateau with a clean compute-neutral failure mode.

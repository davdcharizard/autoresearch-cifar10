# Brainstorm EXP-009
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search this loop. Candidate techniques (WideResNet width scaling, network depth, activation-function swaps SiLU/Mish) are textbook and already grounded in the project's own experimental history. Knowledge base (`.autoresearch/knowledge/README.md`) is empty; nothing to re-load. The strongest evidence available is internal (EXP-001 width result, EXP-004 width-wall, EXP-007 compile enabler).

## Experimental History Review

Current best **96.00%** (EXP-003, commit f59de56): k=4 WideResNet {64,128,256} 4.3M + 1×1 projection shortcuts + Cutout(16) GPU-vectorized + bf16/channels_last + time-fraction cosine (peak 0.2, 5% warmup) + Nesterov + label smoothing 0.1, seed 42, ~77 epochs.

What's been tried (10 experiments, 5 improvements):
- **Capacity/width** — EXP-001 k=1→k=4 = **+2.84pp** (THE dominant lever; capacity was the binding ceiling). EXP-004 k=4→k=6 (9.7M) **regressed to 95.26**: eager 22ms/step → only 35 epochs → underfit. The width failure was purely the *epoch wall*, not a capacity problem.
- **Regularization** — Cutout +0.58/+0.52 (EXP-002/003); WD 5e-4 marginal (+0.05, EXP-005). Near-saturated.
- **Weight-averaging** — EMA 95.97 (EXP-006), redundant with cosine-to-0.
- **Training-length/throughput** — `torch.compile(reduce-overhead)` bought 77→89 epochs but acc 95.92 (EXP-007): **epochs saturated past ~77** for the k=4 recipe. CRITICAL by-product: compile is a validated +~30% throughput enabler with null standalone accuracy effect.
- **Channel-attention** — SE r=16, fair 82-epoch run, 95.86 (EXP-008): not channel-gating-limited.

Untried gaps in the approach space:
1. **Capacity between k=4 and k=6** — k=5 was never tested. EXP-004 only jumped to k=6. The compile enabler (EXP-007) now lifts the epoch wall that sank k=6, so an intermediate width may finally pay off.
2. **Capacity via depth** (more blocks/stage, e.g. ResNet-32 n=5) at k=4 — never tested; a different capacity axis than width.
3. **Activation function** (ReLU → SiLU/Mish) — the nonlinearity axis is completely untried.

## Candidate Ideas

### 1. Compiled k=5 WideResNet (capacity, threading the k4–k6 gap)
**Summary**: Raise `WIDTH_MULT` 4→5 (stages {80,160,320}, ~6.7M params, +56% over k=4) and add the validated `torch.compile(mode="reduce-overhead")` enabler on the training forward (eval stays eager), exactly as EXP-007/008. Everything else byte-identical to the EXP-003 recipe. Single conceptual variable vs the compiled-k4 reference: width.

**Reasoning**: Capacity is the project's proven dominant lever (+2.84pp at k=1→k=4). The *only* reason more width failed was the epoch wall: k=6 ran 22ms/step eager → 35 epochs → underfit. Compile (EXP-007) buys ~30% throughput, partially lifting that wall. k=5 sits in the never-tested gap between the k=4 sweet spot and the compute-bound k=6 cliff: ~1.56× the FLOPs of k=4, so eager ~15ms → compiled ~11–12ms → projected ~55–65 epochs — a fair-ish, not-starved test. EXP-007 proved compiled-k4 alone = 95.92 ≈ baseline (null accuracy effect), so any gain over 96.0 is cleanly attributable to the added width (same clean-attribution logic EXP-008 relied on).

**Sources**: TSV EXP-001 (width lever), EXP-004 (width wall), EXP-007 (compile enabler + epoch saturation); goal-learnings § Patterns (widening dominant up to k=4), § Failed Approaches (k=6); project-insights § High (free VRAM), § Medium (reduce-overhead compile).

**Estimated Effort**: low — two-line change (`WIDTH_MULT=5`, add compile) on a validated pattern.

**Risk Assessment**: Main risk is mild under-training (~55–65 epochs) masking k=5's capacity benefit — landing near 95.7–96.0 (no-improvement) rather than a clean win. Worst case is a soft regression like k=6 but less severe (k=5 is far less compute-bound). Graceful failure mode (no crash). If compiled-k5 throughput comes in worse than projected (toward 18ms → ~45 epochs), the result is confounded by epoch starvation — flag num_epochs as the key informational metric (the EXP-008 protocol).

### 2. SiLU activation in place of ReLU
**Summary**: Replace the two `F.relu` calls in `BasicBlock.forward` (and the stem `F.relu` in `ResNet.forward`) with `F.silu` (Swish, x·sigmoid(x)). Keep k=4 and everything else fixed. A pure nonlinearity-axis test — the one architectural lever completely untried.

**Reasoning**: SiLU/Swish is a smooth, non-monotonic activation repeatedly shown to give small but consistent gains over ReLU on image classifiers (it's the default in EfficientNet). Mechanism: nonzero gradient for small negatives + smoothness → better optimization and slightly better generalization. Cheap (elementwise, negligible extra compute, no params), so no epoch hit. Orthogonal to every saturated axis.

**Sources**: Standard activation-function literature (Swish/SiLU, EfficientNet); codebase `train.py:89-92, 127` (the relu sites). No prior experiment touched activations.

**Estimated Effort**: low — 3-line swap, no enabler needed (negligible cost).

**Risk Assessment**: Most likely outcome is a sub-0.2pp delta within the noise band (no-improvement) — activation swaps are typically small. SiLU adds slight per-step cost (sigmoid); on this launch-bound net it may shave a couple epochs, but far less than SE did. Safe failure mode. Low upside but very cheap and de-risked — a reasonable backup if capacity is to be deferred.

### 3. Compiled deeper net (ResNet-32, n=5) at k=4
**Summary**: Raise `NUM_BLOCKS` 3→5 (ResNet-32: 3 stages × 5 blocks), keep width k=4, add the compile enabler. Adds capacity via depth instead of width (~7.4M params).

**Reasoning**: Depth is an alternative capacity route, untried here. Depth grows compute ~linearly with block count (vs width's quadratic), so per-FLOP it can be cheaper — but it adds many sequential small kernel launches, which is *bad* in this launch-bound regime, and lengthens the gradient path (harder to train in few epochs). Compile/CUDA-graphs help amortize the extra launches.

**Sources**: TSV EXP-001/004 (capacity axis); ResNet depth literature (He et al. 2015). project-insights § Medium (launch-bound regime, compile).

**Estimated Effort**: low — one-line `NUM_BLOCKS=5` + compile.

**Risk Assessment**: Deeper-narrower nets are generally weaker than wider-shallower ones at fixed budget on CIFAR (WRN paper's central finding), and the extra launches hurt the launch-bound throughput more than width would — likely fewer epochs *and* a less favorable capacity allocation than idea 1. Higher chance of underfit regression than k=5.

## Idea Evaluation

**Evidence strength**: Idea 1 (k=5) has by far the strongest internal evidence — capacity is the only lever that ever moved the metric by more than a noise margin (+2.84pp), and we have a precise, mechanistic account of *why* more width previously failed (epoch wall) plus a validated tool (compile) that directly attacks that failure cause. Idea 2 (SiLU) rests on general literature (small consistent gains) — credible but low-magnitude. Idea 3 (depth) is the weakest: the WideResNet paper's central result is that width beats depth at fixed budget, and the launch-bound regime penalizes depth's extra kernel launches.

**Mechanism clarity**: Idea 1 — crisp: more representational capacity, with compile restoring enough epochs to use it. Idea 2 — clear but small-magnitude. Idea 3 — clear but the mechanism points *against* it relative to idea 1 in this regime.

**Expected impact**: Idea 1 has the highest ceiling (capacity is the dominant axis; if k=5 trains adequately it can plausibly reach 96.2–96.5). Idea 2 is low-ceiling (≤~0.2pp, likely noise). Idea 3 ≤ idea 1 and more likely to regress.

**Risk profile**: All three fail gracefully (no crash). Idea 2 is the lowest-variance but also lowest-upside. Idea 1's risk (epoch-starvation confound) is *measurable and bounded* — num_epochs tells us immediately whether the test was fair, exactly as EXP-008 handled it.

**Conclusion**: Idea 1 dominates on evidence, mechanism, and ceiling, with a bounded/diagnosable risk. It is the single highest-upside untried lever and directly tests the live strategic hypothesis (compile re-opens capacity). SiLU (idea 2) is the natural cheap fallback for a later loop.

## Chosen Idea
**Selected**: Compiled k=5 WideResNet (Idea 1)

**Why this idea**:
Capacity is the project's only proven high-magnitude lever, and the single concrete reason it stalled (the k=6 epoch wall at 22ms/step) is precisely what the validated `torch.compile(reduce-overhead)` enabler now mitigates (+~30% throughput, EXP-007). k=5 is the untested intermediate width that should be light enough to train adequately under the freed throughput while adding 56% capacity over the k=4 sweet spot. Clean attribution holds: compiled-k4 ≈ baseline (EXP-007), so any gain is from width. It is low-effort, high-ceiling, and its only real risk (under-training) is directly diagnosable via num_epochs.

**Hypothesis**:
A compiled k=5 WideResNet will fit ~55–65 epochs within the 300s budget (vs the feared ~35 of eager k=6) and its added capacity will lift `best_test_acc` above the 96.10 success bar (expected ~96.2–96.5%). If instead it lands near/below 96.0, num_epochs will reveal whether this is genuine capacity saturation at this budget or residual epoch-starvation — either way closing the capacity question that EXP-004 left open.

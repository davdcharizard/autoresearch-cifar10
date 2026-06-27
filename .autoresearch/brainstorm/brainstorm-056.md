# Brainstorm EXP-056
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the in-project knowledge base + model knowledge of two directly-relevant published methods:

- **Identity Mappings in Deep Residual Networks (He et al. 2016, arXiv:1603.05027 — model knowledge, CIFAR-native)**: full pre-activation blocks (BN→ReLU→conv→BN→ReLU→conv added to a CLEAN identity, no post-addition ReLU) improve CIFAR-10 over post-activation, with gains growing with depth (ResNet-110: 6.61→6.37; ResNet-164: 5.93→5.46; ResNet-1001: large). Mechanism: the identity path carries no nonlinearity/normalization, so forward signal and backward gradient propagate unimpeded; the residual branch is properly normalized at its INPUT.
- **Wide Residual Networks (Zagoruyko & Komodakis 2016, arXiv:1605.07146 — knowledge/README.md row, CAVEAT noted)**: WRN — the source of this project's 4× width scaling (EXP-001's +2.07) — uses PRE-ACTIVATION basic blocks as its default B(3,3) block at exactly our depth/width range (WRN-16-4/22-4 ≈ depth 16–22 at 4×). Our net imported WRN's widths onto POST-activation blocks; the block ordering itself was never aligned. Known caveat from the same README row: WRN's projection shortcuts did NOT transfer (EXP-020) — but pre-activation is the opposite kind of edit (removes ops from the identity path rather than adding learnable ones; zero params, zero new kernels).
- **Project laws this must pass**: deferral (no new learnable parts to "turn on" — reordering existing ops; ep1 tripwire still applies); numerics (same kernel set: BN, ReLU, 3×3 conv — only graph ORDER changes; inductor fusion could shift dt slightly → must gate); noise (none); heat (schedule untouched); tail-pressure (nothing stops moving); absorption (the standard screen: He/WRN evidence is crop+flip fixed-epoch — the honest expected effect is ~0; this is explicitly a closure-class candidate).
- **Engineering carry-over (EXP-055)**: the validated ~90s GPU probe (compile + warm + time 40 steps) runs BEFORE the full run to price the reordered graph's dt; mechanism engagement judged by physical signature, never prints.

## Experimental History Review

State after 56 indexed experiments (rows 000–055): baseline 96.71 @ 1990397, bar ≥ 96.81, recipe mean ≈ 96.57 / σ ≈ 0.16 (EXP-027), 49 consecutive non-improvements. Frontier after EXP-055:

- **Every catalogued axis measured-closed**: recipe constants (audit-complete); loss axis both directions; structural CONTENT classes (shortcut topology/anti-aliasing, heads, attention, depth/width/allocation, activations, init both ends); throughput at the 99.3% kernel floor with numerics closed both directions; gradient-noise bracketed; averaging both kinds; augmentation dose-response peaked; data order/coverage; BN/eval constants two-sided; compound-of-frees; allocation-in-time (EXP-055: tail-pressure law now PARAMETER-side too — nothing may stop moving before budget end).
- **The one un-enumerated structural class: block operation ORDER.** Every structural experiment so far changed block CONTENT (what ops exist, what paths exist); none changed the ORDERING of the existing ops. Post-act vs pre-act is the canonical instance, and it is also the literal last unread entry in the "standard modernization" checklist the goal statement asks for (cosine/warmup/LS/TA/RE/nesterov/selective-WD/channels_last/bf16/compile/widening/mixup/EMA/SE/blurpool/zero-γ/SD all measured; pre-act never).
- Relevant priors: EXP-020 (WRN projection shortcuts lose on early-heat + dt — pre-act ADDS nothing learnable and is dt-neutral by op count); EXP-018 (init-time deferral — pre-act trains from scratch in the reference, no turn-on phase); EXP-030/047 (decision-path discontinuities lose — pre-act leaves the head untouched); absorption law (0-for-15 external transfer — priced into the hypothesis as the central risk).
- Protocol carry-overs: composite gates + D0 dt-gate (any fusion regression caught in ~90s), GPU probe before full run (EXP-055), trajectory criterion, replicate escalation for bar-clearing reads (EXP-052), mean-band/sign pre-registration.

## Candidate Ideas

### 1. Full pre-activation block reorder (ResNet v2), WRN-native form: stem conv bare → [BN→ReLU→conv→BN→ReLU→conv + clean identity] ×9 → final BN→ReLU → GAP → fc
**Summary**: Reorder each BasicBlock to full pre-activation (bn1 normalizes the block INPUT: BN(in)→ReLU→conv1→BN(out)→ReLU→conv2; the pad shortcut takes RAW x; no post-addition ReLU), remove the stem's bn1+ReLU (the first block's BN handles it), add a final BN→ReLU before global pooling (He's CIFAR pre-act design). Zero new parameter classes (BN affine count shifts between positions; net params ≈ unchanged — sanity asserts the exact count), identical kernel set and op count, identical schedule/optimizer/data.

**Reasoning**: (a) It is the last un-enumerated structural class (op ORDER) and the last unread standard-modernization entry — whichever way it reads, the audit completes. (b) The evidence is the best-matched of any architecture candidate ever run here: WRN validated this exact block at our depth/width (16–22 layers, 4×) — unlike SE/whitening/projection anchors which were depth- or regime-mismatched. (c) The mechanism is plateau-relevant, not transit-relevant: a clean identity path changes LATE-training gradient quality through the whole net (the converged plateau LEVEL is what the max-statistic harvests), unlike sample-efficiency mechanisms that decay to zero (EXP-028). (d) It is free in every priced currency pending one dt gate: zero params delta (±a few hundred), zero new kernels, zero noise/heat change, nothing learnable added (no deferral turn-on), nothing stops moving (tail-pressure safe). Honest expected effect: ~0 ± σ with absorption as the central scenario (He's shallow-net gains are small and crop+flip-era); the positive tail is that depth-20 post-act has TWO ReLUs sitting on the identity path per block (18 total) whose removal is exactly the kind of optimization-quality lever that EXP-030 showed matters at ±0.5–0.9 scale — in both directions.

**Sources**: model knowledge: He et al. 2016 (arXiv:1603.05027), Zagoruyko & Komodakis 2016 (arXiv:1605.07146); knowledge/README.md WRN row + caveat; goal-learnings EXP-020/018/030 entries; project-insights absorption law.

**Estimated Effort**: medium-low — BasicBlock.forward + __init__ BN-size reorder, ResNet stem/final-BN edit, exact-param sanity, GPU probe, one gated run.

**Risk Assessment**: Branches all terminal: (i) ≥ 96.81 → replicate-pair escalation (MEAN decides); (ii) mean-band [96.41, 96.73] at family signatures → block-order class closed, standard-modernization audit COMPLETE; (iii) < 96.41 → post-act ordering is load-bearing at shallow depth (sign-closure, consistent with He's depth trend); (iv) D0/probe shows a fusion dt regression > ~1.5ms → gate-kill, class closed on cost (the EXP-026 outcome shape); (v) infra → relaunch (max 2).

### 2. Late batch-size increase (512 → 1024 at p ≥ 0.75, LR unchanged) — "anneal noise by batch" (Smith et al. 2018)
**Summary**: Keep batch 512 until p = 0.75, then switch the loader stream to 1024-sample steps (concatenate two loader batches; dual-warmup both graph variants), LR schedule unchanged.

**Reasoning (and why not the lead)**: Published as "Don't decay the learning rate, increase the batch size" (fixed-epoch). Under fixed time: 1024 runs ~41ms (EXP-012/022) ≈ 9% cheaper per image → ~+120 effective steps over the tail (worth ~+0.03, sub-σ), while the REAL variable is a late noise-scale reduction stacked on the cosine's own annealing — and the noise law (EXP-011/022/023/024) measured the recipe AT the noise optimum with constant offsets losing in both directions. A late-only dose is technically un-bracketed (it changes the noise SCHEDULE, not its level), but the closest measured points (EXP-022 constant-1024 at two LR rules; EXP-024 horizon trades) all read negative, and EXP-055 just showed even a fully-delivered tail-conversion cannot pay for a tail-dynamics change. Expected ≤ 0.

**Sources**: model knowledge: Smith et al. 2018 (arXiv:1711.00489); goal-learnings EXP-012/022/023/024; EXP-055 report.

**Estimated Effort**: medium (batch-shape dual warmup, loader stitching).

**Risk Assessment**: graceful (mean-band or sign) but the prior is negative on three adjacent closures; dominated by Idea 1 on evidence match.

### 3. fc-specific LR/WD structure (head LR × 0.5 or WD × 2 on fc only) — documented, not run
**Summary**: Give the single BN-free, scale-sensitive layer (fc) its own LR or WD constant.

**Reasoning (and why not the lead)**: The WD-with-BN equilibrium argument that pre-refuted LARS does NOT cover fc (no BN after it) — so it is the one layer where a per-layer constant is not redundant. But the loss-geometry closure (EXP-050/051: both margin-up and margin-down lose; logit scale is at a measured local optimum) gives any head-scale intervention a direct negative prior — fc-LR↓ is a margin-pressure-down dose in disguise. Expected ≤ 0; kept on the books as the only non-redundant per-layer constant.

**Sources**: goal-learnings EXP-050/051 closure, EXP-015 (WD account); brainstorm-055 Idea 2 (LARS rejection).

**Estimated Effort**: trivial.

**Risk Assessment**: coin-flip against a measured closure; closes only a micro-corner.

## Idea Evaluation

- **Evidence strength**: Idea 1 has the best-matched external anchor of any architecture candidate in 56 experiments — the SAME block, SAME depth range, SAME width multiplier (WRN B(3,3) at 16–22 layers, 4×) — versus Idea 2's three adjacent measured negatives and Idea 3's direct closure conflict. The absorption law discounts all external anchors equally; among discounted anchors, take the matched one.
- **Mechanism clarity**: Idea 1's mechanism is precise and plateau-relevant: remove 18 identity-path ReLUs + properly normalize each residual branch's input; late-training gradient flow through the identity is what a converged plateau's level rides on. Idea 2's mechanism contradicts the noise law's sign; Idea 3's contradicts the loss-geometry closure.
- **Expected impact**: all three are honestly ~0-centered; Idea 1 has the widest positive tail (EXP-030 showed gradient-path quality moves the metric at the ±0.5–0.9 scale in this exact net — in the negative direction there; pre-act is the canonical positive-direction edit of the same currency).
- **Risk profile**: Idea 1 fails graceful and terminal in every branch, with the dt gate + GPU probe catching the only engineering risk (fusion regression) in ~90s for pennies. No stability risk (He trains these from scratch on CIFAR).
- **Feasibility**: contained edit to BasicBlock + ResNet; the EXP-055 probe/gate/composite tooling carries over verbatim.

Idea 1 dominates: it is the only candidate that opens (and closes) a genuinely new class rather than re-dosing a measured one, and it retires the final entry of the modernization audit the goal statement was built around.

## Chosen Idea
**Selected**: Idea 1 — Full pre-activation block reorder (ResNet v2 / WRN-native B(3,3)), stem bare + final BN→ReLU

**Why this idea**:
Block operation ORDER is the last structural class never enumerated, pre-activation is the last standard modernization never measured, and its anchor (WRN) is the only one in project history matched on dataset+depth+width simultaneously — the same paper whose width scaling produced this project's largest single gain (+2.07, EXP-001). Free in params/kernels/noise/heat/tail-pressure pending one ~90s dt gate; every branch terminal.

**Hypothesis**:
Reordering all 9 blocks to full pre-activation (clean identity end-to-end, residual branches input-normalized, stem bare, final BN→ReLU before GAP) leaves every recipe currency unchanged and improves late-training gradient flow through the identity path; if the ResNet-v2 mechanism survives heavy-aug absorption at depth 20, best_test_acc reads above the recipe mean (≥ 96.81 if the true effect is ≥ +0.3, one-draw detectable). Pre-registered branches: (i) read ≥ 96.81 → byte-identical replicate, improvement iff MEAN of the pair ≥ 96.81 (EXP-052 protocol; reads in (96.73, 96.81) are no-improvement, never single-draw promoted); (ii) read ∈ [96.41, 96.73] at family signatures → absorption-null; block-order class closed and the standard-modernization audit COMPLETE; (iii) read < 96.41 → post-activation ordering is load-bearing at shallow depth (sign-closure consistent with He's gains-grow-with-depth trend); (iv) GPU probe or D0 gate shows the reordered graph pays > ~1.5ms/step (inductor fusion regression) → gate-kill, class closed on cost without a full run; (v) gate/contention/startup kills → infra relaunch (max 2). Expected signatures: dt ≈ 22.3–23.0ms (same kernel set; probe-measured before launch), epochs 136–141, params ≈ 4,286,026 ± 1k with the EXACT count asserted in CPU sanity and pinned as the integrity value, ep1 ≥ 30 (no-deferral tripwire), family trajectory/plateau shape, evals ≤ epochs.

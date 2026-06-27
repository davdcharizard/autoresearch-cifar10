# Brainstorm EXP-044
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches this loop — the knowledge base already covers the relevant family; re-read in place:

- **Wide Residual Networks** (knowledge/README.md § References → arXiv 1605.07146)
  Width beats depth on CIFAR-10 with monotone gains at 1–12× for shallow nets **at fixed epochs**. In-project, the 1×→4× step delivered +2.07 (EXP-001) — the single largest gain in the program's history. WRN widens **uniformly**; per-stage asymmetric widening is the unexplored corner of its design space here.
- **RegNet: Designing Network Design Spaces** (knowledge/papers/regnet-design-spaces.md, arXiv 2003.13678)
  Optimized populations have widths strictly increasing across stages (w1 < w2 < w3) with the largest fraction of capacity in later (cheaper-per-FLOP) stages. The DEPTH version of this allocation ([2,3,4]) lost here (EXP-017), but EXP-017's own report flagged **width asymmetry with all stages intact** as the surviving untried capacity-where-cheap move. 64/160/256 keeps w1<w2<w3 and the depth structure untouched.
- **Identity Mappings in Deep Residual Networks** (He et al. 2016, arXiv 1603.05027 — not in knowledge base; considered from model knowledge for Idea 3)
  Pre-activation reordering's CIFAR gains concentrate at depth ≥110 (large at 164/1001 layers, marginal at 110); no published evidence of gains at depth 20. Cited only to document why Idea 3 is screened out.

## Experimental History Review

State after 44 indexed experiments (TSV: experiment-indices/maximize-cifar10-test-accuracy.tsv): baseline 96.71 @ 1990397 (EXP-006 recipe), bar ≥ 96.81; run-level σ ≈ 0.16 with baseline mean ≈ 96.57 (EXP-027) — TRUE effects must be ≥ +0.3; EXP-043 was the 37th consecutive non-improvement at a metric value.

- **Every identified mechanism class now has at least one measurement.** Closed classes (do not re-derive): recipe constants (bracketed both directions, audit-complete EXP-036), schedule shapes, optimizer geometry, batch, gradient-noise scale, init, activations, shortcuts, head pooling, attention/SE, SAM, regularizer doses, data order/coverage, tail lightening, resolution, BN constants, weight averaging, and — as of EXP-042/043 — the entire ensemble/multiplicity axis (function-space gain real at +0.3–0.5 but starvation costs −0.9; in-one-kernel variants 2.8× hardware-closed).
- **Capacity is closed in every measured currency EXCEPT one untested interior.** The width ladder failures were all starvation- or cliff-confounded: 8× → 40 epochs (−0.82), 5× unaligned/cliff → 52 (−1.11), 6× → 55 (−0.71), 4.5×/5× → GATE_KILL at the 54ms >256-channel cliff (EXP-040). No experiment has ever measured a capacity INCREASE at ≥100 epochs. The only way to add width without crossing the 256-channel cliff is asymmetric widening of stages 1–2.
- **Hardware pricing laws available for pre-screening**: dense ≤256-channel convs price at ∂dt/∂FLOPs ≈ 13.3 ms/unit with width-independent ~2.5ms/block (EXP-034); >256 channels = 2.4× cliff (EXP-040); grouped kernels 2.5–3× dense (EXP-042); kernel-FAMILY changes must be dt-gated, and ANY architecture change is dt-gated as standard practice (goal-learnings § Medium).
- **Laws every candidate must screen against**: deferral (+1ms ≈ −6 ep ≈ −0.08), numerics equivalence, max-statistic (only converged plateau LEVEL pays), gradient-noise optimum, absorption (heavy-aug kills light-aug-calibrated techniques — in-regime evidence required), epoch-boundary law, fixed-budget multiplicity law, EXP-006 conversion arithmetic (~0.014/epoch).
- **Flagged untried gap** (exp-report-017 § Insight, reaffirmed exp-report-043 § Next Steps): within-cliff width asymmetry — widen interior stages while keeping max width at 256. Low prior (capacity-failure-adjacent) but the only single-model configuration class never measured.

## Candidate Ideas

### 1. Within-cliff asymmetric widening: stage widths 64/160/256 (dt-gated)
**Summary**: Change one constant-pair in `ResNet.__init__`: stage widths (64, 128, 256) → (64, 160, 256) — widen stage 2 by 25% while leaving stage 1, stage 3, depth, and every training constant byte-identical. 160 is 32-aligned and every layer stays ≤ 256 channels (no cliff). All 9 blocks and the pad shortcut logic work unchanged (the 160→256 transition pads 96 channels instead of 128). Run behind the standard early dt gate (D0-median over first 200-step windows, kill threshold 28ms — off the 6ms print rungs).

**Reasoning**: This is the first capacity increase the program can run in the **non-starved regime**. Pricing by the dense law: ΔFLOPs ≈ +18% (5 stage-2 convs ×1.5625, two transition convs ×1.25, over ~18 equal-cost convs) → projected dt ≈ 22.4 + 0.18×13.3 ≈ **24.8ms** → ~125 epochs — far above the starvation ladder (EXP-002/005/007 lived at 40–55 epochs) and close enough to 139 that the plateau fully converges. Params +~0.51M (≈4.79M, +12%). Deficit by conversion law ≈ 14 ep × 0.014 ≈ −0.20, so the capacity must buy ≥ +0.45 true level to clear the bar from the mean — demanding, but width is the only lever that has EVER bought level at this scale (+2.07 at the 4× step), WRN's fixed-epoch curve is monotone to 8–12×, and 4×→6× was never a fair level test (55 epochs). Asymmetric placement follows RegNet's w1<w2<w3 finding and EXP-017's surviving flag; stage-2 widening is the cheapest non-cliff placement per added parameter (stage-1 widening costs 2.25× stage-1 FLOPs at 32×32 resolution). Secondary value: either outcome closes the last flagged single-model gap, and a GATE_KILL would discover whether 32-aligned-but-not-64-aligned widths misprice — new hardware law either way.

**Sources**: goal-learnings § Failed Approaches (capacity entry + EXP-017 insight); project-insights § High (dense pricing law) and § Medium (256-cliff entry); knowledge/README.md WRN + RegNet rows; exp-report-040.md (gate design), exp-report-043.md § Next Steps.

**Estimated Effort**: low (a 3-integer change in `ResNet.__init__` + standard composite gated launcher; CPU sanity for params/shortcut padding).

**Risk Assessment**: Main risk is the known one — capacity adds level only when the regularization/epoch regime can use it, and EXP-009 suggested LS+TA+RE saturates 4.29M params. If so, expect plateau ≈ mean − deficit (~96.3–96.5) with family test_loss: a clean closure of the asymmetry gap. Kernel risk: 160-wide convs are 32-aligned but not 64-aligned; if the inductor picks bad kernels the dt gate kills in ~90s (~2 GPU-minutes wasted, new pricing datum gained). No deferral risk (no new module classes, Kaiming init unchanged, no early-heat learning burden beyond normal width). Worst case: −0.4, graceful no-improvement.

### 2. Double-asymmetric widening: stage widths 96/160/256
**Summary**: Same intervention class at a larger dose — widen stage 1 to 96 and stage 2 to 160 (96/160/256, all ≤256, 32-aligned), keeping w1<w2<w3. One launch behind the same dt gate.

**Reasoning**: If capacity-at-converged-epochs pays at all, a larger dose pays more — this probes the same mechanism further up the dose curve. But the pricing is much worse: stage-1 convs run at 32×32, so 64→96 costs ×2.25 on 6 convs ≈ +42% FLOPs alone; combined total ≈ +60% FLOPs → projected dt ≈ 30.8ms → ~101 epochs → deficit ≈ −0.53 by the conversion law. It needs ≥ +0.8 true level — twice what Idea 1 needs — and EXP-017 showed stage-1 representation is already adequate at depth 3 (its deficit isolated "less stage-1" specifically, not "more stage-1 helps").

**Sources**: same as Idea 1; exp-report-017.md § Results (stage-1 evidence); exp-report-034.md (dt pricing).

**Estimated Effort**: low (identical mechanics to Idea 1).

**Risk Assessment**: Strictly worse deficit arithmetic than Idea 1 with no independent mechanism; ~101 epochs also re-approaches the regime where transit effects confound the level read. Dominated as a FIRST probe; only sensible as a follow-up if Idea 1 shows a positive slope.

### 3. Pre-activation block ordering (ResNet v2: BN→ReLU→conv)
**Summary**: Reorder each block to the He et al. 2016 pre-activation form with a clean identity path, plus the required first/last special-casing. dt-free in principle (same ops, reordered) and parameter-neutral.

**Reasoning (and why it fails the screens)**: The one dt-free structural change never tried. But it fails three binding screens at once: (a) published gains concentrate at depth ≥110 and are marginal-to-zero at shallow depths — no in-regime evidence at depth 20, and external fixed-epoch architecture evidence is 0-for-12 transferring here; (b) reordering changes early-optimization dynamics exactly like the structural-arc losers (EXP-018/020/030 — deferral/quality class); (c) it changes the numerics of every block (different BN placement = different normalization statistics path) without any identified mechanism for raising the converged plateau LEVEL, which is the only thing the max-statistic pays for.

**Sources**: arXiv 1603.05027 (model knowledge); project-insights § High (deferral law, 9 confirmations); goal-learnings § Failed Approaches (structural arc).

**Estimated Effort**: medium (touches every block's forward + stem/head special cases + sanity).

**Risk Assessment**: Expected outcome is the structural-arc signature: −0.1 to −0.5 with no level gain. Included to document explicit consideration and rejection of the last dt-free structural reorder.

## Idea Evaluation

- **Evidence strength**: Idea 1 rests on the strongest in-project precedent there is (width = +2.07, the largest single gain ever measured here) plus the explicit, twice-flagged gap in the capacity record (every prior width increase was starvation- or cliff-confounded — exp-report-017/040/043). Idea 2 shares that evidence but with strictly worse arithmetic. Idea 3 has zero in-regime evidence and three laws against it.
- **Mechanism clarity**: Idea 1 is sharp — more stage-2 channels at ~125 fully-annealed epochs raise the converged plateau LEVEL if and only if 4.29M params was a capacity bound rather than a regularization saturation point; both outcomes are mechanistically interpretable. Idea 3's mechanism (easier signal propagation) targets a problem depth-20 nets don't have.
- **Expected impact**: Idea 1 needs +0.45 true; plausible only if the EXP-009 saturation reading was dose-specific (a 4th regularizer) rather than capacity-specific — uncertain, but no other open candidate has any path to +0.3. Idea 2 needs +0.8 — implausible. Idea 3 expected ≤ 0.
- **Risk profile**: Idea 1 is the safest in the set — one-line change, gate-protected against kernel surprises, graceful no-improvement worst case, and closes a flagged gap with either outcome. Ideas 2/3 add cost or engineering surface without improving the odds.
- **Feasibility**: Idea 1 lowest effort; reuses the exp042-style D0-median composite launcher unchanged except the threshold.

Idea 1 dominates: best evidence, cheapest probe, only candidate whose success arithmetic is even reachable, and guaranteed information value (closes the last flagged single-model gap or discovers a new kernel-pricing law).

## Chosen Idea
**Selected**: Idea 1 — Within-cliff asymmetric widening: stage widths 64/160/256 (dt-gated)

**Why this idea**:
It is the only remaining single-model configuration class with no measurement, explicitly flagged by exp-report-017 and exp-report-043 as the surviving capacity-where-cheap move. Width is the only lever that has ever bought plateau level in this program, and every prior width increase was confounded by starvation (40–55 epochs) or the >256-channel hardware cliff — this design is the first that adds capacity while staying within the cliff AND keeping ~125 epochs, making it a clean level test of whether 4.29M params is a true capacity bound. It is a 3-integer change, fully protected by the validated early dt gate, and informative under every outcome branch.

**Hypothesis**:
Widening stage 2 from 128 to 160 channels (64/160/256, +18% FLOPs, +0.51M params) prices at ≈24.8ms by the dense law (~125 epochs) and raises the converged plateau by more than its ~0.20 epoch-deficit, yielding best_test_acc ≥ 96.81. Pre-registered branches: (i) best ≥ 96.81 → improvement (if the read lands 96.70–96.80, run the pre-registered replicate pair before claiming); (ii) plateau ≈ mean − deficit (96.3–96.5) with family-equal test_loss (~0.185) → capacity is regularization/level-saturated at 4× width, asymmetric-width gap closed, capacity axis closed in ALL currencies; (iii) GATE_KILL > 28ms → 32-aligned-but-not-64-aligned widths misprice on this stack — new kernel-pricing law recorded, verdict invalid, idea closed on hardware grounds.

# Brainstorm EXP-020
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

No new external searches this loop — the knowledge base already holds the relevant evidence; re-read for this ideation:

- **Wide Residual Networks** (knowledge/README.md § References — arXiv 1605.07146 + szagoruyko/wide-residual-networks): the reference family our recipe copies (WRN-style 4x widening at depth ~20 on CIFAR-10) uses **learned 1×1 convolution shortcuts at stage transitions** (type-B), not the 2016 ResNet-20 option-A strided-slice + zero-pad. This is direct reference-implementation evidence at our exact depth/width/dataset regime.
- **ResNet (He et al. 2015)**: ImageNet ablation found option B (projection at transitions only) slightly better than option A; the CIFAR experiments used A only to keep parameter counts identical to plain nets — a fairness choice, not a quality one. Our `train.py` inherited A from that lineage.
- **Bag of Tricks** (knowledge/papers/bag-of-tricks-zero-gamma.md, arXiv 1812.01187): its ResNet-D variant (avgpool + 1×1 shortcut) is a further refinement of projection shortcuts — noted as an unexplored variant, but the WRN-faithful plain 1×1 stride-2 has the closer evidence transfer.
- **RegNet** (knowledge/papers/regnet-design-spaces.md): cautionary — population-level architecture results at other depths did NOT transfer to depth 20 (EXP-017). WRN evidence is exempt from this caution: it was measured at depths 16–40 on CIFAR-10 itself, including 4x width.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe). **Fourteen consecutive misses (EXP-007…019).**
- **Certified local optimum** (goal-learnings § Patterns High): every single-constant probe (heat ±, pressure ±, capacity ±, batch, smoothing) AND every zero-cost structural perturbation (EXP-017 free params, EXP-018 easier optimization, EXP-019 information-at-init) converged below baseline with identical signatures.
- **Master failure mode** (project-insights High): under fixed wall clock, every lever that defers progress fails — 7 confirmations across 4 mechanism classes. Selection criterion sharpened by EXP-017/018/019: only interventions that increase what is **learned per unit of schedule heat, from step 1**, can win.
- **Capacity axis**: closed bidirectionally for uniform changes (8x/5x/6x starved; ResNet-14 and [2,3,4] reallocation converged worse). EXP-017's isolate: the deficit came from *removing early-stage depth* specifically — width asymmetry (keep all blocks, widen stage 3) was left as the one untried capacity-where-cheap move, but it pays an epoch tax.
- **Throughput conversion** (EXP-006): extra epochs at unchanged hyperparameters DO convert (+25 epochs → +0.48pp) — the only validated +pp mechanism since EXP-006. Conversely EXP-012: throughput that forces hyperparameter changes is metric-neutral.
- **Untried gaps**: (a) the transition-shortcut type — `train.py` still uses option-A strided slice + zero-pad, the last structural 2016-era component, never probed; (b) stage-3-only width asymmetry; (c) heat-constant momentum+peak trade; (d) GPU-side augmentation to recover ~50s of loader stalls (exp-report-013 measured baseline stalls).

**Synthesis check (required by exp-report-019 § Next Steps #3)**: with every cheap axis closed, a new probe is only justified if it is *qualitatively outside* the closed classes AND carries evidence from a closely-transferable regime. Of the four gaps above: (a) is the only one that is simultaneously not-a-constant, not-zero-cost-free-lunch (it pays a small, real capacity price through a topology change active from step 1), not deferral, and backed by a reference implementation at our exact depth/width/dataset. (b) pays ~20 epochs against a small marginal-width return — the exact trade EXP-012 measured at 1:1. (c) has no comparable-regime evidence. (d) has the strongest validated mechanism (throughput conversion) but requires reimplementing per-image TrivialAugmentWide on GPU — high risk of augmentation-semantics drift on a regularization axis that is peaked and punishes ±0.14–0.46pp for distribution changes; deferred, not discarded. Expected value ranking: (a) > (d) > (b) > (c). The campaign continues with (a); if it misses, (d) is the next-highest-EV direction.

## Candidate Ideas

### 1. Projection shortcuts at stage transitions (ResNet option B, WRN-faithful)
**Summary**: Replace the option-A transition shortcut in `BasicBlock` (strided slice `x[:, :, ::stride, ::stride]` + zero channel padding) with a learned projection: `nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)` + `nn.BatchNorm2d(out_ch)`, applied only in the two transition blocks (64→128 and 128→256). Identity shortcuts everywhere else stay untouched. +41,728 params (+0.97%, → 4,327,754), FLOPs negligible (two 1×1 convs at 16×16/8×8), dt unchanged within noise, ~139 epochs preserved.

**Reasoning**: The current shortcut throws away 75% of spatial positions (strided slicing) and gives HALF the output channels of each transition block a zero shortcut — those 64+128 channels get no identity signal and must be synthesized from scratch by the residual branch. A learned 1×1 projection gives every channel a trained shortcut that mixes all input channels. This is active from step 1 (Kaiming-initialized, trains during warmup) — it passes the learned-per-unit-of-heat criterion, unlike EXP-018's deferral. It is the exact component where our net still diverges from the WRN reference at our width regime, and the task is explicitly "modernize the 2016-era recipe". Mechanism is information-path topology — a class no prior experiment has touched.

**Sources**: knowledge/README.md WRN row (arXiv 1605.07146); He et al. 2015 option A/B ablation; goal-learnings § Patterns (EXP-017/018/019 selection criterion); exp-report-019.md § Next Steps.

**Estimated Effort**: low — ~10 lines in `BasicBlock.__init__`/`forward`; all training constants byte-identical.

**Risk Assessment**: Most likely failure is a graceful wash-out (the zero-padded channels are eventually compensated by 139 epochs, same way whitening's basis was learned — converged no-improvement within ~0.2pp). Historical B-over-A gains are ~0.1–0.3pp, straddling the +0.1 bar. Small dynamics risk: projection+BN at transitions slightly changes early-epoch behavior (2 of 9 blocks); WRN evidence says net positive. Worst case: −0.3pp converged, signatures unchanged, clean attribution.

### 2. Stage-3 width asymmetry: widths 64/128/320, depth [3,3,3] unchanged
**Summary**: Widen only stage 3 (8×8 resolution) from 256 to 320 channels (320 = 64×5, alignment-safe). +1.73M params (+40%) for only +17% FLOPs since stage 3 is spatially cheap. Projected dt ~26–27ms → ~115–120 epochs (above the 70 floor). All training constants unchanged.

**Reasoning**: The one untried "capacity where it is cheap" move; preserves early-stage depth (EXP-017's failure isolate) and channel alignment (EXP-005's failure isolate). WRN showed width gains up to 12x on CIFAR — but uniformly, at fixed epochs.

**Sources**: exp-report-017.md § Next Steps; goal-learnings § Failed Approaches (EXP-002/005/007 starvation; EXP-012 1:1 exchange); project-insights (H20 alignment).

**Estimated Effort**: low-medium — change `ResNet.__init__` width tuple; requires the measured-dt gate (kill by step ~150 if projected epochs < ~100).

**Risk Assessment**: Pays ~20 epochs. EXP-006 conversion arithmetic prices ~20 epochs at roughly −0.4pp, which the marginal width must overcome — and the marginal return of widening one stage from 4x to 5x is far smaller than the 1x→4x step that earned +2.07pp. EXP-012 measured exactly this trade at 1:1. Compiled scaling may be worse than the FLOPs ratio (EXP-007). Failure mode is graceful but the prior is low.

### 3. Heat-constant momentum trade: MOMENTUM 0.9→0.95 with PEAK_LR 0.4→0.2
**Summary**: Raise Nesterov momentum to 0.95 while halving peak LR so the effective per-step size lr/(1−β) is held at the certified value (0.4/0.1 = 0.2/0.05 = 4). Tests whether longer gradient averaging (smoother directions, same integrated heat) buys anything at fixed wall clock.

**Reasoning**: Momentum is the only never-touched constant in the recipe; goal-learnings prices single-knob momentum moves via the closed heat axis, so only the compensated trade is admissible. Smoother updates could lengthen/raise the converged plateau the max-statistic harvests.

**Sources**: goal-learnings § Failed Approaches Medium (heat axis closed twice); exp-report-019.md § Next Steps #2.

**Estimated Effort**: low — two constants.

**Risk Assessment**: No comparable-regime evidence that 0.95 beats 0.9 on CIFAR-scale CNNs; the lr/(1−β) equivalence is first-order only (curvature/noise interactions differ), so the probe may just re-measure the heat optimum with extra variance. Failure graceful. Prior: lowest of the three.

## Idea Evaluation

**Evidence strength**: Idea 1 wins decisively — the WRN reference implementation, at our exact dataset/depth/width regime, uses projection shortcuts; this is the closest evidence transfer available anywhere in the remaining candidate space (contrast EXP-017, where the RegNet evidence lived at alien depths and failed to transfer; WRN was measured at depths 16–40 on CIFAR-10 itself). Idea 2's evidence (WRN width gains) is fixed-epoch evidence that the fixed-clock budget has repeatedly inverted (EXP-002/005/007). Idea 3 has no direct evidence.

**Mechanism clarity**: Idea 1: half the transition channels currently receive zero shortcut and the strided slice discards 75% of activations — a learned projection restores a full-rank, all-positions information path, active from the first step (no deferral). Clear and specific. Idea 2: more params where FLOPs are cheap — but the mechanism must outrun a quantified ~−0.4pp epoch tax. Idea 3: mechanism is speculative ("smoother might help").

**Expected impact**: Idea 1's literature delta (~0.1–0.3pp) straddles the +0.1 bar — modest but real, at zero schedule cost. Idea 2's net expectation is negative after the epoch tax. Idea 3 is most likely within noise.

**Risk profile**: Idea 1 keeps every signature (params +0.97%, dt unchanged, ~139 epochs) → perfect attribution and graceful failure. Idea 2 changes throughput, epochs, VRAM simultaneously. Idea 3 is graceful but uninformative if it washes.

**Feasibility**: All three are small diffs; Idea 1 is the smallest behavior-relevant change.

The deferred direction (d) from the synthesis check — GPU-side augmentation to reclaim ~50s of loader stalls — has the strongest validated mechanism (EXP-006 throughput conversion) but its implementation risk (faithful per-image TrivialAugmentWide on GPU; any semantic drift confounds on the peaked regularization axis) makes it a poor next probe while a cheaper, cleaner candidate exists. It is the designated follow-up if EXP-020 misses.

## Chosen Idea
**Selected**: Projection shortcuts at stage transitions (ResNet option B, WRN-faithful)

**Why this idea**:
It is the last structural 2016-era component in `train.py` (the task is explicitly modernization), the only remaining candidate backed by a reference implementation at our exact regime, and it satisfies the sharpened selection criterion from EXP-017/018/019: it adds trained information-path capacity that operates from step 1 — not a constant tweak, not deferral, not init-time information the schedule can re-learn. It pays a real but tiny capacity price (+0.97% params) while preserving every throughput signature, so attribution will be exact and failure graceful.

**Hypothesis**:
Giving all transition-block output channels a learned full-rank shortcut (instead of zero-padding half of them and discarding 75% of spatial positions) raises what the net learns per unit of schedule heat throughout training. Prediction: early trail at-or-above the baseline family (ep1 ≈ 38–39, ep10 ≥ 75), signatures preserved (params 4,327,754, dt ≈ 22.4ms, ~139 epochs, VRAM ≈ 1613MB), and a converged plateau with **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1). A converged miss within ±0.2pp would close the shortcut-topology axis and route the campaign to the GPU-augmentation throughput direction.

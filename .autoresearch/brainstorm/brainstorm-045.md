# Brainstorm EXP-045
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches — the relevant references are already in the knowledge base and were re-validated last loop:

- **Wide Residual Networks** (knowledge/README.md § References, arXiv 1605.07146): width gains monotone at fixed epochs on shallow CIFAR nets; in-project the 1×→4× step bought +2.07 (EXP-001) — the only lever that ever moved plateau LEVEL.
- **RegNet design spaces** (knowledge/papers/regnet-design-spaces.md): good designs have w1<w2<w3 with capacity weighted toward later stages — 64/192/256 preserves this ordering at the maximal within-lattice stage-2 dose.
- **EXP-044's hardware result** (reports/exp-report-044.md — internal, but it IS the literature for this loop): dense pricing laws hold only on the kernel lattice (widths ≡ 0 mod 64, ≤256 channels, dense kernels); 160 cost 4× the law's marginal prediction. 192's status is UNVERIFIED: EXP-034's 48/96/192 fallback ran 27.4ms at depth 26, a mixed signal confounded by its +3 blocks (~+8.4ms of launch cost on its own).

## Experimental History Review

State after 45 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; baseline mean ≈ 96.57, σ ≈ 0.16 (EXP-027) — TRUE effects must be ≥ +0.3. EXP-044 was the 38th consecutive non-improvement/invalid at the front.

- **Last loop (EXP-044)**: 64/160/256 GATE_KILLed at 32.4ms (clean host) — discovered the 64-alignment law. The within-cliff asymmetric-capacity space collapsed to permutations of {64, 128, 192, 256}; exp-report-044 § Next Steps poses the explicit endgame decision: gate-probe 64/192/256 (~90s to resolve 192's lattice status) or close the class by arithmetic.
- **The capacity-at-converged-epochs question is STILL unmeasured.** Every capacity increase in program history was confounded: 8×/5×/6× by starvation (40–55 epochs), 4.5×/5× by the >256 cliff, 64/160/256 by the alignment law. No experiment has measured what added capacity does at ≥100 fully-annealed epochs. Width remains the only lever that ever bought level (+2.07).
- **Exhaustive-closure context**: every other mechanism class is measured-closed (recipe constants audit-complete; schedule, optimizer, batch, noise, init, activations, shortcuts, head, attention, SAM, regularizer doses, data order, tail, resolution, BN constants, weight averaging, ensembles/multiplicity). The arithmetic-only candidates left in those classes (H2D prefetch ≈ +0.04; label-error removal ≲ +0.1; cold-side LR closed by bracketing inference) all fail the ≥ +0.3 effect-size screen.
- **Lattice arithmetic for the remaining permutations**: 64/256/256 → +83% FLOPs ≈ 33.5ms dense-law (starved before gating); 128/128/256 → +100% ≈ 36ms (same); 96/x/x excluded (96 is non-64-aligned — exactly EXP-044's law). **64/192/256 is the unique surviving instrument**: +40% FLOPs (5 inner stage-2 convs ×2.25, two transitions ×1.5, over ~18 equal-cost convs) → dense-law dt ≈ 27.7ms → ~112 epochs (above the starvation ladder; EXP-035 decomposed cleanly at 112).
- Protocol carry-overs: D0-median dt gate (≥200-step windows, off-rung threshold), dual launch gates (GPU apps + host load), replicate pair for mid-band reads, num_epochs cross-check vs projection.

## Candidate Ideas

### 1. 64/192/256 gate-first — the last lattice point (close the capacity class with data)
**Summary**: Same three-line change as EXP-044 with `STAGE_WIDTHS = (64, 192, 256)` (all widths 64-multiples, all ≤256 — fully on the kernel lattice). Same composite launcher; D0-median gate threshold **31ms** (off-rung; dense-law projection 27.7ms passes with margin, the EXP-044-style mispricing population lands ≥35). If the gate passes, the full run is the FIRST clean measurement of capacity at converged epochs: ~112 epochs, deficit ≈ −0.38 by the conversion law, params +~1.7M (exact count computed at plan time), needing ≥ +0.6 true level to clear the bar.

**Reasoning**: This is the pre-registered endgame from exp-report-044: 192's lattice status is the one unverified hardware fact, and 64/192/256 is the only remaining configuration that can measure the one statistical question still open (does capacity above 4.29M params buy plateau level when epochs stay ≥100?). Every outcome is decisive: GATE_KILL → the lattice is exactly {64,128,256}, the baseline is the unique hardware-fast configuration, asymmetric capacity closed on hardware grounds at ~2 GPU-minutes cost; full run sub-bar at mean−deficit with family test_loss → capacity is level-saturated, class closed statistically; ≥96.81 → first improvement in 39 loops through the only historically-positive lever. The honest prior on the bar branch is low (EXP-009 saturation reading, −0.38 deficit headwind), but no other open candidate has ANY path to +0.3, and the class-closing information is permanent either way.

**Sources**: reports/exp-report-044.md § Next Steps + § Unexplored Avenues; goal-learnings § Failed Approaches (capacity entry count 5 — justification for retry: this probes the SAME question with the LAST unconfounded instrument, explicitly flagged by the entry itself); project-insights § Medium (kernel-lattice law); knowledge WRN/RegNet rows.

**Estimated Effort**: low (identical mechanics to EXP-044: 3-integer change, param-count sanity, threshold edit in the composite script).

**Risk Assessment**: Two stacked risks. Hardware: 192 may misprice (EXP-034's 27.4ms hints it; resolved in ~90s by the gate, pre-registered invalid). Statistical: even a gate-passing run fights a −0.38 deficit and the saturation evidence — most likely full-run outcome is ~96.2 (mean − deficit), a graceful class-closing no-improvement. Worst case: nothing destabilizes; the recipe is untouched.

### 2. 96/128/256 — widen stage 1 instead (rejected: off-lattice)
**Summary**: The other w1<w2<w3 widening direction. REJECTED at generation time: 96 is a 32-multiple but not a 64-multiple — it sits exactly on EXP-044's mispricing law, and even at the (inapplicable) dense law it costs +44% FLOPs at the most expensive resolution (~28.3ms, ~110 epochs) for capacity in the stage EXP-017 showed is already representation-adequate.

**Reasoning**: Documented to show the lattice filter applied: every stage-1 widening option (96, 128=+100% FLOPs) is either off-lattice or starvation-priced. There is no viable stage-1 instrument.

**Sources**: reports/exp-report-044.md (alignment law); exp-report-017.md (stage-1 adequacy); project-insights kernel-lattice entry.

**Estimated Effort**: low (same mechanics) — but fails the hardware screen before launch.

**Risk Assessment**: Near-certain GATE_KILL; running it would spend a loop re-measuring a law established yesterday. Discarded.

### 3. Pre-activation block ordering (ResNet v2) — carried rejection
**Summary**: BN→ReLU→conv reorder; the only dt-free structural change never run. Carried from brainstorm-044 Idea 3 with the same three failing screens: no in-regime evidence at depth 20 (published gains concentrate at depth ≥110), the structural-arc deferral/quality signature (EXP-018/020/030), and no mechanism for raising plateau LEVEL (the only thing the max-statistic pays). External fixed-epoch architecture evidence is 0-for-12 transferring.

**Reasoning**: Retained as the comparison point: if the capacity class closes this loop, the next brainstorm must weigh radical low-prior structural changes like this against further hardware-charted dead ends — but it should never outrank a candidate that resolves an open question.

**Sources**: arXiv 1603.05027 (model knowledge); project-insights § High (deferral law); brainstorm-044 § Idea 3.

**Estimated Effort**: medium.

**Risk Assessment**: Expected −0.1 to −0.5 with no level gain; zero class-closing value (the structural arc is already closed). Dominated.

## Idea Evaluation

- **Evidence strength**: Idea 1 is the pre-registered continuation of EXP-044's own Next Steps, instruments the only lever with a historical level gain (+2.07), and rests on a fresh, precise hardware map telling us it is the unique remaining test. Ideas 2 and 3 fail their screens outright (off-lattice; deferral-class with zero in-regime evidence).
- **Mechanism clarity**: Idea 1's mechanism is the sharpest available — added stage-2 channels at ~112 fully-annealed epochs either raise the converged plateau or prove 4.29M params is regularization-saturated; the gate cleanly separates the hardware confound from the statistical answer.
- **Expected impact**: Idea 1 is the only candidate with any route to +0.3; its more likely outcomes (GATE_KILL or mean−deficit) close the last open class permanently, which is the highest-information result available to the program.
- **Risk profile**: Idea 1 is gate-protected (worst hardware case costs ~2 GPU-minutes) and recipe-untouched (worst statistical case is a graceful no-improvement). Safest in the set.
- **Feasibility**: trivial diff, validated launcher, validated sanity pattern.

Idea 1 dominates on every axis.

## Chosen Idea
**Selected**: Idea 1 — 64/192/256 gate-first (the last lattice point)

**Why this idea**:
EXP-044 left exactly one decision: resolve 192's kernel-lattice status and, with it, the only remaining instrument for the capacity-at-converged-epochs question — the single statistical question still unmeasured in 45 experiments, attached to the only lever that ever bought plateau level. The probe is ~90s if the hardware says no; the full run is the first unconfounded capacity test if it says yes; and every branch permanently closes or advances the class. No other candidate passes both the effect-size screen and the hardware screen.

**Hypothesis**:
With `STAGE_WIDTHS = (64, 192, 256)` (+40% FLOPs, all widths on the kernel lattice), the run prices at ≈27.7ms by the dense law (~112 epochs) and the added stage-2 capacity raises the converged plateau by more than its ~0.38 epoch deficit → best_test_acc ≥ 96.81. Pre-registered branches: (i) best ≥ 96.81 → improvement (mid-band read 96.70–96.80 triggers the replicate pair first); (ii) gate passes but plateau ≈ mean − deficit (~96.1–96.3) with family test_loss → capacity is level-saturated at 4× width; the capacity class is closed in ALL currencies and the program moves to radical structural recombination; (iii) GATE_KILL (D0 > 31ms) → 192 also misprices; the H20 fast lattice is exactly {64, 128, 256}, the baseline architecture is the unique hardware-fast configuration, verdict invalid, class closed on hardware grounds.

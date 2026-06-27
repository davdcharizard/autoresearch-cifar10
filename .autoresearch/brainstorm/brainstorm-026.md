# Brainstorm EXP-026
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new searches; grounding from the knowledge base and known high-signal context:

- **knowledge/README § References — airbench (arXiv 2404.00498, github.com/KellerJordan/cifar10-airbench)**: the airbench94 nets use **GELU** activations throughout (not ReLU). This is the highest-authority budget-race reference for CIFAR-10 and it is regime-matched to us (small CNN, seconds-scale budget, channels_last + half precision, batch ~512–1024).
- **knowledge/README § References — cifar10-fast lineage (davidcpage)**: the "How to Train Your ResNet" final nets moved from ReLU to **CELU** — the lineage's author measured smooth activations as an accuracy-per-budget win on CIFAR-10 ResNets.
- **Known context — smooth activations on small-image CNNs (GELU/SiLU literature, e.g. arXiv 1606.08415, 1710.05941)**: smooth activations consistently give small accuracy gains over ReLU on CIFAR-class benchmarks at equal architecture; the mechanism is a smoother loss landscape and non-zero gradients for negative pre-activations (no dead units). Cost is pointwise and fuses into adjacent kernels under torch.compile.
- **Transfer caveat (in-project)**: two prior airbench/speedrun imports failed (whitening init EXP-019, derandomized tricks never attempted) — both failures were DEFERRAL-class (early-budget resources for late payoff) or BN-interaction artifacts. An activation swap is neither: it is active from step 0 and has no BN-ordering interaction (it sits after BN, exactly where airbench places it).

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Twenty consecutive misses (EXP-007…025).**
- **RECIPE-SPACE IS CLOSED** (goal-learnings § Patterns): every train.py constant bracketed alone or in compensated trades; FOUR LAWS bidirectionally evidenced — deferral, numerics equivalence, max-statistic, gradient-noise optimum.
- **EXP-025 (just closed)**: out-of-recipe class #1 — time-structure of the data distribution (FixRes clean tail) — refuted hard (−0.87): the augmented cosine tail is load-bearing; late pressure-down loses worse than constant pressure-down. Mechanism class dead; surviving fragment is forward-only BN-stat recalibration (cannot overfit).
- **NEVER probed in 26 experiments**: the **activation function**. Every architecture probe touched topology (depth/width/shortcuts/allocation) or init — none touched the nonlinearity. The recipe inherits F.relu from the He et al. reference implementation unchanged.
- **Goal-learnings screen for an activation swap**: deferral — GELU is active from step 0, smooth gradients if anything ADD early learning (the inverse of zero-γ's failure) ✓; epochs — pointwise op, inductor fuses it into the preceding BN/conv kernels, dt impact ~0 (must verify via the EXP-008 early dt gate) ✓; numerics law — that law governs same-recipe execution changes (kernel substitution at fixed math); this is a deliberate MODEL change, so it does not apply, but epoch parity must hold ✓; noise — batch/momentum/averaging untouched, noise-neutral ✓. First candidate since EXP-006 that passes all four laws with an affirmative external prior from the exact regime.
- **Capacity laws**: stage-3-only widening flagged "partially rehabilitated" (EXP-017 insight) but still fights the 3× capacity-starvation failure class; activation swap does not touch capacity.

## Candidate Ideas

### 1. Activation modernization: ReLU → GELU throughout the network (airbench-style)
**Summary**: Replace all `F.relu` calls in train.py with `F.gelu` — three sites: the two activations in `BasicBlock.forward` (post-bn1 and post-residual-add) and the stem activation in `ResNet.forward`. Nothing else changes: same init (Kaiming gain √2 is within ~5% of GELU's optimal gain — airbench does not adjust it either), same schedule, same recipe constants, same compile path.

**Reasoning**: The nonlinearity is the single recipe component never probed. Both regime-matched budget-race lineages independently converged on smooth activations (airbench: GELU; cifar10-fast: CELU) — convergent evolution under the same objective (max accuracy per wall-clock second on CIFAR-10) is strong evidence. Mechanism: smooth activations eliminate dead units and give non-zero curvature everywhere, improving the optimization trajectory from step 0 — the gain accrues during the entire run, not in a deferred phase, so the deferral law (which killed every prior architecture import) does not bite. Execution cost: GELU(erf) is pointwise; inductor fuses it exactly where ReLU was fused, so dt should move ≤0.3ms (early gate verifies; at +0.3ms worst case 139→137 epochs, −2 epochs ≈ −0.03pp by EXP-006 arithmetic — negligible against the expected +0.1–0.3 level shift).

**Sources**: knowledge/README § References (airbench, cifar10-fast); arXiv 1606.08415 (GELU); goal-learnings § Patterns (four laws); exp-report-008.md (early dt gate protocol).

**Estimated Effort**: minimal — 3 one-token edits in train.py.

**Risk Assessment**: (a) dt regression >0.5ms would cost ~4 epochs — caught at the early gate (~step 100), killable before budget waste; (b) GELU could interact with the certified LR/noise optimum (the recipe was tuned under ReLU) — a miss is graceful converged no-improvement, and a small one would still leave the SiLU/CELU variants as one-knob follow-ups; (c) torch.compile recompiles a slightly different graph — same input signature, startup unchanged; (d) no BN-ordering trap (activation sits after BN in both reference and our net).

### 2. Stage-3-only width increase: widths (64,128,256) → (64,128,320) — capacity where it is cheapest
**Summary**: Widen only the third stage (8×8 resolution) from 256 to 320 channels (multiple of 64, H20-aligned). ~+56% stage-3 params (~+1.5M) at small FLOPs cost (stage 3 is the cheapest resolution); keep all other constants.

**Reasoning**: The one capacity move goal-learnings explicitly rehabilitates after EXP-017 ("width asymmetry — widen stage 3, keep all stages intact — remains the one untried capacity-where-cheap move"). EXP-017 isolated the [2,3,4] deficit to REMOVED stage-1 depth, not to added stage-3 capacity; this candidate only adds.

**Sources**: goal-learnings § Failed Approaches (EXP-017 entry); project-insights (H20 channel-alignment law, EXP-005).

**Estimated Effort**: low — one constant change in ResNet.__init__ (w3) plus the fc in_features follows automatically.

**Risk Assessment**: fights the strongest failure class on the goal (capacity-without-throughput, 3 High-importance misses); estimated dt +5–10% → ~127–133 epochs (above the 70 floor but every epoch lost costs ~0.015pp); the larger fc/conv kernels also shift VRAM/dt signatures, complicating contention forensics. Graceful failure, lower prior than idea 1.

### 3. Terminal BN-stat recalibration (forward-only clean passes before the final tail evals, budget-charged)
**Summary**: In the final ~5 epochs, before each eval, run ~50 forward-only batches of CLEAN (test-transform) data in train() mode under no_grad to re-converge BN running stats to the test distribution, charging the wall time to total_training_time. Running stats do not affect training (training normalizes by batch stats), so the trajectory is untouched — only what the evaluator sees changes.

**Reasoning**: The surviving fragment of EXP-025's mechanism — captures the BN-alignment component (≤ +0.35, measured from a depressed level) with zero overfitting risk. Honest accounting: forward passes are charged to the 300s budget (~0.5s each, ~2.5s total ≈ 1 epoch lost).

**Sources**: exp-report-025.md § Unexplored Avenues / Next Steps #1; AdaBN-class BN-recalibration (known context).

**Estimated Effort**: low-medium — clean loader (reuse EXP-025's construction), a recalib function, tail trigger logic, budget charging.

**Risk Assessment**: the pure-BN component of EXP-025's +0.35 is unknown and possibly ~0 or negative from the TOP of the augmented plateau (augmented stats may actually be well-matched at convergence); if recalibrated tail evals are WORSE, the max loses its richest harvest window (~−0.1–0.2 exposure). More moving parts than idea 1 (second loader, state juggling) for a weaker prior. Confidence medium-low per exp-report-025.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest external evidence of any remaining candidate: BOTH regime-matched budget-race lineages (airbench GELU, cifar10-fast CELU) independently adopted smooth activations under our exact objective — and unlike the two prior speedrun imports that failed (whitening init, zero-γ), this one has no deferral structure and no BN-interaction caveat. Idea 2 has one rehabilitating sentence against three High-importance capacity failures. Idea 3 has a weak in-project measurement (+≤0.35 from a depressed level, confounded with weight tuning).

**Mechanism clarity**: Idea 1: no dead units + smooth curvature → better per-step optimization from step 0 → higher converged plateau; clear and law-compliant. Idea 2: more parameters where FLOPs are cheap → better fit IF epochs stay ≥ ~130; mechanism clear but historically the trade has never paid. Idea 3: stats-distribution alignment for the evaluator only; mechanism clear but magnitude unknown and possibly adverse.

**Expected impact**: Idea 1: +0.1–0.3pp (literature-typical GELU-over-ReLU on CIFAR at this scale) landing exactly as a plateau-level shift — what the max-statistic rewards. Idea 2: sign uncertain, history says negative. Idea 3: +0–0.2 at best.

**Risk profile**: Idea 1 is the safest radical change available — 3-token diff, dt-gated, graceful failure, and a miss cleanly closes the activation axis with one run. Idea 3 risks degrading the harvest window. Idea 2 perturbs the most signatures.

**Feasibility**: Idea 1 is trivially implementable and leaves the compiled-graph/loader/eval machinery untouched — baseline-comparable signatures keep contention forensics sharp.

## Chosen Idea
**Selected**: Activation modernization: ReLU → GELU throughout (airbench-style)

**Why this idea**:
It is the only untouched component of the entire recipe after 26 experiments, and the only remaining candidate with an affirmative, regime-matched external prior (two independent CIFAR-10 budget-race lineages converged on smooth activations). It passes all four campaign laws affirmatively rather than marginally: active from step 0 (no deferral), pointwise-fused (no epoch cost — verified at the early gate), a deliberate model change (numerics law inapplicable, epoch parity preserved), and noise-neutral. The expected gain is a converged-plateau LEVEL shift, the exact quantity the max-statistic harvests.

**Hypothesis**:
Replacing ReLU with GELU at all three activation sites improves per-step optimization quality from the first step at unchanged execution signatures (dt 22.4±0.3ms, 137–139 epochs, VRAM ≈1613MB, params 4,286,026): the early trail (ep1–5) is at or above the baseline family, the converged plateau forms ≥0.1pp higher, and **best_test_acc ≥ 96.81 with final-7-evals median ≥ 96.6**. A converged miss at clean signatures closes the activation axis (GELU; SiLU/CELU become low-prior one-knob variants) and routes the campaign to the BN-recalibration fragment or the baseline variance replicate.

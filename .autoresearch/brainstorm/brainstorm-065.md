# Brainstorm EXP-065
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **EfficientNetV2 progressive learning** (knowledge/papers/efficientnetv2-progressive-learning.md; arXiv 2104.00298, ICML 2021)
  Ramping regularization low→high over training speeds convergence at equal accuracy; the reg-ramp ablation holds modestly even at fixed image size. The image-size half of this technique already measured ZERO conversion here (EXP-031); the reg-ramp half at fixed 32px is the residual unmeasured claim.
- **Knowledge base re-scan** (knowledge/README.md): tail-side aug profile measured (EXP-025/033), head-side absent from the record across all 65 experiments.

## Experimental History Review

State after EXP-064 (58 consecutive closures; baseline 96.71 @ 1990397, family mean 96.57, σ 0.16, bar 96.81):

- **Every plateau-raising mechanism class is measured-closed**: capacity (width/depth/allocation/lattice), regularization (dose AND type), loss geometry (4 pathways), averaging (weight/function/schedule-free), schedule (family-level), ensembling/multiplicity (all three funding currencies, EXP-063), attention/activations/heads/shortcuts/pre-act, BN constants (two-sided), structural reparameterization (family-level launch pricing, EXP-064), step-time engineering (EXP-021/048: charged step overhead-free; extra steps at plateau worthless).
- **The tail-pressure law is HALF-mapped**: EXP-025 (clean-data tail −0.87), EXP-033 (light-aug tail −0.46) measured that the TAIL must keep full pressure; EXP-055 extended it to the parameter side. The HEAD side — whether the warmup phase needs full aug pressure — has never been measured. The law as written ("neither the data distribution nor any parameter subset may stop moving before budget end") is about the tail; its head-side converse is an open quadrant.
- **Two adversarial precedents pull opposite directions**: EXP-033 showed light-aug phases bank a real one-time alignment gain (+0.48 banked before its tail-freeze cost); EXP-018 showed mechanisms that switch on during peak heat lose (−0.99, zero-γ blocks turning on mid-warmup). An aug ramp banks early progress but hardens the data distribution exactly at peak LR — which precedent governs is exactly the unmeasured question.
- **Pricing**: augmentation runs in CPU loader workers; the charged step is GPU-bound (99.3% kernel time, EXP-048). An aug-profile change has ZERO dt toll — launch-certain, byte-identical run signatures expected (family bands apply directly, no probe-revision needed).
- **Resources audit** (per exp-report-064 next-step 1): charged time fully spent on steps; SM compute unreachable (dispatch-bound, EXP-063); VRAM idle but inconvertible without compute; the only unpriced degrees of freedom left in train.py are time-PROFILES of existing pressures (aug schedule) and micro-structures with sub-screen effect sizes (pooling shape, init flavor, WD coupling form).

## Candidate Ideas

### 1. Warmup-phase augmentation lightening (head-side quadrant of the tail-pressure law)
**Summary**: Disable TrivialAugmentWide + RandomErasing during the LR warmup phase (progress < 0.15 — the schedule's own principled boundary), enable both at full strength for the entire anneal (85% of budget). Crop+flip stay always-on (the EXP-033 "light-aug" floor — retains baseline-class regularization, avoids naked overfit). Implemented with the EXP-041-validated shared-memory flag pattern: a `torch` shared-memory float tensor read inside a small custom transform wrapper (persistent workers see updates via fork-shared memory); the main loop sets the flag once when `total_training_time/TIME_BUDGET_S ≥ WARMUP_FRAC` at an epoch boundary. All other recipe constants byte-identical; dt unchanged.

**Reasoning**: Completes the pressure-profile law to a four-quadrant map (head/tail × light/full) — three quadrants measured, this is the fourth. Mechanism: the warmup phase trains at high noise (LR ramping to 0.4) where regularization contributes least and costs most progress; light-aug epochs are ~"easier" gradient signal, banking alignment (EXP-033 measured the banked gain is real: +0.48) with no tail to forfeit this time — the full-pressure anneal follows. EfficientNetV2's reg-ramp ablation provides external fixed-size evidence. Zero toll → the failure mode is purely informational (family-band null or the EXP-018 peak-heat-transition loss), never starvation.

**Sources**: knowledge/papers/efficientnetv2-progressive-learning.md; EXP-025/033 (tail side), EXP-018 (peak-heat inversion risk), EXP-041 (shared-memory worker flag machinery), EXP-031 (image-size half zero), EXP-048 (aug is uncharged).

**Estimated Effort**: low-medium (transform wrapper + flag + composite run; no probe gate needed — no graph or dt change).

**Risk Assessment**: (a) Effect size honest estimate +0.0–0.3 — below the screen's nominal +0.3 mid; justified as a LAW-COMPLETING measurement with a credible upside path (21 light epochs × banked-alignment rate). (b) EXP-018 inversion: hardening data at peak LR may cost more than banked; pre-register branch interpretation. (c) Transform-switch machinery must be sanity-checked (flag propagation to persistent workers) — EXP-041 pattern is validated. (d) BN stats see a distribution shift at p=0.15 — transient, 100+ epochs of full-aug stats follow (EXP-029 satisfied at eval time).

### 2. Baseline replicate pair — adversarial audit of the ceiling (protocol experiment)
**Summary**: Two byte-identical baseline runs to tighten σ and test whether the recorded 96.71 top (the bar's anchor) reproduces, sharpening the bar's statistical position after 58 closures.

**Reasoning**: EXP-027 established σ ≈ 0.16 from three draws; 58 family-band reads since then implicitly corroborate, making dedicated replicates largely redundant — and the run cannot produce an `improvement` verdict by construction.

**Sources**: EXP-027; goal-learnings § Protocol Findings.

**Estimated Effort**: low.

**Risk Assessment**: No metric upside; spends two full charged runs on protocol that the family record already approximates. Defer unless the idea drought is total.

### 3. Pooling-head micro-variant (GAP → concat[GAP, GMP])
**Summary**: Concatenate global max pooling with average pooling before the fc (256→512-d classifier input), a one-line head change.

**Reasoning**: Head modifications measured negative (multi-scale concat EXP-047 −); pointwise op pricing (EXP-026) makes even the pooling op non-free; no published CIFAR evidence ≥ +0.3 for classification.

**Sources**: EXP-047, EXP-026.

**Estimated Effort**: low.

**Risk Assessment**: Fails the effect-size screen on evidence; EXP-047 triangulates negative. Listed for completeness, not viable.

## Idea Evaluation

**Evidence strength**: Candidate 1 holds the only real evidence among the three — an ICML ablation in its favor (modest), one in-project measurement showing the banked-gain mechanism is real (EXP-033's +0.48), and one in-project inversion precedent against (EXP-018). Candidate 2 has no metric evidence by design; Candidate 3 has negative in-project evidence (EXP-047).

**Mechanism clarity**: Candidate 1's mechanism is sharp and falsifiable: light-aug warmup banks alignment that the full-pressure anneal either compounds (gain) or destroys at the p=0.15 transition (EXP-018-class loss). Either outcome completes the pressure-profile law's fourth quadrant — the measurement is informative in both directions, unusual at this stage of the program.

**Expected impact**: Candidate 1's honest band is +0.0–0.3 (sub-screen mid) but is the largest expected value among all remaining unmeasured interventions; everything stronger has been closed. Candidate 2 is zero by construction. Candidate 3 is negative-prior.

**Risk profile**: Candidate 1 is the safest substantive experiment available: zero dt toll, launch-certain (breaks two consecutive NO-LAUNCH loops), family bands apply unchanged, failure is a clean quadrant closure.

**Feasibility**: The only nontrivial machinery (mid-run transform switch under persistent workers) is EXP-041-validated; everything else is reusable.

## Chosen Idea
**Selected**: Warmup-phase augmentation lightening (head-side quadrant of the tail-pressure law)

**Why this idea**:
It is the last unmeasured time-profile degree of freedom of the existing recipe, it is the only remaining candidate with any supporting evidence (external ablation + in-project banked-gain measurement), it costs zero charged toll with byte-identical signatures (launch-certain), and it is informative in BOTH outcomes — completing the pressure-profile law's fourth quadrant either as a gain mechanism or as a data-side confirmation of the EXP-018 peak-heat-transition law.

**Hypothesis**:
Disabling TA+RE during the warmup phase (p < 0.15, ~21 epochs) and running the full recipe for the entire anneal banks early alignment at the phase where regularization is least needed, raising best_test_acc above the family band — pair-mean ≥ 96.81 under the EXP-052 protocol if the banked gain (~+0.3–0.5 by the EXP-033 alignment-rate analogy) survives the p=0.15 transition. If instead the transition at peak LR destroys the banked progress (EXP-018-class), the run reads ≤ 96.41 and the head quadrant closes with the law extended to "pressure must be full-on from step 0".

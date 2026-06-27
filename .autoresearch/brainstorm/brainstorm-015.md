# Brainstorm EXP-015
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **L2 Regularization versus Batch and Weight Normalization (van Laarhoven, 2017)** (https://arxiv.org/abs/1706.05350)
  With BN, conv-weight L2 does not constrain the function (predictions are scale-invariant in w); its real effect is on the EFFECTIVE learning rate: WD shrinks ||w||, and since gradient steps scale ~1/||w||², smaller norms mean larger relative steps and more gradient noise.
- **Three Mechanisms of Weight Decay Regularization (Zhang et al., ICLR 2019)** (https://arxiv.org/abs/1810.12281)
  For scale-invariant (BN) networks the dominant WD mechanism is effective-LR control; the gradient-noise increase acts as a stochastic regularizer. Confirms our WD on conv weights (selective, ndim>1) is mostly an optimization-dynamics knob, not a direct capacity penalty.
- **Why Do We Need Weight Decay in Modern Deep Learning? (NeurIPS 2024)** (https://arxiv.org/abs/2310.04415)
  Modern view: for over-parameterized nets WD changes training dynamics rather than acting as classical regularization; without WD, weight norms grow and the effective LR decays, hampering training — so the right dose is setup-specific, not transferable folklore.
- **Knowledge base** (.autoresearch/knowledge/README.md; papers/why-warmup-lr.md)
  The warmup-paper caveat (EXP-014) applies generically: fixed-iteration intuitions about schedule/optimizer constants can invert under the time-keyed budget — predictions below are made in TIME-domain terms accordingly.

## Experimental History Review

- Current best: **96.71** (EXP-006 recipe @ 1990397). Nine consecutive no-improvements (EXP-007…014).
- Measured boundaries (goal-learnings): capacity closed bidirectionally; augmentation additions saturated (4-point dose-response, ANY diversity increase = pressure); batch/throughput closed (1024+linear LR neutral); EMA collapses the max-statistic; **integrated LR heat closed from the HOT side twice** (peak +50% → −0.57pp EXP-010; warmup-halving → −0.22pp EXP-014 — count-2 Medium failed approach).
- **The single never-probed recipe constant is WEIGHT_DECAY = 5e-4** (set in EXP-000, inherited through all 15 experiments). Every other constant (peak, warmup, batch, LS-by-composition, augmentation set, width, depth) now has at least one measured probe.
- Two open mechanism questions the history cannot answer: (1) is total regularization pressure past optimum (the augmentation curve crossed zero between TA and reflect — "at optimum" reading) or exactly at it? (2) is the heat optimum strictly below current (two hot-side losses with monotone dose-response suggest non-positive marginal heat slope) or at current? A WD probe gives evidence on BOTH (see Candidate 1); a cold peak probe answers only (2).
- Protocol: contention protocol mandatory (EXP-011/014 — inline watchdog in the launch chain, post-hoc windowed profile); no added per-image CPU work (loader margin ~3%, EXP-013); max-statistic rewards variance (EXP-011) — variance-reducing side effects must be netted against mean gains.

## Candidate Ideas

### 1. REDUCE WEIGHT DECAY: WEIGHT_DECAY 5e-4 → 2.5e-4
**Summary**: Halve the selective weight decay on conv/linear weights (BN/bias groups already at 0). One-constant diff; throughput, memory, data pipeline, schedule all byte-identical.

**Reasoning**: Three independent arguments converge. (a) LAST UNMEASURED AXIS: after 15 experiments WD is the only constant never probed — even a clean negative completes the recipe's measurement map. (b) PRESSURE-DOWN: the four-point augmentation dose-response says marginal regularization pressure at the current dose is negative; WD is the only component that can move pressure DOWN without touching the saturated data-augmentation set or the loader (EXP-013 margin). (c) MECHANISM REFRAME (new, from literature): with BN, conv-weight WD acts chiefly through the effective LR — halving WD lets ||w|| grow, SHRINKING effective step size and gradient noise late in training. That makes this also a mild COOLING move, probing the untested cold side of the heat curve that EXP-010/014 bracketed from above. 5e-4 is 2016 folklore tuned for 64-epoch step-decay runs without TA/RE/LS; nothing says it is right for a 139-epoch one-cycle under triple augmentation.

**Sources**: arXiv 1706.05350, 1810.12281, 2310.04415; goal-learnings § Patterns High (dose-response), § Failed Approaches Medium (heat, EXP-010/014); reports/exp-report-013/014.md § Next Steps.

**Estimated Effort**: low — one-constant diff, ~483s runtime, baseline signatures expected (dt ~22.3ms, 139 epochs, 1613MB).

**Risk Assessment**: Graceful failure (no-improvement). Two ways to lose: (i) the recipe is AT the regularization optimum (augmentation curve's zero-crossing reading) and less WD over-fits the 139-epoch tail — best lands mid-schedule but the max-statistic tolerates that; (ii) the effective-LR cooling reduces eval variance at convergence, shaving the max (EMA lesson, smaller magnitude). Worst case ≈ −0.2 to −0.3pp converged. No crash/cap risk.

### 2. COLD-SIDE PEAK PROBE: PEAK_LR 0.4 → 0.35
**Summary**: Single-constant probe of the heat curve's untested side; warmup shape unchanged.

**Reasoning**: The two hot-side losses (−0.22 small increment, −0.57 large) imply a non-positive marginal heat slope at the current configuration, so a small heat REDUCTION could sit closer to the optimum. −12.5% peak is the mirror-image of EXP-014's increment.

**Sources**: goal-learnings § Failed Approaches Medium (heat dose-response, EXP-010/014); exp-report-014.md § Unexplored Avenues.

**Estimated Effort**: low.

**Risk Assessment**: Graceful. Counter-evidence: EXP-006's healthy convergence shape at 0.4 suggested near-optimality, and a colder run makes less early progress (mild deferral risk in the other direction). The curve may be flat to the left (like the augmentation curve) — small expected magnitude either way.

### 3. COMPENSATED SCHEDULE RESHAPE: WARMUP_FRAC 0.08 + PEAK_LR 0.35
**Summary**: Combine EXP-014's shorter warmup with a lowered peak chosen so integrated heat is roughly baseline-level — isolates the anneal-LENGTH effect from the heat effect that confounded EXP-014.

**Reasoning**: EXP-014's mechanism analysis showed warmup-shortening was a heat increase in disguise; if a longer anneal has any independent positive effect, this surfaces it at ~constant heat. Scientifically the cleanest follow-up on the schedule axis.

**Sources**: reports/exp-report-014.md § Unexplored Avenues; goal-learnings § Failed Approaches Medium.

**Estimated Effort**: low (two-constant diff).

**Risk Assessment**: Graceful, but a two-variable change on an axis with two fresh losses; the "rough" heat compensation is eyeballed (no closed-form equivalence), so a null result is ambiguous between "anneal length is worthless" and "compensation was off". Weakest inference per run.

## Idea Evaluation

- **Evidence strength**: Idea 1 leads. It is backed by the project's own strongest internal signal (the only unmeasured constant + the dose-response saturation argument) AND a coherent external literature (three papers agreeing WD-with-BN is an effective-LR knob whose right dose is setup-specific — i.e., the inherited 5e-4 has no special claim to optimality here). Idea 2 rests on a two-point slope extrapolation; idea 3 on a mechanism decomposition with an eyeballed compensation.
- **Mechanism clarity**: Idea 1's dual mechanism is explicit (pressure down + late-schedule effective-LR cooling), and crucially BOTH sub-mechanisms point in directions the history says are the live ones (pressure has negative marginal value; heat has non-positive marginal value). Idea 2 is single-mechanism; idea 3's mechanism is clean but its measurement is confounded by the compensation guess.
- **Expected impact**: All sub-half-point. Idea 1 is the only one that can settle TWO open questions in one run (pressure optimum AND a cooling data point); ideas 2/3 each answer one.
- **Risk profile**: Identical — graceful no-improvement, zero cap/loader/crash exposure, byte-identical signatures.
- **Information value on failure**: Idea 1's failure modes are diagnosable from the trajectory (over-fit tail = pressure was at optimum; flat-but-lower peak = variance shaved), feeding directly into whether idea 2 is worth running next loop. A two-constant idea-3 failure is the least interpretable.

Idea 1 wins on evidence, mechanism, and information value; ideas 2/3 are sequenced behind it.

## Chosen Idea
**Selected**: REDUCE WEIGHT DECAY: WEIGHT_DECAY 5e-4 → 2.5e-4

**Why this idea**:
It is the single never-probed recipe constant after 15 experiments, the only untried lever that moves regularization pressure DOWN (the direction the saturated 4-point augmentation curve points), and — per the WD-with-BN literature — simultaneously a mild effective-LR cooling probe on the heat curve's untested cold side. One-constant diff, graceful failure, and its failure trajectory disambiguates which follow-up (cold peak vs nothing) is worth running.

**Hypothesis**:
Halving WD to 2.5e-4 reduces excess regularization pressure (and slightly cools the late-schedule effective LR), letting the converged tail sit higher: best_test_acc ≥ 96.81 (baseline 96.71 + 0.1) at byte-identical throughput signatures (~139 epochs, dt ~22.3ms, ~1613MB). Diagnostic sub-predictions: train loss runs lower than baseline from mid-schedule; if instead the tail OVER-fits (test_acc peaks then decays while train loss keeps falling), the recipe was at the pressure optimum and the axis closes.

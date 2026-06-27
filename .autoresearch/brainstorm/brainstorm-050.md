# Brainstorm EXP-050
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + model knowledge:

- **Project's own measured limitation (the anchor)**: goal-learnings § Failed Approaches, EXP-011/032 entry — "the accuracy ceiling is decision-boundary-limited, not confidence/calibration-limited — solution-smoothing moves logit geometry, not argmaxes." This is the only mechanism-level statement the record makes about WHAT bounds the metric at the current recipe. No experiment in 50 loops has ever applied pressure to decision boundaries THROUGH THE LOSS — the only loss-level probes were target-distribution changes (LS 0.1→0.2, EXP-036 absorbed-null; mixup, EXP-009 −0.46). Logit-geometry and per-sample-weighting classes: unmeasured.
- **Large-margin softmax literature (model knowledge)**: L-Softmax (Liu et al., ICML 2016) and AM-Softmax/CosFace (Wang et al. 2018) show additive/angular margins on the true-class logit improve CIFAR-10/100 and face-verification generalization by directly enlarging inter-class margins. The additive-margin form is loss-only: subtract m from the true-class logit before CE; eval untouched. Caveat honored: those gains predate TrivialAugment-strength augmentation — the absorption law applies, but margin pressure is a mechanism augmentation cannot supply (aug moves inputs; margin reshapes the loss gradient AT the boundary for every sample, all training long).
- **"When Does Label Smoothing Help?" (Müller et al., NeurIPS 2019, model knowledge)**: LS caps the converged true-class logit gap at log((1−ε+ε/K)/(ε/K)) ≈ 4.51 for ε=0.1, K=10 — gives a principled scale for choosing the margin dose (m as a fraction of the LS-implied gap), and confirms LS+margin compose (LS shapes targets, margin shifts the gap optimum up by exactly m).
- **Focal loss caveat (model knowledge, Lin et al. 2017)**: hard-example UP-weighting is known to be neutral-to-negative on balanced clean classification; under TrivialAugment+RandomErasing many high-loss samples are augmentation-destroyed (label effectively wrong), making naive focal weighting actively risky here. Shapes Idea 2's form (band-pass, not high-pass) and its runner-up status.

## Experimental History Review

State after 50 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 43 consecutive non-improvements. EXP-049 completed the recipe-constant audit (LR optimum flat over [0.3, 0.4]; heat axis closed both directions). The full frontier:

- **Closed**: every recipe constant (bracketed or measured-flat); all structural classes (absorbed-null 046 / active-negative 030, 047 / cost-priced 020, 037, 040–045); throughput (048: 99.3% irreducible kernel math); gradient noise (implicit + explicit, 011/022/023/024/035); weight/function averaging (011, 032, 043); init (018, 019); BN constants (038, 039); activations (026); optimizer (028); width lattice (001, 002, 005, 007, 040–045).
- **Unmeasured classes remaining**: (a) loss-geometry — logit margins, per-sample weighting (never probed; only target-distribution probes exist: 036, 009); (b) compound interventions of certified components (EXP-009 precedent negative); (c) auxiliary training signals (self-distillation — never probed, but charged-time-priced, see Idea 3).
- **Laws every candidate must pass**: absorption (must supply a mechanism aug cannot); deferral (+1ms ≈ −6 ep ≈ −0.08pp); heat closed (a candidate must be heat-neutral or it confounds); max-statistic (only converged plateau level pays); detection arithmetic (true effect must be ≥ +0.3 for one-draw detection at bar = mean + 1.5σ).
- Protocol carry-overs: composite gates (26ms), step-count ledger, trajectory-criterion numerics, replicate band 96.70–96.80.

## Candidate Ideas

### 1. Additive logit margin on the true class (training-loss-only, single dose knob)
**Summary**: One line in the training loss: `loss = F.cross_entropy(outputs - MARGIN * F.one_hot(targets, NUM_CLASSES), targets, label_smoothing=0.1)` with `MARGIN ≈ 0.75` (a new constant). The same margin applies in the compile-warmup loss line for graph identity. Eval, model, schedule, data — all byte-identical. Cost: one fused elementwise subtract on a 512×10 tensor, < 0.01ms (inductor folds it into the CE kernel).

**Reasoning**: Targets the project's only measured mechanism-level limitation — decision-boundary-limited accuracy (EXP-011/032 insight) — through the only unmeasured intervention class that is free in every currency. Mechanism: subtracting m from the true-class logit makes every sample look "m less confident" to the loss, so gradient pressure persists until the true-class gap exceeds the LS-implied optimum PLUS m — converged boundaries sit m further from training points in logit space. This is pressure augmentation cannot supply (absorption-law test: aug perturbs inputs; margin reshapes the loss landscape at the boundary for every sample identically). Heat-neutral: no LR/steps/noise change; the loss offset is constant per sample, and gradient magnitude changes only through the softmax saturation point — the same channel any loss choice uses. Dose: m = 0.75 ≈ 17% of the LS-converged gap (4.51) — large enough to move boundaries, small enough not to fight LS (m ≥ gap would make the optimum unreachable). Failure shape: under heavy aug, destroyed-label samples also get pushed harder (their "true" class is wrong) — if that dominates, plateau lands below mean (clean branch-iii negative).

**Sources**: goal-learnings EXP-011/032 entry (decision-boundary-limited); model knowledge: L-Softmax ICML 2016, AM-Softmax 2018, Müller et al. 2019 (gap arithmetic); reports/exp-report-036.md (LS dose absorbed — target-distribution probes don't reach boundary geometry).

**Estimated Effort**: trivial — one new constant + one changed loss line (×2 with warmup), standard gated composite.

**Risk Assessment**: Safest possible class: byte-identical signatures, zero cost, graceful failure. Branches: (i) ≥ 96.81 improvement; (ii) mean band → loss-geometry class absorbed at this dose, boundary limitation not addressable via static margins; (iii) < 96.42 → margin actively harmful under heavy aug (destroyed-label amplification measured); (iv) gate/contention infra screens.

### 2. Boundary band-pass per-sample reweighting (w ∝ p_true(1−p_true), heat-normalized)
**Summary**: Per-sample CE with `reduction='none'`, weights w_i = 4·p_i(1−p_i) (p_i = softmax prob of the true class, detached), renormalized to mean 1 per batch (heat-neutral by construction), then weighted mean.

**Reasoning (and why not the lead)**: Same boundary-targeting intent, dynamic rather than static: maximal weight at p=0.5 (genuine boundary samples), suppressed at both p→1 (trivially easy) and p→0 (augmentation-destroyed labels — the focal-loss failure mode under TA+RE is explicitly band-passed away). But: (a) no direct literature anchor for this exact form (invented composite — weaker evidence than margin's published record); (b) two coupled choices (weight shape AND normalization) make attribution mushier than Idea 1's single constant; (c) the detached-weight pathway adds a softmax+gather to the charged step (still < 0.05ms, fine). A clean follow-up if Idea 1's branch (ii)/(iii) suggests dynamic dosing.

**Sources**: focal loss (Lin et al. 2017) for the up-weighting caveat; self-paced learning literature for the down-weighting side; goal-learnings absorption entries.

**Estimated Effort**: trivial-plus (4–5 lines; needs care that weights are detached and normalized).

**Risk Assessment**: Graceful failure; slight risk of EMA-loss display distortion (cosmetic). Dominated by Idea 1 on evidence and attribution this loop.

### 3. EMA-teacher self-distillation — documented reserve (priced out by the deferral law)
**Summary**: Maintain an EMA copy of weights; add KL(student ‖ EMA-teacher) auxiliary loss each step.

**Reasoning (and why rejected this loop)**: The only other unmeasured signal class (dark-knowledge soft targets — genuinely something aug cannot supply). But the teacher forward pass costs ~6ms inside the charged step (≈ 27% of dt); by the deferral law that's ≈ −36 epochs ≈ −0.7pp paid up front, so the mechanism must deliver > +1.0pp true effect to clear the bar — larger than any published self-distillation gain on CIFAR-10 at this accuracy level (typ. +0.3–0.7 under fixed-epoch, not time-charged, budgets). Every cheap variant degrades the signal (periodic teacher forwards dilute; previous-step logit reuse is label echo, not dark knowledge). Recorded so the arithmetic isn't re-derived.

**Sources**: deferral law (goal-learnings); EXP-048 (charged step 99.3% kernel math — no overhead room to hide a teacher forward); self-distillation literature (model knowledge: Born-Again Networks, Zhang et al. self-distillation).

**Estimated Effort**: moderate — and rejected regardless on cost arithmetic.

**Risk Assessment**: Cost-priced negative expectation; would also confound throughput. Reserve only if a zero-cost teacher source ever appears.

## Idea Evaluation

- **Evidence strength**: Idea 1 has a published anchor (margin losses improving CIFAR generalization) AND an internal mechanism anchor (the decision-boundary-limited insight is the project's own measurement). Idea 2 is an invented composite with only adjacent literature. Idea 3 has strong literature but fails the project's own cost law before it starts.
- **Mechanism clarity**: Idea 1's is exact and arithmetic: converged gap optimum shifts from 4.51 to 4.51 + m; boundaries move m further from training points. Idea 2's mechanism is the same in spirit but its dynamics are harder to predict. Idea 3's mechanism is clear but priced out.
- **Expected impact**: honestly small-to-moderate; sized against the ≥ +0.3 detection requirement, margin losses are one of few published mechanisms with reported CIFAR gains of that order at zero cost. Both branches (ii) and (iii) open/close the entire loss-geometry class — high information value either way.
- **Risk profile**: Idea 1 is the safest in the remaining space: signatures byte-identical, one constant, graceful failure, no heat confound.
- **Feasibility**: one constant + one line (Idea 1) vs 4–5 lines (Idea 2) vs moderate (Idea 3).

Idea 1 dominates. Idea 2 recorded as the dynamic follow-up; Idea 3 recorded so its cost arithmetic is never re-derived.

## Chosen Idea
**Selected**: Idea 1 — Additive logit margin on the true class (MARGIN = 0.75, training-loss-only)

**Why this idea**:
It is the first probe of the last zero-cost unmeasured class (loss geometry), aimed at the only mechanism-level limitation the project has actually measured (decision-boundary-limited ceiling, EXP-011/032), with a published anchor, a single-constant dose, byte-identical signatures, and a clean pre-registered branch for every outcome. It passes every law: supplies a mechanism augmentation cannot (absorption), costs nothing (deferral), changes no heat (heat axis), and pays only at converged plateau level (max-statistic).

**Hypothesis**:
Subtracting MARGIN = 0.75 from the true-class logit in the training loss (and warmup loss) shifts the LS-converged logit-gap optimum from 4.51 to 5.26, pushing decision boundaries further from training points and raising converged test accuracy: best_test_acc ≥ 96.81 if boundary placement is the binding limitation at the current recipe. Pre-registered branches: (i) best ≥ 96.81 → improvement (replicate pair first if 96.70–96.80); (ii) mean band 96.42–96.72 → static margin absorbed; loss-geometry class closed at this dose with a no-effect read; (iii) < 96.42 → margin actively harmful under heavy aug (destroyed-label amplification); class closed from below; (iv) GATE_KILL/contention → infra (an elementwise subtract cannot alter dt; standard screens).

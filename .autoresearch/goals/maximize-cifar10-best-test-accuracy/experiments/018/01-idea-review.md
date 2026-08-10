# EXP-018 Adversarial Idea Review

## Prioritized Critique

### 1. Late SWA: The averaging window sits in the worst possible part of the schedule

- **Unverified dependency**: SWA/LAWA benefits from averaging diverse iterates that move around a basin. The accepted schedule cosine-decays to `1e-4`, while proposal idea-01 averages snapshots in `[90%,98%]`, where movement progressively shrinks.
- **EXP-010 interaction**: the accepted tail improved toward a final-equals-best result. Equal averaging of an improving, increasingly converged trajectory may pull the solution backward, before also paying the roughly 2% exposure cost.
- **Refinement**: consider averaging over the higher-LR 80-90% interval where iterates have greater spread, or use a textbook constant/cyclic SWA window. Either changes the proposal but better exercises the averaging mechanism.

### 2. Late SWA: The defensible success mechanism is NLL/calibration, not just top-1

EXP-017 indicates that late confidence/generalization, not short-phase fit, is the active limiter. SWA is best supported as an NLL/calibration and flat-solution intervention. Pre-register final NLL against EXP-010's 0.1934 as the main mechanistic diagnostic while retaining top-1 as the goal's required primary metric.

### 3. Late SWA: Online checkpoints remain in `best_test_acc`

Online evaluations through finalization still contribute to the maximum, while reserving 2% for BN refresh removes the accepted trajectory's final SGD interval. The proposal correctly forbids adding both an online and SWA terminal evaluation, which avoids reward hacking, but a positive outcome requires explicit online-versus-SWA attribution. A snapshot-count gate alone does not establish meaningful iterate diversity.

### 4. ECA: Mechanistically misaligned with the stated limiter

ECA is another representation/capacity lever after EXP-012/015/017 showed that changing fit or residual representation need not improve late generalization. If pursued, final NLL and tail slope matter more than switch fit; a suppress-only gate could be closer to the diagnosis than a `(0,2)` amplifying gate.

### 5. ECA: CutMix descriptor ambiguity and artificial channel adjacency cap upside

Global pooling blends two CutMix regions into one gate descriptor, while length-5 channel convolution assumes locality in an arbitrary learned channel order. The proposal's identity initialization, first-update bounds, real-batch trajectory gate, and timing discipline are excellent, but target selection is weaker than execution quality.

### 6. Nesterov: Clean test, no positive accuracy evidence

The one-keyword change cleanly resolves EXP-001's confound and is likely compute-neutral, but the cited SGD-noise evidence does not establish a Nesterov gain. Historical evidence warns that faster optimization need not improve generalization. It is best framed as a low-cost confound-resolution probe, with special attention to the 80% momentum/distribution transition, not as a high-confidence accuracy intervention.

### Cross-Cutting Notes

- None of the candidates violates the hard constraints.
- All three include paired timing and production-distribution safety gates informed by EXP-013 through EXP-016.
- No proposed double-evaluation or other metric-opportunity reward hack survives in the finalists.

## Scored Verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---:|---:|
| Late Arithmetic SWA + BN recalibration | 8/10 - strongest literature and only candidate directly targeting late generalization/calibration | 6/10 - plausible modest benefit, limited by correlated low-LR snapshots and exposure cost |
| Final-stage ECA recalibration | 5/10 - rigorous engineering but only directional transfer evidence and weaker limiter alignment | 5/10 - small plausible gain, with meaningful risk of repeating EXP-017's fit/NLL tradeoff |
| Isolated Nesterov | 4/10 - exact deconfounding but no positive metric evidence | 3/10 - likely noise-level effect around a ten-example threshold |

## Strongest Pick

**Late Arithmetic SWA with In-Budget BN Recalibration**.

It is the only finalist that directly attacks the diagnosed late generalization/confidence limiter, has the strongest literature support, explicitly solves BatchNorm-statistics validity inside the fixed budget, and avoids a second terminal metric opportunity. The pick is conditional on addressing the averaging-window weakness and making final NLL/calibration the primary mechanism diagnostic. If a meaningful earlier window cannot be justified without compromising isolation, isolated Nesterov is a defensible low-cost alternative, but it remains materially less supported.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06


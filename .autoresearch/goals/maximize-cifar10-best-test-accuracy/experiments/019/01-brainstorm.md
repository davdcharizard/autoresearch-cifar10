# Brainstorm EXP-019
**Created**: 2026-08-06

## Web Search & Literature Review

- **Towards Understanding How Momentum Improves Generalization in Deep Learning** (`experiments/019/papers/momentum-generalization.md`; https://proceedings.mlr.press/v162/jelassi22a.html)
  ICML 2022 gives momentum a shared-feature/generalization mechanism under heterogeneous margins, but does not directly compare ordinary with Nesterov momentum.
- **Accelerated Gradient Descent Escapes Saddle Points Faster than Gradient Descent** (`experiments/019/papers/nesterov-nonconvex.md`; https://proceedings.mlr.press/v75/jin18a.html)
  COLT 2018 supports single-loop Nesterov-style nonconvex exploration, while its algorithm and objective remain only directional evidence for PyTorch SGD on CIFAR.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`)
  Input/target interpolation improves CIFAR generalization by encouraging local linearity, offering geometry complementary to accepted regional CutMix.
- **CutMix** (`knowledge/papers/cutmix.md`)
  Regional class-bearing replacement produced the current local frontier; any mixture should preserve its total mixed-batch probability rather than repeat EXP-011's stronger regularization.
- **How to Scale Your EMA / weight averaging** (`experiments/018/papers/ema-scaling.md`; `knowledge/papers/weight-averaging.md`)
  EMA can improve supervised robustness/generalization, but EXP-018 directly shows that a lagging uniform mean over this cosine tail is harmful.

## Experimental History Review

- EXP-010 remains the 94.15% frontier with width-2 postactivation ResNet-20, all-parameter decay 1e-4, N1/M7 plus 50% alpha-1 CutMix through 80%, then hard weak cosine refinement. Preserve this complete contract unless the candidate isolates one declared lever.
- Stronger CutMix probability, alternate decay, early weak switching, Cutout, label smoothing, full preactivation, and zero-gamma all failed. Identity-oriented residual changes are now a recurring failure family; short-fit improvement is not reliable evidence.
- EXP-017 improved switch/first-weak fit but worsened late NLL; EXP-018 averaged eight genuinely separated tail states and still degraded its own online checkpoint by 0.17. Better fit and generic smoothing are not the missing mechanism.
- EXP-001 bundled Nesterov with the harmful 15% LR hold, so Nesterov's sign remains unresolved. Its one-keyword scope is unusually attributable, but evidence for a top-1 gain is modest and the first update is 1.9x ordinary momentum.
- Mixup has never been tested. Replacing some CutMix events while keeping exactly 50% total mixed batches can change invariance geometry without repeating EXP-011's probability increase.
- Systems constraints are unchanged: model backward is 75.46% of counted cost, loader overhead is hidden, and memory is abundant. Small optimizer/collator changes have the best exposure odds; sequential attention or custom BN must prove their cost.

## Collected Ideas

- **Isolated Nesterov momentum** - Add only `nesterov=True` to accepted SGD. This cleanly resolves EXP-001's confound and changes the temporal filter on the identical hard/CutMix gradient stream, with nearly zero model cost; first-update overshoot and noise amplification are the primary risks.
- **Balanced Mixup/CutMix geometry** - Keep 50% of strong batches mixed but assign a fixed half of those events to Mixup and half to CutMix, retaining 50% hard batches. This imports global interpolation without increasing mixed-target frequency, potentially complementing regional occlusion while avoiding EXP-011's stronger total regularization.
- **Short-timescale tail EMA** - Replace uniform lagging SWA with a pre-registered recent-step EMA whose half-life is tied to one weak epoch. It could smooth local noise without averaging far-back inferior states, but decay selection, BN buffers, per-step cost, and final attribution are harder than the literature headline suggests.
- **Deterministic pool-first Option-A** - Replace transition slicing with 2x2 average pooling while retaining zero channel padding and no learned projection/BN. This isolates spatial anti-aliasing from EXP-017's harmful learned normalized basis at low parameter cost, though pooling backward still consumes exposure.
- **Identity-scale final-stage ECA** - Gate only the three semantic layer3 residual outputs with zero-start `2*sigmoid` ECA. It preserves initial function and transitions, but global descriptors conflict with CutMix regions and prior evidence says another fit-side representation lever may miss late generalization.
- **Ghost BN at optimizer batch 128** - Split BN statistics into virtual groups of 64 while keeping gradient batch, LR, and data fixed. This changes normalization noise rather than batch throughput, but batch 128 may already be in the healthy regime and custom reshaping touches every backward path.
- **Strong-phase magnitude taper** - Retain RandAugment through 80% but reduce magnitude from 7 to a fixed lower value only during the final strong epochs. This creates a gentler distribution transition without early weak data, yet introduces a second worker lifecycle/config boundary and resembles the failed early-adaptation family.
- **Late stochastic depth** - Apply a small fixed drop-path rate only in the weak tail, preserving strong fit while ensembling residual subnetworks during refinement. This is a moonshot because the tail is short, BN statistics see changing graphs, and identity-oriented changes have repeatedly underfit.

## Combinations

- **Nesterov + balanced Mixup/CutMix**: Nesterov may preserve shared features across two mixed-target geometries while the geometry mixture broadens invariance. The mechanisms could complement, but combining before isolated tests would make first-step and generalization attribution impossible.
- **Short EMA + Nesterov**: EMA could smooth Nesterov's stronger current-gradient response. This directly combines exploration and stabilization, but EXP-018 warns that smoothing can lag and the two optimizer-state mechanisms require separate evidence first.
- **Pool-first Option-A + final-stage ECA**: deterministic anti-aliasing retains spatial samples while semantic recalibration changes only late channels. It avoids EXP-017's learned transition basis, but remains a two-component representation experiment with unclear causal credit.

## Candidate Ideas

### Balanced Mixup/CutMix Geometry
**Summary**: Preserve the accepted 50% hard-batch probability during the strong phase, but split the mixed half into 25% alpha-1 CutMix and 25% alpha-0.2 Mixup using one worker-local categorical draw. The weak tail remains hard and Mixup-free. See `proposals/idea-02.md`.

**What it targets**: Generalization geometry under the short strong-view phase without increasing the soft-target rate that caused EXP-011 to underfit. It combines regional class-bearing occlusion with whole-image vicinal linearity while retaining a 50% hard anchor.

**Reasoning**: Accepted CutMix improved the frontier by 0.60 points, while increasing its probability to 0.75 lost accuracy. Mixup supplies a materially different invariance; replacing half of accepted CutMix events, rather than adding more mixed events, is the cleanest way to test complementarity without repeating the known regularization failure.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/mixup.md`; `knowledge/papers/cutmix.md`; EXP-010 and EXP-011 analyses.

**Estimated Effort**: medium

**Risk Assessment**: Global interpolation may blur already augmented spatial evidence, alpha 0.2 and the 25/25 split remain unvalidated choices, forkserver RNG/provenance must be exact, and reduced regional occlusion could surrender CutMix's localization benefit.

### Isolated PyTorch Nesterov Momentum
**Summary**: Add only `nesterov=True` to the accepted single SGD group while preserving momentum 0.9, coupled all-parameter decay 1e-4, model, data, schedule, timer, and evaluator. This cleanly resolves the unresolved EXP-001 schedule/Nesterov confound. See `proposals/idea-01.md`.

**What it targets**: Online optimization geometry and basin selection at essentially unchanged model cost. It retains the complete 94.15% recipe and changes only how the current decay-augmented gradient combines with the momentum buffer.

**Reasoning**: PyTorch's installed recurrence adds the current direction to the historical buffer without a second gradient evaluation. Momentum literature supports shared-feature filtering and Nesterov-style nonconvex exploration only directionally; it does not establish that this stochastic recurrence beats ordinary momentum here. The candidate's strongest evidence is attribution and the fact that its sign remains locally unresolved.

**Sources**: `proposals/idea-01.md`; `papers/momentum-generalization.md`; `papers/nesterov-nonconvex.md`; EXP-001, EXP-002, EXP-010, and EXP-018 analyses.

**Estimated Effort**: low

**Risk Assessment**: The first update is exactly 1.9x ordinary momentum, the current-gradient correction may amplify CutMix/RandAugment noise, coupled decay dynamics change, and faster fit need not improve the late generalization bottleneck.

### One-Epoch-Half-Life Weak-Tail EMA
**Summary**: Starting at 90% counted progress, copy online parameters into an FP32 CUDA shadow and update it every step with fixed decay `0.5 ** (1/390)`. Existing late evaluations temporarily install EMA parameters with current online BN buffers and replace, rather than supplement, online evaluations. See `proposals/idea-03.md`.

**What it targets**: Sub-epoch weak-tail SGD noise and late calibration using a far more recent weighting kernel than the harmful uniform EXP-018 SWA. The online optimizer still runs through 100% and all EMA/copy overhead is charged to training time.

**Reasoning**: EXP-018 showed that a broad uniform 86-98% mean lagged a still-improving trajectory. A one-epoch half-life gives exponentially little weight to those older states while retaining possible local smoothing. The proposal pre-registers a training-only BN agreement gate because current online moments are only an approximation for EMA parameters.

**Sources**: `proposals/idea-03.md`; `experiments/018/papers/ema-scaling.md`; `knowledge/papers/weight-averaging.md`; EXP-010 and EXP-018 analyses.

**Estimated Effort**: high

**Risk Assessment**: Even short averaging may lag the improving tail, current BN moments may be invalid, per-step shadow updates cost exposure, and EMA-only late evaluation can hide a better online endpoint.

## Review

Mandatory external Claude review completed successfully and is preserved verbatim in `01-idea-review.md`. It ranked balanced Mixup/CutMix first (evidence 7/10, impact 8/10), isolated Nesterov second (6/10, 4/10), and weak-tail EMA third (3/10, 3/10).

The decisive criticism was that the proposed Mixup `alpha=0.2` under-delivers its claimed geometry: a fixed-seed two-million-sample Beta probe found only 13.37% of draws in `[0.3,0.7]` and mean minor-class mass 0.1011. Thus, replacing half of accepted CutMix with alpha-0.2 Mixup would mostly reduce effective regularization rather than add substantial global interpolation. The reviewer required either an honest reduced-regularization framing or a stronger pre-registered alpha.

The refinement adopts `alpha=0.4`, the upper end of the cited common CIFAR range, without changing the 50/25/25 hard/CutMix/Mixup probabilities. The same probe gives 22.53% central draws and mean minor-class mass 0.1604: materially more interpolation than alpha 0.2 while remaining weaker than alpha-1 mixing (40.02%, 0.2501). This is a single pre-run mechanism correction from adversarial review, not a metric-informed tune. The 87.08% switch-accuracy underfit marker remains diagnostic only; it cannot trigger a retry or fallback.

## Idea Evaluation

| Idea | Evidence / reasoning | Potential impact | Decision |
| --- | ---: | ---: | --- |
| Balanced Mixup/CutMix geometry | 7/10 | 8/10 | Advance with pre-registered Mixup alpha raised from 0.2 to 0.4. |
| Isolated PyTorch Nesterov | 6/10 | 4/10 | Retain as a later attribution experiment; it does not directly add invariance and has ambiguous generalization sign. |
| One-epoch-half-life weak-tail EMA | 3/10 | 3/10 | Reject for this loop; the monotonic tail and EXP-018 both predict averaging lag. |

## Chosen Idea
**Selected**: Balanced 50/25/25 hard/CutMix/Mixup geometry with Mixup alpha 0.4

**Why this idea**:
It directly attacks the diagnosed generalization bottleneck using the only intervention family that has recently moved the frontier. Total mixed-target frequency stays at the accepted 50%, so it does not repeat EXP-011's over-regularization by probability, while alpha-0.4 Mixup now supplies enough whole-image interpolation to make the complementarity claim testable. Nesterov is cleaner but lower-ceiling; EMA contradicts the observed monotonic weak tail.

**Hypothesis**:
Replacing half of accepted strong-phase CutMix events with alpha-0.4 Mixup, while preserving 50% hard batches and the complete weak-tail recipe, will retain at least 99% of accepted exposure and healthy strong fit, and raise `best_test_acc` from 94.15% to at least 94.25% by combining regional occlusion with whole-image linearity.

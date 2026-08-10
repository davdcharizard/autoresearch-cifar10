# Brainstorm EXP-023
**Created**: 2026-08-06

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (`knowledge/papers/eca-net.md`; CVPR 2020)
  A tiny channel-axis convolution over global descriptors can recalibrate ResNet channels without an SE bottleneck. Transfer to shallow CutMix training is indirect, so identity initialization, exact-corpus safety, and real H20 timing remain mandatory.
- **Generalizing Pooling Functions in Convolutional Neural Networks** (`knowledge/papers/mixed-pooling.md`; AISTATS 2016)
  Learned or mixed spatial statistics can outperform a universal pooling rule. After EXP-014's raw-max collapse, only smooth, bounded aggregation with explicit first-update scale gates is credible.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`; BMVC 2016)
  Width can be more effective than extreme sequential depth on CIFAR. EXP-007 confirms the direction locally, while the fixed-time cost and current width-2 frontier make a width/depth rebalance a high-risk net hypothesis.
- **CutMix** (`knowledge/papers/cutmix.md`)
  Class-bearing regional evidence is a validated local ingredient. Attention and pooling candidates must account for its area-proportional targets rather than amplifying isolated peaks without bounds.

## Experimental History Review

- The current frontier remains EXP-010 at 94.15%: width-2 postactivation ResNet-20 with N1/M7 and p=0.5 alpha-1 CutMix through 80%, followed by a hard weak tail. It reached 89.73% at the switch, 93.16% at the first weak checkpoint, and finished at its best with 0.1934 NLL and 26,898 updates.
- Validated gains remain the long high-LR plateau (EXP-002), phase-bounded RandAugment (EXP-004), width 2 (EXP-007), and conservative CutMix (EXP-010). Every finalist must preserve these unless architecture is the isolated intervention.
- Repeated failure families now rule out decay changes, identity-oriented residual changes, pool-first transition shortcuts, and unwarmed optimizer-path changes. EXP-020/022 both produced lower-loss class transients under altered optimizer dynamics; retain ordinary standard momentum.
- EXP-014 proved that zero output is not optimization continuity: a raw max classifier had a 4.10x first gradient and collapsed to chance. Smooth GeM is materially different but must gate feature/logit/gradient scale before production.
- EXP-016's width-3 BF16 candidate was unsafe, while its aligned full-width-3 FP32 control remained stable. Width-3 FP32 therefore remains live, but full ResNet-20 cost was steep; a shallower width-3 graph is a distinct capacity-versus-depth bet.
- EXP-017's learned transition shortcut improved strong fit but worsened late NLL/top-1, and EXP-021's deterministic pool-first shortcut failed safety. Leave Option-A transitions untouched.
- EXP-018's uniform SWA worsened the online endpoint; EXP-022's Lookahead failed before production. Final-stage conditional representation and smooth aggregation remain untested, while backward still accounts for 75.46% of step cost and demands exposure gates.

## Collected Ideas

- **Identity-scale final-stage ECA** — gate only the three `layer3` residual outputs with zero-start length-5 channel convolutions and `2*sigmoid`, preserving the exact initial function. It targets example-dependent use of proven width-2 capacity while avoiding transitions and optimizer state; CutMix descriptor ambiguity and sequential backward launches are the main risks.
- **Fixed GeM-3 final pooling** — replace pure global average with a fixed cubic power mean on nonnegative final features. It targets compact class evidence with distributed magnitude-weighted gradients and no new parameters, while explicit scale gates separate it from EXP-014's unbounded raw-max branch.
- **FP32 width-3 ResNet-14** — use two blocks per stage at widths 48/96/192, trading three residual blocks for wider channels. It targets the representation frontier and partially attacks sequential backward depth; timing must show at least 20,000 updates and full-phase fit must justify lost depth.
- **RMS-normalized average-plus-max blend** — add a small fixed max contribution after matching max-feature RMS to the average path on production data. It directly addresses EXP-014's scale mechanism, but any data-derived normalization constant risks benchmark tuning and a fixed max coefficient remains area-insensitive under CutMix.
- **Suppress-only final-stage attention** — use a gate initialized near one whose range is `(0,1]`, preventing amplification. It aligns with late calibration concerns, but exact identity with nonzero gradient is awkward and could recreate the residual suppression family.
- **Final-stage stochastic DropBlock** — schedule low structured masking only late in the strong phase and disable it for refinement. Literature supports spatially correlated masks, but accepted augmentation already creates strong underfit pressure and the timed mask kernel attacks the wrong cost center.
- **FP32 channels-last probe** — convert model and inputs to NHWC physical storage to seek convolution/backward speed on H20. It directly targets the 75.46% backward bottleneck, but prior review found no direct tiny-FP32 speed evidence and extra exposure has no proven accuracy mechanism.
- **Moonshot: width-4 ResNet-8** — collapse to one block per stage while widening to 64/128/256, yielding a similar parameter scale with much less sequential depth. It might improve H20 utilization dramatically, but loses too much hierarchical processing and has no local evidence, so it is not ready for a scored slot.

## Combinations

- **ECA + GeM**: channel recalibration could select semantic features while GeM preserves their compact spatial evidence, potentially improving both channel and spatial allocation. The cross is stronger only if both isolated mechanisms work; combining them now would multiply CutMix ambiguity and destroy attribution.
- **Width/depth rebalance + channels-last**: NHWC speed could fund wider features while removed blocks reduce sequential launches. This might beat either alone as a hardware-aware architecture, but channels-last has not cleared a cheap FP32 timing probe and cannot be assumed as free budget.
- **Final-stage ECA + weak-tail EMA**: attention could improve representation while a short EMA smooths learned gates. EXP-018 and EXP-022 make weight-trajectory intervention the weaker component, so this combination is deferred.

## Candidate Ideas

### Fixed GeM-3 Final Pooling
**Summary**: Replace global average pooling with fixed `p=3`, epsilon `1e-6` generalized-mean pooling over the final nonnegative 8x8 features; retain the same classifier, parameter count, optimizer, RNG, and all training mechanics. Full proposal: `proposals/idea-02.md`.

**What it targets**: Spatial aggregation at the final readout: pure averaging may dilute compact class-bearing responses, while a cubic power mean emphasizes salience without raw max's single-index gradient.

**Reasoning**: Mixed-pooling/GeM literature provides a smooth average-to-max mechanism, and EXP-010 establishes that localized class evidence matters. Unlike EXP-014, GeM adds no independent classifier and distributes gradients across every positive activation; strict pooled-feature, logit, classifier-gradient, and same-batch displacement gates directly test whether its scale remains controlled.

**Sources**: `knowledge/papers/mixed-pooling.md`; `proposals/idea-02.md`; EXP-010 and EXP-014 analysis.

**Estimated Effort**: Medium — one endpoint expression plus scale, safety, numerical, and paired timing tests.

**Risk Assessment**: Retrieval evidence may not transfer to classification; cube/root operations can amplify RandAugment artifacts, conflict with CutMix area labels, alter logit scale, and add sequential backward kernels. Fixed `p=3` has no local tuning evidence.

### Identity-Scale Final-Stage ECA Recalibration
**Summary**: Add one zero-initialized `Conv1d(1,1,5)` gate to each of the three `layer3` residual outputs and multiply by `2*sigmoid(logit)` before the unchanged shortcut addition. Initial gates are exactly one, shared state/RNG/function are aligned, and only 15 parameters are added. Full proposal: `proposals/idea-01.md`.

**What it targets**: Conditional allocation of the accepted width-2 model's high-level channels—the representation/generalization limiter—without changing early CutMix processing, Option-A transitions, optimizer dynamics, or the initial residual strength.

**Reasoning**: ECA-Net supplies direct ResNet attention evidence, and final-stage-only scope addresses the strongest prior critiques of all-block attention: shallow channel semantics, early mixed-descriptor interference, and sequential overhead. Exact unit gates avoid EXP-012/015's initial residual suppression and the `(0,2)` bound avoids EXP-014's unbounded independent logits.

**Sources**: `knowledge/papers/eca-net.md`; `proposals/idea-01.md`; EXP-010, EXP-012, EXP-014, EXP-017, EXP-018, and EXP-022 reports/reviews.

**Estimated Effort**: Medium — modest production diff, substantial identity/first-update/exact-corpus/timing verification.

**Risk Assessment**: Channel adjacency is artificial, CutMix creates semantically mixed global descriptors, identity lasts only until the first update, three tiny attention chains may still lose exposure, and the effect ceiling may be below ten test examples.

### FP32 Width-3 ResNet-14 Depth-Width Rebalance
**Summary**: Change to two blocks per stage and width multiplier three, producing channels 48/96/192, six residual blocks, 13 convolutions, and 1,540,474 parameters while preserving FP32, Option-A, data, optimizer, schedule, seed, and evaluator. Full proposal: `proposals/idea-03.md`.

**What it targets**: A joint representation and systems trade: wider strong-view features with fewer sequential Conv/BN blocks, directly rebalancing the 75.46%-backward-limited model.

**Reasoning**: EXP-007 showed a +1.25-point width gain despite 29.2% fewer updates, Wide Residual Networks supports width/depth tradeoffs on CIFAR, and EXP-016's matched width-3 FP32 control was stable. ResNet-14 width 3 is only 63.85% of full width-3 ResNet-20 parameters, but the full net effect remains untested.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `proposals/idea-03.md`; EXP-007, EXP-008, EXP-010, and EXP-016.

**Estimated Effort**: Low code effort, high verification and run risk due architecture/timing/exposure checks.

**Risk Assessment**: Width-2 gains may not extrapolate; losing three blocks can reduce hierarchy and receptive-field refinement; roughly 1.456x MACs may leave only ~20k updates and ten weak epochs; unchanged LR/decay may be miscalibrated for the new parameterization.

## Review

The external Claude harness returned an empty file, so the required independent idea-critic fallback wrote `01-idea-review.md`. It selected FP32 width-3 ResNet-14 because width is the only current in-goal lever with a demonstrated effect well above the ten-image threshold (EXP-007 +1.25 points), and EXP-016 localized its width-3 safety failure to BF16 rather than the aligned FP32 control.

I accept the two significant cautions. First, the candidate is a net fixed-time architecture test, not clean evidence for width alone: removing one block per stage is an untested depth reduction. The plan must always capture switch accuracy, first-weak accuracy, NLL, and realized exposure so a miss near the 20,000-step floor is diagnosed as depth loss/underoptimization rather than overclaimed as capacity saturation. Second, the paired timing gate is load-bearing because the 1.456x MAC estimate sits near the allowed 1.345x mean ratio; timing no-go is an informative feasibility result and must not trigger LR, decay, precision, or architecture rescue.

The review's GeM concern is accepted: fixed p=3 has indirect retrieval evidence and structurally emphasizes salience while CutMix trains area-proportional targets. ECA remains the best-engineered fallback but has both a tight sequential-kernel timing gate and a plausible effect below the 0.10-point acceptance margin.

## Idea Evaluation

| Candidate | Evidence / reasoning | Potential impact | Decision |
|---|---:|---:|---|
| FP32 width-3 ResNet-14 | 7/10 | 8/10 | Select; highest in-goal evidence and ceiling, subject to strict timing/exposure diagnostics. |
| Identity-scale final-stage ECA | 7/10 | 5/10 | Defer; rigorous and distinct, but likely small effect and live sequential-kernel veto. |
| Fixed GeM-3 | 4/10 | 4/10 | Reject for this slot; fixed-statistic evidence is weak and CutMix semantics conflict. |

## Chosen Idea
**Selected**: FP32 Width-3 ResNet-14 Depth-Width Rebalance

**Why this idea**:
Width produced the largest local improvement in the entire history, and a shallower width-3 FP32 model tests whether channel capacity can advance again without paying full width-3 ResNet-20 cost. The method preserves ordinary momentum, accepted transitions, augmentation, schedule, and evaluation, escapes every recurring failure family, and has a credible effect ceiling above single-seed resolution. Its depth confound and exposure risk are explicit parts of the net fixed-time hypothesis, not hidden claims of equal-compute superiority.

**Hypothesis**:
Changing to two residual blocks per stage at widths 48/96/192 will retain at least 20,000 optimizer steps and enough weak-tail refinement under the fixed 300-second budget for the wider representation to outweigh lost depth, raising `best_test_acc` from 94.15% to at least 94.25%. A safety/timing no-go retires this exact architecture before production; a valid miss must be interpreted using switch fit, first-weak accuracy, NLL, and actual exposure rather than as broad evidence against width.

# Brainstorm EXP-026
**Created**: 2026-08-06

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`; ICLR 2018)
  Whole-image and target interpolation improves CIFAR generalization by encouraging local linearity. EXP-019's balanced alpha-0.4 policy remains scientifically untested because its safety data were not replayable, not because the method produced a trustworthy failure.
- **CutMix** (`knowledge/papers/cutmix.md`)
  Regional class-bearing mixing is the current local frontier mechanism. A Mixup combination should replace—not add to—some accepted CutMix events so the total 50% mixed-target probability remains fixed.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`; BMVC 2016)
  WRN recipes use dropout within residual blocks as regularization alongside width. Transfer to this short fixed-time postactivation/CutMix recipe is uncertain because strong-phase underfit is already a recurring local failure.
- **Large-minibatch SGD / Ghost-BN motivation** (`knowledge/papers/large-minibatch-sgd.md`)
  Batch statistics and gradient-batch scale need not be identical. Virtual BN groups can restore normalization noise while preserving optimizer batch 128, but the local model is not known to suffer large-batch BN generalization.

## Experimental History Review

- EXP-010 remains the 94.15% frontier: width-2 ResNet-20, standard momentum and all-parameter decay, N1/M7 with 50% alpha-1 CutMix through 80%, then a hard weak tail. Preserve its 89.73% switch and 93.16% first-weak trajectory as diagnostics.
- EXP-019 implemented balanced hard/CutMix/Mixup and passed semantics, 20,000-collation proportions, worker lifecycle, finite state, and lower terminal loss. Its first safety attempt produced a real but unattributable concentration veto while a second passed on different source augmentations; the report calls the method unproven and prescribes exact post-transform persistence for reconsideration.
- EXP-024 established a three-bucket immutable corpus pattern. EXP-025 further showed exact initial identity and one safe update do not guarantee multi-step recruitment; all new trainable branches require output-trajectory bounds.
- Static architecture routes have narrowed: global width with lost depth scored 94.00, asymmetric late width hit a class transient, and ECA gates saturated. Transition, decay, optimizer-path, identity-suppression, and uniform tail averaging families have repeated failures.
- Systems headroom is backward-bound (75.46%), with abundant memory and hidden loader cost. Host-side mixing is likely exposure-neutral; BN reshaping or dropout must prove real H20 cost and healthy strong fit.

## Collected Ideas

- **Exact-corpus balanced Mixup/CutMix retry** — keep 50% hard batches but split the mixed half into 25% CutMix alpha1 and 25% Mixup alpha0.4. Persist pre-policy N1/M7 tensors and policy draws so accepted and candidate arms see identical source images; this directly repairs EXP-019's only decisive flaw.
- **Ghost BN groups of 64** — keep optimizer batch128 but compute training BN statistics independently on two virtual groups. This injects normalization noise without losing image throughput, though every BN call changes and CutMix pairs may land in different statistic groups.
- **Strong-phase residual dropout** — apply small dropout between the two convolutions of same-width residual blocks only while strong augmentation is active, then disable at the weak tail. It imports WRN regularization but risks compounding the accepted model's limited strong fit.
- **Mixup-only quarter probability** — replace only 25% of accepted CutMix events with hard batches plus 25% Mixup, reducing total mixed frequency to 25%. This could recover fit but conflates geometry with weaker regularization and retreats from EXP-010's validated 50% mix.
- **Bounded endpoint GeM blend** — add a very small norm-matched smooth GeM contribution to GAP. Analytic output bounds avoid EXP-014 raw-max collapse, but EXP-025 warns that learned scalars can saturate and literature transfer is indirect.
- **Channels-last accepted graph** — seek convolution-backward speed while preserving model mathematics. It attacks the measured systems bottleneck, but EXP-025 review found the exposure-to-accuracy link unproven and tiny FP32 NHWC speed doubtful.
- **Batch-96 accepted recipe** — trade image efficiency for more optimizer updates and noise. This is orthogonal but likely sees fewer images, retains an arguably high LR, and creates evaluation-count bias unless tightly controlled.
- **Moonshot: manifold Mixup at layer3 input** — mix feature maps and labels after layer2 on a quarter of strong batches. It could impose linearity directly in semantic space without global pixel ghosts, but changes the forward graph/data coupling and has weak local safety evidence.

## Combinations

- **Balanced Mixup/CutMix + Ghost BN**: complementary input geometry and normalization noise might generalize better than either alone, but both change soft-batch optimization and destroy attribution; isolate first.
- **Residual dropout + Mixup**: WRN-style feature stochasticity plus input interpolation could regularize width-2 capacity broadly, but accepted strong fit has little headroom for compounded noise.
- **Channels-last + balanced Mixup/CutMix**: a layout speedup could fund mixing overhead, but worker-side mixing is already effectively hidden and the layout change adds no demonstrated accuracy mechanism.

## Candidate Ideas

### Exact-Corpus Balanced Mixup/CutMix Retry
**Summary**: Re-run EXP-019's unchanged strong policy—50% hard, 25% alpha-1 CutMix, 25% alpha-0.4 Mixup—but repair its evidence protocol. Persist 200 natural post-N1/M7, pre-policy source batches and each worker RNG state, then apply accepted and candidate policies independently to cloned identical sources. Full proposal: `proposals/idea-01.md`.

**What it targets**: Generalization geometry without increasing the accepted 50% soft-target rate: regional localization plus whole-image linearity.

**Reasoning**: CutMix is the last frontier gain; Mixup has direct CIFAR evidence and materially different invariance. EXP-019 passed implementation/lifecycle checks and remained finite with lower loss, but its invalid result was solely non-replayable evidence. This new protocol is exactly the report's prescribed condition for reconsideration.

**Sources**: `knowledge/papers/mixup.md`, `knowledge/papers/cutmix.md`; EXP-010/011/019/024; `proposals/idea-01.md`.

**Estimated Effort**: high — small production diff, demanding pre-policy corpus/RNG/provenance controller.

**Risk Assessment**: Mixup may replace useful localization and deepen underfit; alpha/split are compound. Capturing the true post-transform worker state is implementation-sensitive. Natural geometry floors, source hashes, semantic equality, exact-corpus safety, real-loader timing, and no rematerialized rescue are mandatory.

### Ghost BatchNorm with Virtual Groups of 64
**Summary**: Keep optimizer batch128 but normalize two contiguous 64-example training groups independently in all 19 BN sites, sharing affine parameters. Update one evaluation running-stat buffer from the full logical batch once per step; evaluation uses ordinary BN. Parameter count stays 1,073,962. Full proposal: `proposals/idea-02.md`.

**What it targets**: Normalization-statistic noise and generalization without changing gradient batch or image throughput.

**Reasoning**: Ghost BN has large-batch generalization evidence and avoids new residual/optimizer/target paths. Yet batch128 with large spatial maps may already have adequate BN noise, and all 19 sites change from step one. The custom dual-normalization plus full-batch buffer policy is a substantial semantic implementation.

**Sources**: Ghost BN/GhostNorm/BN papers cited in `proposals/idea-02.md`; EXP-007/010/024/025; system understanding.

**Estimated Effort**: high — custom BN semantics, reference math, state validation, and strict timing.

**Risk Assessment**: Train/eval statistic mismatch may harm the abrupt weak-tail conversion; extra BN/reduction launches may exceed a 3% exposure budget; CutMix donor/group interactions and broad strong underfit are plausible.

### Strong-Phase Residual Activation Dropout
**Summary**: Apply elementwise inverted dropout `p=0.05` after the first Conv-BN-ReLU in exactly seven same-shape residual blocks during the strong phase, excluding both transitions, then disable it at the existing 80% switch. Full proposal: `proposals/idea-03.md`.

**What it targets**: Width-2 feature co-adaptation during composite-view training while preserving shortcuts and giving the full deterministic network the hard weak tail.

**Reasoning**: WRN motivates this within-block placement, and phase-limiting it is locally coherent. But WRN's CIFAR-10 dropout results were mixed, accepted strong fit has little margin, and local identity-oriented regularizers have repeatedly suppressed the short strong phase.

**Sources**: Wide Residual Networks and stochastic-depth sources cited in `proposals/idea-03.md`; EXP-007/010/011/012/015.

**Estimated Effort**: medium — small diff with exact RNG/mask/lifecycle and timing controls.

**Risk Assessment**: Compounded RandAugment/CutMix/dropout underfit is dominant; CUDA RNG intentionally diverges; noisy BN activations may not settle in the short tail; a ten-example gain has weak causal resolution.

## Review

The fallback independent critic selected the exact-corpus balanced Mixup/CutMix retry (evidence/reasoning 7.5/10, impact 7.5/10), ahead of Ghost BN (5.5/10, 6/10) and residual dropout (4/10, 5/10). It judged the retry uniquely valuable because it closes EXP-019's explicit open avenue while preserving architecture, optimizer, total mixed frequency, and likely exposure.

I adopted two refinements. First, EXP-019's initial concentration event remains adverse evidence; non-replayability made it unattributable, not imaginary. EXP-026 adjudicates that signal on one immutable natural corpus rather than erasing it. Second, per-record source hashes, a no-policy-draw source collator, replayed categorical `u`, and byte-identical shared hard/CutMix branches are launch-critical evidence—not optional provenance metadata. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt **Exact-Corpus Balanced Mixup/CutMix Retry**. It has the clearest direct generalization mechanism and is explicitly authorized by the prior invalid report's reconsideration condition. Ghost BN's 19 custom normalization sites mismatch the batch-128 regime and threaten timing; dropout points toward recurring strong underfit and has mixed direct CIFAR evidence.

## Chosen Idea
**Selected**: Exact-Corpus Balanced 50/25/25 Hard/CutMix/Mixup Retry

**Why this idea**:
It tests whether whole-image linearity complements the only recent frontier mechanism while holding total soft-target probability at the accepted 50%. Unlike a repeated failed approach, EXP-019 never produced attributable safety evidence or a scored result and explicitly prescribed immutable post-transform replay before reconsideration. The new pre-policy corpus and worker-state protocol makes control and candidate differences causal: identical sources and total-mix decisions, bitwise-equal hard/CutMix shared branches, and differences only where accepted CutMix is replaced by alpha-0.4 Mixup.

**Hypothesis**:
Replacing half of accepted strong-phase CutMix events with alpha-0.4 Mixup, while preserving 50% hard batches and the complete weak tail, will remain safe on one immutable natural corpus, retain at least 99% of accepted exposure, preserve healthy strong fit, and raise `best_test_acc` from 94.15% to at least 94.25% through complementary regional and whole-image interpolation.

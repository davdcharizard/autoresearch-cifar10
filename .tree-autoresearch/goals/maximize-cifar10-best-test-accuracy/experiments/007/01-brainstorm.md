# Brainstorm EXP-007
**Created**: 2026-08-05

## Web Search & Literature Review

- **How to Scale Your EMA** (`experiments/007/papers/how-to-scale-your-ema.md`)
  EMA behavior depends on update cadence and should be specified through an effective averaging horizon, which matters when total steps are wall-clock-dependent.
- **A Modern Look at the Relationship between Sharpness and Generalization** (`experiments/007/papers/modern-sharpness-generalization.md`)
  Adaptive sharpness is not a universal predictor of generalization and often correlates with training choices such as learning rate, tempering an automatic ASAM preference.
- **Averaging Weights Leads to Wider Optima and Better Generalization** (`knowledge/papers/stochastic-weight-averaging.md`)
  Late SGD-trajectory averaging improves CIFAR residual networks at low arithmetic cost, but trajectory diversity and BatchNorm state are essential to faithful transfer.
- **ASAM: Adaptive Sharpness-Aware Minimization** (`knowledge/papers/adaptive-sharpness-aware-minimization.md`)
  Scale-aware perturbations improve full-run SAM on CIFAR in the published regime, but the effect must be discounted when transferred to sparse late pulses.
- **Residual and phase-dependent regularization** (`knowledge/papers/shakedrop.md`, `knowledge/papers/time-matters-regularization.md`)
  Richer residual perturbations and time-localized regularization can improve CIFAR ResNets, while the current lineage already shows that mechanism choice matters more than scalar strength.

## Experimental History Review

- The accepted lineage is BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23% -> EXP-004 95.40%. Architecture/throughput, front-loaded CutMix, and clean-tail periodic SAM are the three validated mechanisms.
- EXP-003 showed that selected scalar CutMix/drop-path gains of 0.14-0.29 points can disappear on confirmation. Candidate mechanisms now need a plausible effect near 0.3 points.
- EXP-005 preserved optimizer steps but halved new-identity exposure through forced half-overlap, lowered final loss, and lost 0.12 points. EXP-007 must retain the parent's independent-image stream.
- EXP-006 replaced one quarter of selected CutMix with manifold mixup. Accuracy remained within noise and final loss worsened, so substitution from a validated mechanism is low-information and should not be repeated.
- EXP-004's 2.75M-parameter model uses only 1,190.5 MiB of a 97,871 MiB H20 and completes 25,560 steps. Memory permits shadow weights or modest representation modules; extra full model passes directly consume the fixed step budget (`02-system-understanding.md`).
- The limiting gap is detectable generalization from an already strong, well-fitted trajectory. Untried spaces include additive weight averaging, scale-aware refinement of validated SAM, lightweight channel recalibration, BatchNorm noise control, anti-aliasing, and deeper architectural reallocation.

## Collected Ideas

- **Horizon-matched full-state EMA** - Maintain a functional EMA copy on top of every parent optimizer update, with a preregistered sample/epoch horizon rather than a familiar but arbitrary decay. It targets late-iterate variance without removing CutMix, SAM, images, or forward passes; abundant VRAM makes a full shadow state cheap.
- **Sparse late ASAM geometry** - Replace only EXP-004's rho-0.05 Euclidean SAM perturbation with literature-derived parameter-scale-aware perturbations at the same late period-two cadence. It directly refines a validated mechanism with matched full-run CIFAR evidence, but modern sharpness results warn that geometric plausibility alone is insufficient.
- **Identity-initialized channel recalibration** - Add lightweight SE/ECA-style channel gates only in the 128- and 256-channel stages, initialized to preserve or closely match the parent's residual scale. It uses memory headroom to improve representation selectivity while retaining every training mechanism and image.
- **Ghost BatchNorm at fixed physical batch** - Compute training BatchNorm statistics over fixed virtual groups inside each 256-image batch while retaining one optimizer step and the same data stream. This attacks large-batch normalization/generalization without extra model passes, but changes all residual activations and may conflict with CutMix.
- **Anti-aliased stage transitions** - Move stride from the learned transition convolution into a fixed low-pass downsample or blur-pool path. This targets spatial aliasing and shift sensitivity rather than regularization, but adds memory traffic and risks reducing the time-budgeted step horizon.
- **Deep-supervision auxiliary head** - Add a small intermediate classifier after stage two during training only, with its loss removed in the clean tail or inference. The extra gradient could improve early feature separability at low parameter cost, though weighting creates a scalar choice and the auxiliary computation may have a compressed effect.
- **Clean-only mild label smoothing** - Apply fixed epsilon-0.05 smoothing only to early clean batches, retaining CutMix's area-weighted labels and hard clean tail. It is additive and nearly free, but EXP-004's existing CutMix/drop-path/SAM stack makes the expected effect likely smaller than the measured noise floor.
- **Lookahead slow online weights** - Periodically interpolate live parameters toward a slow copy while leaving evaluation on the live model. This attacks trajectory oscillation with one model, but it changes the optimizer path and momentum semantics rather than supplying a clean inference-time average.
- **Compact deeper residual moonshot** - Reallocate the 2.75M parameters into more, narrower preactivation blocks with matched estimated MACs, using the large H20 headroom and residual-depth evidence. It has first-order upside but abandons the validated architecture and may lose throughput on small kernels.

## Combinations

- **EMA + channel recalibration**: channel gates could add representational capacity while EMA stabilizes their late trajectory. The combination attacks bias and variance together, but bundling two unvalidated changes would obscure attribution and should follow evidence for one component.
- **ASAM + EMA**: scale-aware SAM could produce a flatter set of late iterates for EMA to average, potentially exceeding either alone. Both operate in weight space, however, and their interaction is too confounded for a first test.
- **Channel recalibration + anti-aliasing**: selective channels and cleaner downsampling target complementary feature-quality errors. The combined architecture has a larger plausible ceiling but creates enough throughput and initialization risk that each should first earn independent evidence.

## Candidate Ideas

### Literature-Scale ASAM in the Validated Clean Tail
**Summary**: Replace only EXP-004's Euclidean rho-0.05 SAM perturbation package with element-wise p=2 ASAM at the same period-two clean-tail cadence. Use the fixed CIFAR literature package `rho=0.5`, `eta=0.01`: scale non-bias parameters by `abs(w)+eta`, compute one global FP32 scaled-gradient norm, and apply the required second scale multiplication. Preserve CutMix, image stream, two-pass RNG/BatchNorm safeguards, and the sole Nesterov update. Full design: `proposals/idea-02.md`.

**What it targets**: EXP-004 validated late sharpness-aware optimization but left perturbation geometry unexplored. ASAM targets parameter-scale sensitivity across convolution, normalization, and classifier weights without reducing any data/augmentation dose (`02-system-understanding.md`).

**Reasoning**: Published ASAM-over-SAM CIFAR gains are 0.20-0.46 points across related residual models, including 0.30 on WRN-28-10. The parent already shows this mechanism class works, making geometry refinement a focused bet with an effect ceiling near the measured noise boundary.

**Sources**: `knowledge/papers/adaptive-sharpness-aware-minimization.md`; `experiments/007/papers/modern-sharpness-generalization.md`; EXP-004 analysis; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Full-training literature gains may vanish under sparse late pulses; rho 0.5 is a package change, not an isolated scale ablation; extra tensor kernels reduce exposure; and modern evidence warns adaptive sharpness is not universally causal for generalization.

### Identity-Centered Efficient Channel Recalibration
**Summary**: Add one fixed three-tap ECA gate to each of the six residual branches after `conv2` and before drop path/addition. Represent each kernel as a standalone zero-initialized three-element parameter and gate with `2*sigmoid(logits)`, making the candidate exactly parent-equivalent at initialization without consuming initialization RNG. All 18 new parameters participate in ordinary SGD and SAM. Full design: `proposals/idea-03.md`.

**What it targets**: The 2.75M-parameter WRN uses only 1.2% of GPU memory, and prior recipe changes have plateaued. Channel recalibration adds representation selectivity without sacrificing independent images, CutMix, or SAM (`02-system-understanding.md`).

**Reasoning**: ECA improves residual backbones with negligible parameters/MACs, and the identity-centered adaptation removes standard sigmoid gating's initial 0.5 residual shrinkage. Estimated arithmetic is below 0.12% of parent MACs, though a GPU latency gate protects against small-kernel launch overhead.

**Sources**: `experiments/005/papers/eca-net.md`; ECA-Net, CVPR 2020; EXP-004/005/006 analyses; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Evidence is mainly deeper ImageNet models, `2*sigmoid` departs from standard ECA and can amplify noisy channels, zero initialization may learn slowly, and six tiny operator chains may cost more latency than their FLOP count suggests.

### Time-Constant Late Full-State EMA
**Summary**: Add a no-gradient EMA copy of the complete inference state on top of EXP-004, starting at progress 0.75 after optimizer/SAM restoration. Sample every 32 steps but derive each decay from elapsed charged time with a fixed 18.75-second time constant, so wall-clock throughput changes do not redefine the averaging horizon. Average parameters and floating BatchNorm buffers, copy integer buffers, and use the existing single evaluator call on the EMA model after activation. Full design: `proposals/idea-01.md`.

**What it targets**: The measured limiter is detectable generalization amid 0.14-0.29-point run/selection variation and a 0.15-point late-checkpoint span, while 98.8% of GPU memory is unused (`02-system-understanding.md`). EMA directly reduces iterate variance without consuming CutMix, images, SAM pulses, or model passes.

**Reasoning**: SWA improves CIFAR residual networks, and EMA scaling evidence supports an elapsed-time horizon rather than an arbitrary decay. The shadow adds about 11-15 MiB and sparse foreach traffic. Unlike prior failed children, it is additive and leaves the full accepted recipe in place.

**Sources**: `knowledge/papers/stochastic-weight-averaging.md`; `experiments/007/papers/how-to-scale-your-ema.md`; EXP-004 and EXP-006 analyses; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: EXP-004's final accuracy already equaled its best, SAM may make averaging redundant, the low-LR tail may lack SWA-like trajectory diversity, and averaged BatchNorm buffers are only an approximation to statistics under averaged weights.

## Review

Claude selected literature-scale ASAM because it changes the exact optimizer mechanism EXP-004 validated while retaining every data, augmentation, and evaluation dose. The review's strongest concern is duty-cycle transfer: published ASAM gains are full-run, while this experiment uses about one eighth of all steps. The design therefore treats +0.30 as an exploratory evidentiary target, instruments both adaptive-coordinate and Euclidean perturbation radii, and must preserve near-parent step exposure. EMA was downgraded because EXP-004 final equaled best and because averaged BatchNorm/evaluation ownership creates regression risk; ECA was downgraded for weak matched-regime evidence and likely launch overhead. The critic's ECA initialization-RNG concern was not adopted because `proposals/idea-03.md` already specifies standalone zero parameters that consume no initializer draws. Full review: `01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. ASAM scored 7/10 for evidence/reasoning and 6/10 for impact, ahead of EMA at 5/10 and ECA at 3/10/4/10. The selection is not based on ASAM being safer: it is based on the closest matched CIFAR evidence and the cleanest single-component comparison against a successful parent mechanism. A result from 95.50% to 95.69% can satisfy the frozen tree gate but will be reported as below the preregistered mechanism-sized effect.

## Chosen Idea
**Selected**: Literature-Scale ASAM in the Validated Clean Tail

**Why this idea**:
EXP-004 already proves that spending a second pass on late sharpness-aware optimization helps this exact lineage. ASAM tests whether parameter-scale-aware perturbations use that paid pass more effectively, without reducing CutMix, independent-image exposure, or evaluation cadence. The fixed literature package `rho=0.5, eta=0.01` avoids metric-driven scalar search, while exact adaptive-radius, restoration, RNG, BatchNorm, latency, and exposure checks contain the larger correctness surface.

**Hypothesis**:
Replacing EXP-004's late period-two SAM package with literature-scale p=2 ASAM will retain at least 25,000 optimizer steps and improve `best_test_acc` from 95.40% to at least 95.70% in one fixed-seed physical-GPU-0 run. The formal necessary threshold remains 95.50%, but a result below 95.70% falsifies the preregistered mechanism-sized effect expectation. Any geometry, runtime, or protocol failure is not repaired by changing rho, eta, cadence, parameter exclusions, or seed.

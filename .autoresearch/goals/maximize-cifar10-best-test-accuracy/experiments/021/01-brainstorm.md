# Brainstorm EXP-021
**Created**: 2026-08-06

## Web Search & Literature Review

- **Channels-last memory format for convolutional networks** (`papers/channels-last-memory-format.md`; official PyTorch and NVIDIA documentation)
  PyTorch represents NHWC physical storage while preserving logical NCHW dimensions and recommends converting both model and inputs. NVIDIA recommends NHWC for Tensor Core convolution because NCHW can require transposes. The transfer is uncertain here: the documented large gains emphasize AMP/reduced precision, while this candidate must remain FP32 after EXP-016 rejected the tested BF16 path.
- **CutMix** (`knowledge/papers/cutmix.md`; ICCV 2019)
  CutMix preserves class-bearing pixels while mixing targets according to pasted area. EXP-010 validated p=0.5, alpha=1 on this exact recipe; alpha controls event geometry/ambiguity independently of the already-failed probability increase.
- **Pool-first transition shortcuts** (`knowledge/papers/resnet-d-downsampling.md`; CVPR 2019)
  ResNet-D supports pooling before shortcut projection to avoid lossy strided sampling. Its evidence concerns ImageNet learned projections, so it motivates but does not validate a deterministic zero-padded Option-A variant on CIFAR-10.
- **Existing goal knowledge** (`knowledge/README.md`)
  Width, augmentation, data mixing, residual initialization, transition projection, attention, optimizer, averaging, and batch scaling evidence constrain this loop. New candidates must preserve the 94.15% EXP-010 recipe unless their mechanism explicitly replaces one element.

## Experimental History Review

- EXP-010 remains the frontier: width-2 postactivation ResNet-20, all-parameter decay `1e-4`, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, then hard weak cosine refinement. It reached 94.15%, 26,898 steps, 89.73% at the switch, 93.16% at the first weak evaluation, 0.1934 final NLL, and its best at the terminal state.
- The measured systems bottleneck is model backward (75.46% of GPU-stage time), with forward adding 22.11%. Input, loss, optimizer, and launch gaps are too small for isolated micro-optimizations to plausibly clear the accuracy threshold through exposure.
- Stronger CutMix probability, alternate decay, Cutout, label smoothing, early weak switching, identity-oriented residual initialization/ordering, learned projection-BN shortcuts, raw max readout, late SWA, and no-warmup Nesterov did not improve. EXP-012's 94.22% is close but belongs to the recurring strong-phase underfit family.
- EXP-013 showed that systems ideas need fresh paired timing and a material exposure threshold. EXP-019/020 showed that numerical gates must persist exact post-transform tensors; forkserver seed replay alone is not evidence of paired behavior.
- Faster training can complete extra weak-tail epochs and thereby create extra `best_test_acc` looks. Any exposure candidate must cap production at EXP-010's 19 unique evaluations and reserve a terminal look.

## Collected Ideas

- **FP32 full-model channels-last** - Convert the accepted model, training inputs, and evaluator-boundary inputs to channels-last physical storage. This attacks the 97.57% convolutional forward/backward share and may increase useful exposure, but official speed evidence is strongest for AMP and hidden layout repairs can erase the gain.
- **cuDNN autotune on fixed shapes** - Enable `torch.backends.cudnn.benchmark=True` so cuDNN can select faster convolution algorithms for the few stable shapes. Search/startup behavior is partly inside counted first steps, and modern heuristics may already choose near-optimal kernels; it has a lower and noisier ceiling than an end-to-end layout change.
- **Conv-weight-only channels-last** - Convert only Conv2d weights using PyTorch's selective utility and let convolutions choose preferred activation layout. This may avoid touching unrelated tensors, but input activations begin contiguous and repeated boundary conversions are more likely; full propagation is the cleaner primary test.
- **CutMix alpha 0.5 at fixed p=0.5** - Keep the successful number of mixed batches but use a U-shaped Beta distribution, reducing expected two-class ambiguity while retaining regional replacement. It could recover strong fit, but its ceiling is small around a validated alpha-1 point and endpoint-heavy rectangles can become near-no-op or near-full replacement.
- **Deterministic pool-first Option-A shortcuts** - Replace only the two `::2` shortcut samples with fixed 2x2 average pooling, then preserve zero channel padding. It isolates anti-aliasing from EXP-017's learned projection/BN, but box filtering may erase tiny CIFAR details and adds two backward kernels.
- **Final-stage identity-scale ECA** - Add zero-start `2*sigmoid` channel gates only in layer3. It offers a tiny representation lever, but CutMix makes global channel descriptors semantically mixed and identity-start additive mechanisms have required unusually strict first-update gates.
- **Partial same-width preactivation** - Apply preactivation only to ordinary identity blocks while leaving entries/transitions postactivated. EXP-012 ended just 0.03 below the gate, but both preactivation and zero-gamma suppressed short strong fit; partializing the same family lacks evidence that the removed blocks caused the miss.
- **Low-rate strong-phase stochastic depth** - Skip a small fraction of same-width residual branches during strong training and restore the full graph in the weak tail/evaluation. It could trade dominant backward work for regularization, but shallow residual underfit and BN/state discontinuity make the compute dividend fragile.
- **Hard-tail CutMix-off lead-in** - Disable CutMix before RandAugment so the classifier adapts to hard strong labels before the weak tail. This targets the same recovery mechanism as EXP-011's first weak jump, but adds a lifecycle boundary and reduces the region-mixing dose without direct evidence about timing.

## Combinations

- **Channels-last plus cuDNN benchmark** could jointly expose faster NHWC algorithms, but the combination loses attribution and benchmark search can mask whether layout alone helps. Test neither as a fallback inside this loop.
- **Pool-first Option-A plus channels-last** might offset pooling cost through layout speed, but it combines a representation change with an exposure change. Each deserves an isolated verdict first.
- **Alpha 0.5 plus an earlier CutMix-off boundary** could reduce both per-event and total mixing strength. The two knobs point in the same direction and would make any accuracy outcome uninterpretable.

## Candidate Ideas

### FP32 Full-Model Channels-Last

**Summary**: Store every Conv2d weight and 4D activation in channels-last format, transfer training inputs directly into that format inside counted work, and convert fixed-evaluator inputs at the model boundary. Keep FP32 and all accepted graph, optimizer, data, schedule, timer, and seed semantics. Add a 19-evaluation parity guard that is behaviorally unchanged for EXP-010 and reserves the final look if speed adds an epoch. See `proposals/idea-channels-last.md`.

**What it targets**: The measured 97.57% forward/backward convolutional share. A required at least 3.09% synchronized weighted speedup projects at least 27,705 updates versus 26,898, roughly 103,000 more presented images under the fixed budget.

**Reasoning**: Official PyTorch and NVIDIA guidance supports NHWC propagation and faster Tensor Core convolution. This model's later channels are aligned, cuDNN TF32 is already enabled, and peak VRAM is negligible. The proposal nevertheless treats FP32 transfer as unproven, profiles every intermediate for layout repair, includes input restriding in counted time, and vetoes production below a material paired speed threshold.

**Sources**: `papers/channels-last-memory-format.md`; `proposals/idea-channels-last.md`; `02-system-understanding.md`; EXP-010, EXP-013, and EXP-016 reports.

**Estimated Effort**: medium-high

**Risk Assessment**: Tiny FP32 CIFAR kernels may prefer NCHW; input restriding, Option-A slice/pad, or pooling can force hidden copies; different cuDNN reductions change the numerical path; and extra exposure has no direct local causal validation. The evaluation cap is a second production edit required for metric parity, not an accuracy mechanism.

### Deterministic Pool-First Option-A Shortcuts

**Summary**: At only the two stride-2/channel-double transition shortcuts, replace raw even-phase `::2` sampling with nonoverlapping 2x2 average pooling, then retain exact zero channel padding. Add no learned projection, BN, parameter, RNG draw, or other graph change. See `proposals/idea-pool-first-option-a.md`.

**What it targets**: Transition information and phase sensitivity under crop/flip/RandAugment. All four positions contribute to the shortcut instead of only the top-left phase, while original channel provenance and all same-shape identities remain fixed.

**Reasoning**: EXP-017's pooled learned projection improved switch and first-weak fit but worsened late NLL. This candidate removes the random projection basis and shortcut BN, cleanly testing whether fixed anti-aliased downsampling was the helpful component. ResNet-D gives directional information-preservation evidence, but exact CIFAR Option-A transfer remains speculative.

**Sources**: `proposals/idea-pool-first-option-a.md`; `knowledge/papers/resnet-d-downsampling.md`; EXP-017 report.

**Estimated Effort**: medium

**Risk Assessment**: Box filtering can dilute tiny edges and CutMix boundaries, direct identity gradients fall from one selected pixel to 0.25 across four, residual and shortcut spectra can mismatch, and the expected gain is near single-seed resolution.

### CutMix Alpha 0.5 at Fixed Probability

**Summary**: Change only `CUTMIX_ALPHA` from 1.0 to 0.5 while preserving the validated p=0.5 event rate and the complete accepted training recipe. Prove the adjusted discrete rectangle/target distribution actually lowers ambiguity before production. See `proposals/idea-cutmix-alpha.md`.

**What it targets**: The accuracy/generalization balance within the successful regional-mixing mechanism. A U-shaped Beta distribution retains mean lambda 0.5 but lowers continuous expected `2 lambda (1-lambda)` by 25%, potentially recovering strong-phase fit without reducing mixed-batch frequency.

**Reasoning**: EXP-010 gained 0.60 points but ended its strong phase 0.35 below the non-CutMix width-2 run; EXP-011 proved higher event probability overregularizes. Alpha changes geometry/ambiguity rather than event count and is maximally attributable as one literal. Torchvision clipping and 32x32 quantization make a large empirical distribution gate necessary.

**Sources**: `proposals/idea-cutmix-alpha.md`; `knowledge/papers/cutmix.md`; EXP-010 and EXP-011 reports.

**Estimated Effort**: medium

**Risk Assessment**: Alpha 1 may already be well balanced; more endpoint events can weaken moderate-size occlusion/localization, same-class mixing reduces effective intervention, and a scalar refinement around the frontier has limited upside.

## Review

Mandatory external Claude review completed successfully with exit code 0 and is preserved verbatim in `01-idea-review.md`; no fallback reviewer was used. Claude selected deterministic pool-first Option-A (evidence/reasoning 4/5, potential impact 3.5/5) over CutMix alpha 0.5 (3/5, 2/5) and FP32 channels-last (2.5/5, 2/5).

The decisive distinction is alignment with the diagnosed limiter. Channels-last depends on an unproven FP32 speedup and then on unproven exposure-to-accuracy transfer; alpha 0.5 deliberately removes moderate-lambda events that may be the source of CutMix's benefit. Pool-first directly isolates deterministic downsampling from EXP-017's learned projection and shortcut BN.

The review's requested discriminator is pre-registered: compare EXP-021 against EXP-017's 90.20% switch, 93.45% first-weak, 0.2024 final NLL, and 94.09% best. If pool-only repeats the early-fit gains but again raises NLL and misses the frontier, pooling rather than the learned normalized projection is implicated in the late-generalization harm. This diagnostic never overrides the primary 94.25% acceptance threshold.

## Idea Evaluation

| Idea | Evidence / reasoning | Potential impact | Decision |
| --- | ---: | ---: | --- |
| Deterministic pool-first Option-A | 4/5 | 3.5/5 | Advance; clean isolation of transition downsampling and directly targets representation quality. |
| CutMix alpha 0.5 | 3/5 | 2/5 | Reject; endpoint-heavy geometry has unsupported direction and limited ceiling. |
| FP32 channels-last | 2.5/5 | 2/5 | Reject; likely timing veto and exposure is not the diagnosed accuracy limiter. |

## Chosen Idea

**Selected**: Deterministic Pool-First Option-A Shortcuts

**Why this idea**:
It is the only finalist that changes representation quality at a specifically unresolved site while preserving parameterization, channel provenance, accepted residual activity, optimizer/data semantics, and nearly all exposure. It cleanly separates fixed average downsampling from EXP-017's random projection basis and shortcut BN, making either outcome informative.

**Hypothesis**:
Replacing the two transition shortcut `::2` samples with deterministic nonoverlapping 2x2 average pooling before the existing zero channel padding will retain at least 98% of EXP-010's optimizer exposure and raise `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of 94.30%. A repeated EXP-017 early-fit/NLL signature would instead attribute the prior late-generalization harm to pooling.

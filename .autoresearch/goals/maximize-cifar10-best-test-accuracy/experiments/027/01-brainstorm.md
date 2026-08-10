# Brainstorm EXP-027
**Created**: 2026-08-06

## Web Search & Literature Review

- **Deep Residual Learning for Image Recognition** (`experiments/027/papers/deep-residual-learning.md`; CVPR 2016)
  The original ResNet defines Option B as learned projection shortcuts when dimensions increase. EXP017/021 tested pool-first ResNet-D variants, so the original stride-2 projection remains a distinct transition mechanism.
- **Channels Last Memory Format in PyTorch** (official PyTorch tutorial: https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  Conv2d and BatchNorm2d support channels-last propagation through cuDNN, but the strongest documented GPU gains are reduced-precision large models. FP32 CIFAR tensors can be slower, so this is a timing-first hypothesis rather than an assumed optimization.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`; ICML 2025)
  Late averaging can improve generalization with annealing, but EXP018's uniform average lagged its online path. A recent-state EMA is mechanistically different but has limited ceiling on EXP010's already smooth tail.
- **CutMix** (`knowledge/papers/cutmix.md`; ICCV 2019)
  Regional mixing retains class-bearing pixels and is the accepted frontier mechanism. EXP026 shows that replacing half with whole-image Mixup worsens switch fit, motivating phase decoupling rather than another geometry substitution.

## Experimental History Review

- EXP010 remains the `94.15%` frontier: width-2 postactivation ResNet-20, all-parameter `1e-4` decay, N1/M7 plus 50% alpha-1 CutMix through 80%, then a hard weak tail. Its key health markers are 89.73% at the switch, 93.16% first weak, 0.1934 final NLL, and 26,898 steps.
- Increasing CutMix to 75% (EXP011) and replacing half with alpha-0.4 Mixup (EXP026) preserved exposure but deepened switch underfit. EXP026 recovered to 93.37% immediately after hard labels yet peaked at only 94.22%, suggesting the regularizer-to-refinement boundary—not more mixing—is the live augmentation question.
- Architecture attempts have narrowed. Full preactivation reached 94.22% but harmed switch fit; zero-gamma suppressed the strong phase; pool-first learned shortcuts worsened NLL and deterministic pooling caused a class transient; lost-depth/global or abrupt late-width changes failed. The original non-pooling Option-B projection is still untested.
- EXP018 rejects uniform late SWA, not short-timescale EMA. Its averaged model lagged its own online checkpoint by 0.17, while EXP010's accepted tail finished at its best, leaving only a modest averaging ceiling.
- Backward remains 75.46% of counted step time and VRAM is only 598.7 MiB. Host augmentation, transfer, loss, optimizer, and Python overhead are not limiting. FP32 channels-last is the simplest untested same-graph attack on cuDNN convolution layout, but official evidence is strongest for FP16 and therefore demands an early timing veto.
- Repeated immutable-corpus gates show that short loss improvements do not predict phase-scale accuracy and that new branches/optimizer paths can concentrate classes. Any representation change needs multi-step trajectory bounds; any data-policy comparison needs exact post-transform sources.

## Collected Ideas

- **CutMix-off / RandAugment-on bridge** — Keep the accepted strong loader through 70% counted time, then rebuild the same N1/M7 loader with hard default collation until the existing 80% switch to weak crop/flip. This simplification isolates label/region mixing from view strength and targets the late strong-fit decline seen in EXP011/026 while preserving the validated augmentation boundary.
- **Original ResNet Option-B transitions** — Replace only the two dimension-changing Option-A shortcuts with stride-2 `1x1` Conv-BN projections, leaving all identity blocks and residual branches intact. It targets lossy zero-padding/channel transport without the pool-first operator implicated by EXP017/021 and has small parameter/compute cost.
- **FP32 channels-last accepted graph** — Convert model weights and each GPU input to `torch.channels_last` while keeping all mathematics, optimizer, and data unchanged. It attacks the measured 75.46% convolution/BN backward bottleneck; any accuracy effect must come through extra fixed-time exposure, and a paired timing veto should make this cheap to reject.
- **Short weak-tail EMA evaluator** — Track a decay-0.995 parameter EMA only after 80% and evaluate that recent-state model at the existing once-per-epoch cadence with online BN buffers. It avoids EXP018's uniform backward bias but may have too little variance ceiling because the accepted tail already ends at its best.
- **Ghost BatchNorm groups of 64** — Reshape each training BN input into two virtual groups while retaining optimizer batch 128. This injects normalization noise without changing image throughput, but touches all 19 normalization sites and may compound an accuracy bottleneck that currently looks like strong-view fit rather than oversized-batch generalization.
- **Strong-only residual stochastic depth** — Randomly suppress a small fraction of same-width residual branches during N1/M7, restoring the full graph for the weak tail. Literature supports residual-path ensembles, but this model is shallow and prior branch suppression/extra regularization repeatedly worsened strong fit.
- **Convolution-gradient centralization** — Before ordinary SGD, subtract each Conv2d weight gradient's output-channel-wise mean while leaving BN, bias, decay, and momentum unchanged. This could regularize filter updates at tiny cost, but it is an optimizer-path intervention with no local positive evidence and must clear immutable-corpus concentration gates.
- **Moonshot manifold CutMix** — On a minority of already-selected mixed batches, replace a spatial region in the layer-3 input rather than pixels and mix targets by area. Semantic-space regional composition might preserve useful low-level views while regularizing class features, but it couples policy RNG to the forward graph and has high implementation and attribution risk.

## Combinations

- **CutMix-off bridge + Option-B projection**: learned channel transport could improve strong representation, while the hard N1/M7 bridge could consolidate it before annealing. The combination plausibly addresses both capacity transport and late regularization, but it destroys attribution and should only follow isolated evidence.
- **Channels-last + Option-B projection**: layout speed could fund the projection's extra convolution/BN cost, potentially preserving exposure while improving transition features. Neither benefit is guaranteed in FP32, so combining before isolated timing and accuracy results would conceal two independent failure modes.
- **CutMix-off bridge + tail EMA**: an earlier hard-label bridge could reduce the tail's starting bias while EMA damps late checkpoint noise. This spans two phases and might beat either alone, but EXP010's smooth tail makes the averaging half low-value relative to the extra complexity.

## Candidate Ideas

### CutMix-Off, RandAugment-On Refinement Window
**Summary**: Keep the accepted N1/M7 plus 50% CutMix path through 70% counted time, then disable only CutMix through a forkserver-safe shared collator flag while retaining N1/M7 until the existing 80% hard weak-tail switch. Explicit worker provenance bounds prefetch drain without rebuilding or reseeding the loader. Full proposal: `proposals/idea-01.md`.

**What it targets**: The accuracy limiter is recoverable late strong-phase fit. EXP011 and EXP026 show large immediate gains when soft mixed targets disappear, while EXP005 shows dropping RandAugment early is harmful; this design separates those effects.

**Reasoning**: It preserves roughly 87.5% of accepted CutMix exposure and the complete validated RandAugment phase, then allocates about 2,700 high-LR updates to hard-label strong views. It adds no GPU path and should preserve the accepted 26.9k-step exposure. The 70% boundary is locally motivated rather than literature-validated, and removing CutMix jointly changes pixels and targets.

**Sources**: EXP005, EXP010, EXP011, EXP026; `knowledge/papers/cutmix.md`; `knowledge/papers/randaugment.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium — compact production change, demanding forkserver propagation/provenance and immutable-policy gates.

**Risk Assessment**: Accepted p=0.5 CutMix may have no adaptation debt and hard N1/M7 may not reproduce the recovery seen after all strong augmentation disappeared. Shared-state prefetch smear must remain bounded and observable; a miss cannot tune the boundary in place.

### FP32 Channels-Last Accepted Graph
**Summary**: Convert the accepted FP32 model weights and CUDA images to `torch.channels_last`, preserving public NCHW shapes, values, architecture, optimizer, augmentation, schedule, and evaluator. Proceed to accuracy only if five fresh paired trials prove at least a 3% synchronized step reduction and at least 27,700 projected steps. Full proposal: `proposals/idea-03.md`.

**What it targets**: The measured systems bottleneck: convolution/BN backward is 75.46% of GPU-stage time, with model forward plus backward at 97.57%; host and optimizer changes have little ceiling.

**Reasoning**: Official PyTorch documentation confirms cuDNN Conv/BN channels-last support and semantic dimension preservation. More accepted-recipe exposure is the only accuracy mechanism. This is a cheap timing-first probe with high attribution if it wins, but official GPU gains are strongest for reduced precision and large vision models; FP32 32x32 kernels or layout conversions may be slower. Faster epochs also require a fixed 19-look evaluation schedule to avoid max-metric bias.

**Sources**: official PyTorch channels-last tutorial; `02-system-understanding.md`; EXP013, EXP016, EXP023; `proposals/idea-03.md`.

**Estimated Effort**: medium — tiny production diff, rigorous numerical equivalence/layout propagation and paired timing.

**Risk Assessment**: The H20 may select no faster FP32 kernels, Option-A slice/pad may break propagation, and extra updates have not been causally shown to improve the accepted recipe. Numerical reduction-order drift must stay within pre-registered tolerances.

### Original Option-B Strided Projection Shortcuts
**Summary**: Replace only the two dimension-changing Option-A slice/pad shortcuts with the original ResNet Option-B stride-2 `1x1` Conv-BN form. Same-shape shortcuts remain identities and all residual branches, widths, optimizer, and data policy stay unchanged. Full proposal: `proposals/idea-02.md`.

**What it targets**: Representation transport at stage boundaries. Option A copies existing channels but inserts zeros into every newly introduced shortcut channel; Option B gives all output channels a learned normalized direct path while preserving the accepted `::2` sampling lattice.

**Reasoning**: The original ResNet paper establishes this as a canonical transition. It is not the pool-first ResNet-D mechanism rejected by EXP017/021, so it cleanly tests whether those failures came from pooling or from learned shortcut transport. EXP017's higher switch fit suggests transition capacity can help, but its worse NLL is serious counterevidence. Active random projection/BN paths require shared-initialization isolation and immutable-corpus scale/concentration gates.

**Sources**: `experiments/027/papers/deep-residual-learning.md`; EXP017, EXP021, EXP024, EXP025; `proposals/idea-02.md`.

**Estimated Effort**: medium — small model diff, substantial initialization/recruitment/timing verification.

**Risk Assessment**: The ResNet paper chose economical Option A for CIFAR, and learned projection/BN may itself be the late-generalization liability seen in EXP017. The new paths change the initial function and add backward cost, though nearby timing suggests at least 26k steps.

## Review

Claude's independent review selected the CutMix-off/RandAugment-on window with `7/10` evidence/reasoning and `6/10` potential impact. It judged this the only finalist directly aligned with the generalization/refinement limiter and backed by dense local evidence, while emphasizing that EXP010's accepted p=0.5 path already has healthy switch fit and ends at its best; the proposed “adaptation debt” may not exist. The recoveries in EXP011/026 also followed removal of all strong augmentation from over-regularized states, not CutMix-only removal under continuing N1/M7.

I adopt those concerns as load-bearing interpretation. The 70% point is one pre-registered curriculum operating point aligned with the existing last strong-phase checkpoint and a meaningful 10%-of-budget hard-N1/M7 bridge, not a claimed optimum. A valid miss with no switch-fit rise broadly falsifies the proposed debt mechanism; a miss with higher switch fit shows that late regional examples were useful for generalization despite suppressing fit. No boundary may be tuned after observation. Full review: `01-idea-review.md`.

The critic scored Option B `5/10` evidence and `4/10` impact because EXP017's learned projection+BN—not pooling alone—is the parsimonious source of its worse NLL. Channels-last scored `3/10` and `2/10`: official FP32 evidence is weak, and more updates alone do not address the diagnosed accuracy limiter. Those ideas remain diagnostic probes, not the lead metric bet.

## Idea Evaluation

Adopt **CutMix-Off, RandAugment-On Refinement Window**, the reviewer's pick. It preserves the four validated pillars—width-2 capacity, all-parameter decay, N1/M7, and early 50% CutMix—while testing the one remaining temporal composition question with no new GPU work. Option B has cleaner architectural attribution but stronger local counterevidence; channels-last has a cheap veto but no demonstrated accuracy mechanism beyond exposure.

## Chosen Idea
**Selected**: CutMix-Off, RandAugment-On Refinement Window

**Why this idea**:
EXP010 proves early plateau CutMix is valuable, EXP005 proves RandAugment should remain until 80%, and EXP011/026 show that mixed-target deficits can recover immediately under hard labels. The candidate composes those findings instead of retrying a failed strength or geometry: retain accepted CutMix through 70%, then use hard N1/M7 views for 10% of the budget before the unchanged weak tail. It is exposure-neutral and makes a clear negative result possible: no switch-fit recovery means there was no debt to repair; recovery without top-1 gain means late CutMix generalization outweighed fit.

**Hypothesis**:
Disabling only CutMix at 70% while retaining RandAugment N1/M7 until 80% will preserve most regional supervision, add roughly 2,700 high-LR hard-label strong updates, maintain at least 26,629 total steps, raise the 80% switch checkpoint to at least 89.73%, and improve `best_test_acc` from 94.15% to at least 94.25%. The point prediction is 94.32%; a valid lower result or any registered shared-worker/lifecycle failure rejects this exact operating point without a boundary retry.

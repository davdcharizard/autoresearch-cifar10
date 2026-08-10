# Brainstorm EXP-034
**Created**: 2026-08-06

## Web Search & Literature Review

- **PyTorch Channels Last Memory Format tutorial** (`knowledge/references/pytorch-channels-last.md`; https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  Channels-last preserves logical NCHW semantics while changing physical strides; CUDA convolution and BatchNorm can propagate it, but unsupported operators may insert conversions and FP32 32x32 gains must be measured end to end.
- **Torchvision ResNet implementation and PyTorch initialization documentation** (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py; https://docs.pytorch.org/docs/stable/nn.init.html)
  Official ResNet initializes Conv2d with Kaiming fan-out while PyTorch's default fan-in preserves forward variance. BN makes the forward scale approximately invariant, but the local optimizer's relative updates are not invariant.
- **Huang et al., Deep Networks with Stochastic Depth** (`knowledge/papers/stochastic-depth.md`)
  Mini-batch residual bypass can regularize and shorten the expected training graph, but its published benefit is concentrated in networks far deeper than this nine-block model.

## Experimental History Review

- EXP010 remains the 94.15% frontier: width-2 postactivation ResNet-20, N1/M7 plus p0.5 alpha-1 CutMix through 80%, then weak hard-label cosine refinement. The data curriculum, ordinary momentum, all-parameter decay, and simultaneous boundary are now strongly validated.
- Accuracy changes have plateaued: preactivation and balanced Mixup reached 94.22% but missed the +0.10 gate, while stronger CutMix, earlier hard labels, more tail LR, momentum reset, and both aggressive and sparse unlabeled occlusion failed. New data-policy interpolation has poor expected value.
- Recruited branches, optimizer-state paths, max aggregation, abrupt width changes, BF16, and even mild erasing have repeatedly produced candidate-only class concentration. Any representation candidate needs immutable production-batch, update, and class-geometry gates; lower short loss never clears them.
- The measured systems limiter remains convolution/BN backward at 75.46% of the counted step. Peak VRAM is only 599 MiB of 97,871 MiB, while Python/optimizer overhead has under 1% ceiling. Layout or graph changes are more plausible exposure levers than helper code.
- EXP013 rejected batch 256 because image throughput rose only 18.91% against its 20% gate, and EXP029 rejected safe all-Conv gradient centralization at 1.97% overhead. Exposure is precious, but its accuracy value on the accepted recipe is still not causally established.
- Untried gaps include convolution memory layout/kernel selection, canonical Conv-only fan-out initialization, shallow batchwise stochastic depth, exact classifier symmetry, and BN/statistics refinements. The first three span systems, initialization, and joint regularization/compute mechanisms without disturbing the accepted data curriculum.

## Collected Ideas

- **End-to-end channels-last layout** — Convert the initialized model and every 4-D training/evaluation input to `torch.channels_last`, retaining logical NCHW shapes and all math. This attacks the 75.46% convolution/BN backward bottleneck directly; official PyTorch support is real, but tiny FP32 CIFAR kernels may gain nothing or pay conversion overhead.
- **cuDNN stable-shape autotuning** — Enable `torch.backends.cudnn.benchmark=True` before model work so cuDNN can select kernels for the fixed 128x32x32 training shape. Startup search is outside the counted budget and could raise exposure, but determinism changes, the terminal eval batch is smaller, and modern heuristics may already select the same kernels.
- **Conv2d-only Kaiming fan-out** — Match torchvision ResNet's Conv initialization while preserving the Linear call exactly. Only the 3->32 stem and two widening convolutions change scale, so recurring compute and RNG draw order stay fixed; BN approximately preserves initial features but may amplify relative data updates by 10.7x in the stem.
- **Zero classifier bias only** — Explicitly zero the final Linear bias after its accepted constructor while preserving every weight draw. This restores initial class-offset symmetry at zero recurring cost and may reduce early class imbalance, but BN-pooled random features dominate the logits and the effect may be below ten examples.
- **Conservative batchwise stochastic depth** — Use one predeclared high-survival schedule over the nine residual blocks, bypassing complete residual branches on a mini-batch and applying expectation-preserving scaling. It can jointly regularize and reduce backward work, but one skipped block is a large fraction of this shallow model and variable graphs/RNG can destabilize geometry.
- **In-place ReLU simplification** — Mark safe post-BN and post-add ReLUs in-place to reduce activation traffic without changing their mathematical function. It is a code simplification with possible memory-bandwidth benefit, yet peak memory is not limiting and convolution backward dominates; alias safety and actual kernel savings are uncertain.
- **Per-channel CIFAR standard deviation** — Replace the intentional `(1,1,1)` scale with canonical CIFAR channel standard deviations while leaving centering/data policy fixed. It changes input-channel conditioning at no recurring operation, but the first Conv+BN largely cancels positive channel scaling and fixed LR/decay absorb the remaining reparameterization unpredictably.
- **Weak-tail BN-statistics freeze** — Freeze BN running means/variances at the 80% policy boundary while keeping affine parameters trainable, preventing weak-tail statistics from replacing the broad-view representation. This is orthogonal to weights and nearly free, but evaluation uses weak clean images, so preventing adaptation may preserve exactly the wrong distribution.
- **Moonshot full residual Fixup-style reparameterization** — Remove BN dependence via analytically scaled residual initialization and biases while preserving depth/width. It could reduce BN backward and change generalization, but it is a wholesale graph/optimizer intervention with high collapse risk and exceeds the isolation warranted by the current plateau.

## Combinations

- **Channels-last + cuDNN benchmark**: layout exposes a different kernel family and autotuning could select the best implementation for the exact fixed shapes, plausibly beating either alone. The combination is attribution-heavy and should follow a channels-last-only result rather than be the first test.
- **Channels-last + Conv fan-out**: one component targets exposure and the other gradient transport, so a faster trajectory could exploit a better initialization. Their independent effects on step count and early optimizer geometry would be impossible to separate; test each alone first.
- **Channels-last + stochastic depth**: layout speeds active convolutions while block bypass removes some of them, potentially compounding exposure gains with regularization. Variable execution and memory format together create a large systems/trajectory surface, so this is a later composition only after both primitives pass independently.
- **Zero FC bias + Conv fan-out**: canonicalizes both classifier symmetry and convolution scaling with no recurring work. Fan-out dominates the optimizer geometry, making the tiny bias effect uninterpretable; retain a single initialization intervention.

## Candidate Ideas

### Conv2d-Only Kaiming Fan-Out Initialization
**Summary**: Change `_weights_init` so Conv2d uses `mode="fan_out", nonlinearity="relu"`, exactly matching torchvision ResNet, while Linear retains the literal accepted default fan-in call. Sixteen same-width Conv tensors remain bitwise identical; only the stem and two widening convolutions rescale, with no new runtime operator. Full specification: `proposals/idea-02.md`.

**What it targets**: Representation formation and backward transport through the stem/stage expansions without spending any of the fixed 300-second exposure budget.

**Reasoning**: The official convention provides direct implementation precedent, and the change preserves tensor shapes, draw order, graph, and all accepted data/optimizer settings. BN should approximately preserve initial normalized features, but that is also the main risk: the smaller stem norm can amplify raw gradients about 3.27x and relative updates about 10.67x. Exact construction/RNG proofs and byte-identical 200-strong/64-weak trajectory gates are therefore mandatory.

**Sources**: PyTorch `nn.init` docs; installed torchvision ResNet source; EXP012/015/024/025/031/033; `proposals/idea-02.md`.

**Estimated Effort**: medium-high.

**Risk Assessment**: Medium-high. BN may erase most benefit while optimizer reparameterization creates the downside; the stem's effective-step amplification can fail class/update gates. Its upside may also be below the +0.10 threshold because only three tensors differ.

### Three-Percent Batchwise Stochastic Depth
**Summary**: During only the strong phase, independently drop each of the six same-width non-entry residual branches with fixed p=0.03 per mini-batch, never dropping transitions; surviving residuals scale by 1/0.97. A dedicated CPU generator preserves all accepted RNG streams, and the weak tail/evaluation always executes the complete unscaled network. Full specification: `proposals/idea-03.md`.

**What it targets**: Jointly reduce the convolution/BN backward bottleneck and introduce a mild effective-depth ensemble without changing the accepted N1/M7+CutMix curriculum.

**Reasoning**: Stochastic depth has CIFAR evidence and true batchwise bypass can save both forward and backward work, unlike output-only masks. The proposal is conservative: stage entries stay active, expected depth is 8.82/9, complete-graph probability is 83.3%, exact branch/RNG/BN/update gates precede five paired timing runs, and production needs >=1% measured schedule speedup. The transfer from a 1,202-layer paper network to nine blocks remains weak.

**Sources**: `knowledge/papers/stochastic-depth.md`; EXP012/015 identity-underfit results; EXP019/021/026 corpus protocol; `proposals/idea-03.md`.

**Estimated Effort**: high.

**Risk Assessment**: High. Skipping one block removes 11.1% of local residual depth, sparse BN/momentum state may hurt the boundary, Python branching may erase a theoretical 1.69% compute saving, and conditional paths can create early class geometry failures.

### End-to-End FP32 Channels-Last Training
**Summary**: Convert the ordinarily initialized model to `torch.channels_last`, transfer every counted training input with the same memory format, and normalize evaluator inputs once at the `ResNet.forward` boundary. Preserve logical NCHW shapes, values, FP32/default-TF32, optimizer/data/model semantics, and charge the host-to-device restride. Add a conservative 19-look cap that reserves the terminal evaluation so faster epochs cannot game the maximum. Full specification: `proposals/idea-01.md`.

**What it targets**: The measured systems limiter in `02-system-understanding.md`: model backward consumes 75.46% and forward 22.11% of GPU-stage time, mostly convolution/BN, while visible host/optimizer overhead has little ceiling.

**Reasoning**: Official PyTorch documentation supports CUDA Conv2d/BatchNorm propagation in channels-last without changing logical tensor semantics. It preserves batch-128 update noise and can only help accuracy through additional same-recipe exposure. The proposal requires stride hooks, hidden-conversion profiling, immutable-corpus numerical safety, and seven fresh timing pairs with a load-bearing >=3% speedup before production; this prevents a shape/dtype/layout artifact from becoming a scored run.

**Sources**: `knowledge/references/pytorch-channels-last.md`; `02-system-understanding.md`; EXP013 timing/evaluation-count findings; EXP029 nominal-overhead finding; `proposals/idea-01.md`.

**Estimated Effort**: high.

**Risk Assessment**: High. Tiny FP32 32x32 kernels may not benefit, Option-A slice/pad or pooling/view may force conversions, and even >=3% more seed-42 updates may not improve an already saturated trajectory. The evaluator cap expands the tracked diff but is an integrity control, not an accuracy lever.

## Review

Claude's independent review (`01-idea-review.md`) selected **Conv2d-Only Kaiming Fan-Out Initialization**, scoring evidence/reasoning 6.5/10 and potential impact 5/10. It judged channels-last's systems protocol excellent but its sole accuracy link—roughly 3% more exposure—weak against EXP026/032, and likely blocked by the FP32 32x32 timing gate. It rejected stochastic depth because identity-oriented changes and added strong regularization repeatedly deepen local underfit, while its theoretical 1.69% compute saving is vulnerable to branch overhead.

I adopt the selection and its main caution. Fan-out's official torchvision precedent is not local accuracy evidence; BN approximately cancels the forward rescaling, so amplified relative optimizer steps are both the intended lever and the principal failure mode. The chosen spec therefore keeps exact tensor/RNG proofs, byte-identical strong/weak corpus replay, candidate-only concentration vetoes, and explicit whole-model/per-layer update bounds. I remove confidence implied by a narrow point prediction: the registered claim remains the formal >=94.25% threshold, but the plausible outcome band is wide and a safe sub-threshold result is the modal risk. No layer subset, interpolation, LR compensation, or post-veto rescue is allowed.

## Idea Evaluation

- **Conv-only fan-out** — Advance. It is the cleanest untried representation/optimizer reparameterization, adds no recurring cost, preserves accepted data and graph semantics, and has a precise safety screen for its known stem-step risk.
- **Channels-last** — Defer. It remains a valuable systems-enabling experiment, but a standalone accuracy run depends on two weak links: >=3% speedup for tiny FP32 kernels and an accuracy gain from only extra exposure.
- **Three-percent stochastic depth** — Reject. It compounds a shallow identity-depth perturbation with extra strong regularization, both opposed by repeated local underfit evidence, for marginal theoretical compute savings.

## Chosen Idea
**Selected**: Conv2d-Only Kaiming Fan-Out Initialization

**Why this idea**:
It changes only three unequal-fan Conv tensors, preserves every accepted runtime/data choice and RNG draw, and costs no fixed-budget exposure. Unlike the other finalists, its accuracy mechanism is not directly contradicted by local exposure or strong-regularization failures. Its downside is measurable before scoring: the 0.306x stem norm may amplify relative data updates about 10.67x, so byte-identical trajectory geometry—not initial BN similarity or lower loss—decides whether production is safe.

**Hypothesis**:
Changing Conv2d initialization from default fan-in to torchvision-style fan-out while leaving Linear initialization literal will preserve near-identical initial train-mode features, safely improve gradient transport through the stem/stage expansions, retain at least 99% accepted exposure, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. The plausible result band is intentionally wide (roughly 94.00-94.35%); any safety veto or valid miss retires this exact all-Conv fan-out point without layer selection or scale tuning.

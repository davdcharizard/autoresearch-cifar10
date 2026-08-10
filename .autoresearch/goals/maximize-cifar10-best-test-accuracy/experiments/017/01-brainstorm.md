# Brainstorm EXP-017
**Created**: 2026-08-06

## Web Search & Literature Review

- **Bag of Tricks for Image Classification with Convolutional Neural Networks** (`papers/resnet-d-downsampling.md`; CVPR 2019): ResNet-D preserves transition information with average pooling followed by learned stride-1 shortcut projection; direct evidence is ImageNet rather than CIFAR BasicBlocks.
- **Deep Residual Learning for Image Recognition** (https://arxiv.org/abs/1512.03385): projection shortcuts are the original learned alternative when dimensions increase; option-B evidence supports legitimacy but reports only a small advantage and does not isolate this CIFAR recipe.
- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (`papers/eca-net.md`; CVPR 2020): a tiny channel-axis convolution can recalibrate residual features without SE dimensionality reduction and with negligible nominal FLOPs.
- **Balanced Mixture of Supernets for Learning the CNN Pooling Architecture** (`../016/papers/resnet20-downsampling-search.md`; https://proceedings.mlr.press/v224/roshtkhari23a.html): direct CIFAR-10 ResNet20 evidence that downsampling configuration materially affects accuracy.
- **Amortized Nesterov's Momentum** (https://proceedings.mlr.press/v124/zhou20a.html): Nesterov has a plausible stochastic acceleration/noise tradeoff, but no direct evidence for the accepted long high-LR recipe.

## Experimental History Review

- EXP-010 remains the 94.15% frontier: active postactivation width-2 branches, N1/M7 plus p=0.5 CutMix through 80%, and a weak hard-label tail reached 89.73% at the switch and 94.15% final/best in 26,898 updates.
- Representation changes that suppress active residual learning are the dominant recurring failure: full preactivation and selective zero-gamma lowered switch fit by 2.85-3.25 points. Any new architecture should keep residual branches active from step one.
- Compute-funding premises failed before accuracy twice: batch256 delivered only 1.189x image throughput, and full-forward BF16 width3 triggered candidate-only class concentration. Favor FP32 candidates unless a distinct precision policy has independent justification.
- Stronger CutMix, decay changes, label smoothing, and early weak switching failed; preserve the accepted data, regularization, and schedule around a single new mechanism.
- EXP-014 shows that nominally zero-output additions can have uncontrolled first updates. New shortcut or attention paths require real-batch update-scale checks, not only functional equality.
- Two EXP-016 runners-up remain untested: isolated Nesterov has excellent attribution but modest expected impact; full-path BlurPool is coherent but overlaps crop augmentation and risks detail suppression/exposure loss.

## Collected Ideas

- **Learned anti-loss transition shortcuts** — replace only the two raw Option-A slice/zero-pad paths with `2x2` average pooling, learned stride-1 `1x1` projection, and BatchNorm. Targets information and channel loss at stage boundaries while preserving the accepted residual path, ordering, and all active branches; literature support is indirect but downsampling is a direct CIFAR ResNet20 lever.
- **ECA residual channel attention** — apply deterministic-kernel ECA gates to each block's residual output immediately before shortcut addition. Targets inefficient channel allocation under strong spatial augmentation with almost no parameters, but global pooled gates may react badly to CutMix mixtures and add backward kernels.
- **Isolated Nesterov on EXP-010** — change only `nesterov=True`, keeping the validated long plateau, decay, data, and graph. It resolves the EXP-001 schedule confound at near-zero compute cost, though its likely gain is close to the ten-image threshold and first updates are 1.9x ordinary momentum.
- **Full-path BlurPool transitions** — consistently filter both residual and shortcut paths before stride-2 sampling. It targets aliasing and transition information with published support, but padded random crops already create translation pressure and the dense transition convolution can reduce exposure.
- **Strong-phase stochastic depth** — use a small linearly increasing drop-path probability only on ordinary same-shape blocks during the 80% strong phase, then disable it for the weak tail. It could regularize width-2 capacity without stronger CutMix, but the short strong phase already shows underfit sensitivity and branch dropping may repeat that signature.
- **Smooth activation substitution** — replace ReLU by SiLU throughout the accepted graph while preserving normalization, shortcuts, width, and data. Smooth nonzero negative gradients may improve representation learning, but every activation kernel changes, compute cost may be material, and local evidence is absent.
- **Moonshot: learned transition shortcuts plus ECA** — combine information-preserving stage entry with channel recalibration throughout the network. The mechanisms are complementary, but coupling them before either is isolated would weaken attribution and make first-update/timing diagnosis unnecessarily complex.

## Combinations

- **Learned shortcut + ECA**: preserve more boundary information and then allocate expanded stage features adaptively. This could beat either alone if transition transport and channel selection are both limiting, but should follow isolated evidence because both alter residual sums.
- **Nesterov + late weight averaging**: faster online response plus trajectory smoothing could improve basin quality more than either alone. Existing tail best-final gaps are tiny, so averaging has weak local support and the combination is deprioritized.
- **ECA + channels-last**: layout speed could offset ECA's extra kernels while keeping FP32. This is only defensible after independent timing shows channels-last helps the accepted tiny convolutions; it cannot rescue a failed ECA timing gate inside one experiment.

## Candidate Ideas

### Identity-Scale ECA Residual Channel Attention

**Summary**: Add one deterministic ECA module to each of nine residual blocks after `bn2` and before shortcut addition. Use stage kernels `3/3/5`, zero-initialized channel convolutions, and the single identity-centered gate `2*sigmoid(logit)`, so every initial gate is exactly one and the accepted model function/shared gradients/RNG are preserved. The candidate adds 33 scalar weights for 1,073,995 total parameters. See `proposals/idea-02.md`.

**What it targets**: Conditional use of accepted width-2 capacity. ECA lets each block recalibrate residual channels from its current example, potentially allocating features better across RandAugment and mixed regions without changing width, data, or residual initialization.

**Reasoning**: ECA reports low-cost gains across ResNet backbones, and the identity-centered adaptation directly addresses the recurring local failure of halving/zeroing residual activity. Its bounded `(0,2)` gate avoids the unbounded raw-max failure family while remaining learnable on backward one. Nominal FLOPs are not enough: nine sequential pool/Conv1d/sigmoid/multiply chains may be launch-bound, and global CutMix descriptors may be semantically conflicted. Hard/soft update gates, 64 real batches, paired timing, and 97% exposure retention are conjunctive.

**Sources**: `papers/eca-net.md`; EXP007/010/012/014/015/016; `proposals/idea-02.md`.

**Estimated Effort**: Medium. Module/placement code is compact; exact identity/RNG proof and all-block timing/recruitment checks dominate effort.

**Risk Assessment**: Medium-high. Published evidence is deeper ImageNet networks, channel adjacency has no guaranteed semantics, centered scaling differs from paper-standard ECA, global pooling may mishandle CutMix, and tiny sequential kernels previously cost much more than FLOPs suggested.

### Learned Anti-Loss Transition Shortcuts

**Summary**: Replace only the two raw Option-A transition shortcuts with fixed nonoverlapping `2x2`, stride-2 average pooling followed by a learned bias-free stride-1 `1x1` projection and BatchNorm. Keep both accepted stride-2 residual branches and all seven same-shape identity shortcuts unchanged. Fork constructor RNG and use a marker initializer so every shared accepted tensor and the post-construction RNG remain bitwise aligned; the two full-scale Kaiming projections add 10,624 parameters for a total of 1,084,586. See `proposals/idea-01.md`.

**What it targets**: Stage-boundary information transport. The accepted shortcut retains one of four spatial samples and supplies no shortcut signal to newly introduced channels, while width capacity is the strongest local accuracy lever. The candidate uses every `2x2` input and learns transport into all expanded channels without suppressing the active residual path.

**Reasoning**: ResNet-D supports pool-before-projection as a transition-information improvement, the original ResNet establishes learned dimension-matching shortcuts, and direct CIFAR ResNet20 research shows downsampling choices matter. Unlike EXP012/015, the residual branches retain accepted ordering and full activity. Because a random full-scale shortcut can interfere with the residual sum, production is conditional on hard/soft first-update, 200-distinct-batch concentration, paired timing, at least 25,500 projected steps, and no extra metric observations.

**Sources**: `papers/resnet-d-downsampling.md`; `../016/papers/resnet20-downsampling-search.md`; original ResNet; EXP007/010/012/014/015/016; `proposals/idea-01.md`.

**Estimated Effort**: Medium. Two modules and initializer plumbing are localized, but branch-scale, RNG, BN, real-batch safety, and paired timing gates are substantial.

**Risk Assessment**: Medium-high. The mechanism has the highest direct local leverage, but learned full-scale shortcuts can weaken identity transport, destructively interfere, dilute CutMix boundaries, add phase-sensitive BN state, or lose enough updates to erase a modest gain.

### Isolated PyTorch Nesterov Momentum

**Summary**: Add only `nesterov=True` to the accepted SGD construction, retaining momentum `0.9`, coupled all-parameter decay `1e-4`, the complete EXP010 schedule/data/graph/evaluator, and FP32 numerics. Installed semantics make the first update direction exactly `1.9x` ordinary momentum. See `proposals/idea-03.md`.

**What it targets**: Online optimizer trajectory and basin selection. It asks whether current-gradient correction helps the long noisy N1/M7+CutMix plateau and abrupt hard weak tail without changing capacity, representation, data, or exposure.

**Reasoning**: EXP001 never isolated Nesterov because it bundled an early 15% hold; every accepted experiment since EXP002 uses ordinary momentum. This is the cleanest unresolved ablation and should retain 99% of updates. Its evidence and ceiling are weaker than the architecture candidates: it adds no accuracy signal and a 94.25-94.30 pass is close to single-seed resolution. Exact recurrence/RNG gates, 200 distinct real batches, candidate-only concentration checks, and paired timing prevent first-step overshoot or hidden cost from reaching production.

**Sources**: Amortized Nesterov paper; installed PyTorch 2.9.1 SGD semantics; EXP001/002/010/015/016; `proposals/idea-03.md`.

**Estimated Effort**: Low code effort and medium verification effort.

**Risk Assessment**: Medium. Attribution and compute neutrality are excellent, but the accuracy mechanism is thin, the first step is 1.9x at LR 0.1, noisy lookahead can oscillate, coupled decay is effectively amplified, and the plausible upside barely exceeds the formal gate.

## Review

The mandatory external Claude review completed successfully and selected learned transition shortcuts; no fallback reviewer was used. The review judged this candidate the strongest combination of evidence and upside because it operates at the width-expansion boundaries while preserving every accepted residual branch at full activity. Three refinements are adopted. First, the mechanism is named precisely as shortcut information transport rather than full anti-aliasing because the residual stride-2 convolution remains unchanged. Second, direct CIFAR ResNet20 downsampling evidence is treated as load-bearing, while ImageNet ResNet-D supports direction rather than effect size. Third, the registered 4x shortcut/residual RMS gate remains a catastrophic veto, but any observed value above 2x is a predeclared diagnostic warning for analysis, never a post-hoc tuning trigger.

The review rejected ECA as the lead because nine sequential attention chains repeat a known all-block timing risk and its centered gate weakens literature transfer. It rejected Nesterov as the lead because its clean attribution attaches to the thinnest accuracy mechanism and a ceiling close to the ten-image gate. Full review: `01-idea-review.md`.

## Idea Evaluation

- **Learned transition shortcuts**: selected. Highest local leverage and strongest combined source support, with first-update and timing risks directly measurable before the one production run.
- **Identity-scale ECA**: not selected. Exact initial identity is attractive, but all-block sequential overhead, global CutMix descriptors, shallow channel counts, and modified ECA scaling lower both feasibility and transfer confidence.
- **Isolated Nesterov**: not selected. It remains a valid separately reviewed future experiment, not an EXP017 fallback; its attribution is excellent but expected impact is too close to noise.

## Chosen Idea
**Selected**: Learned Anti-Loss Transition Shortcuts

**Why this idea**:
The accepted Option-A transitions discard three of four shortcut samples and provide no shortcut signal to newly added channels exactly where width creates its strongest proven capacity gain. Pool-first learned normalized projection tests a coherent information-transport mechanism at only two boundaries, keeps the successful postactivation residual graph active from step one, and has stronger direct CIFAR downsampling support and a higher credible ceiling than ECA or Nesterov. It proceeds only if full-scale random shortcut outputs, first updates, 200 real batches, paired timing, exposure, evaluator count, and wall time all pass without modifying the fixed design.

**Hypothesis**:
Replacing the two Option-A shortcuts with `AvgPool2d(2,2) -> Conv1x1(stride=1) -> BatchNorm` will preserve more boundary information and learn transport into every expanded channel while retaining at least 25,500 updates. Under the unchanged EXP010 FP32 recipe it will keep the strong switch above the 87.08% underfit marker, preferably near 89.73%, and raise `best_test_acc` from 94.15% to at least 94.25%.

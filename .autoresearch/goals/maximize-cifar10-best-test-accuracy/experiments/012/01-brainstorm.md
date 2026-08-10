# Brainstorm EXP-012
**Created**: 2026-08-06

## Web Search & Literature Review

- **Wide Residual Networks** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/wide-residual-networks.md`): width was the largest local gain and supports spending modest extra compute on better feature routing within the accepted shallow model.
- **CutMix** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/cutmix.md`): preserve the validated alpha-1, p=0.5 plateau composition and hard-label weak tail while testing representation changes.
- Additional primary-paper evidence for projection, channel-attention, and preactivation candidates is developed in the experiment-scoped proposals.

## Experimental History Review

- The accepted frontier is EXP-010 at 94.15%: width-2 ResNet-20 plus p=0.5 CutMix/N1-M7 plateau and hard weak tail, with 26,898 steps and 598.7 MB peak VRAM.
- EXP-011 showed that p=0.75 crosses the strong-underfit marker and loses 0.15 points at equal exposure. CutMix probability above 0.5 and unsupported interpolation are low-value.
- All-parameter `1e-4` decay is bracketed by EXP-008/009; early weak switching, Cutout replacement, and label smoothing have approach-specific failures. Preserve the accepted optimizer, phase boundary, and data geometry.
- Width delivered the largest gain despite fewer updates, so remaining headroom plausibly lies in representation quality rather than exposure alone. The current raw Option-A downsampling shortcut discards spatial samples and pads half the channels with zeros; the network also lacks channel recalibration and modern residual initialization.
- The target is now 94.25%. A candidate needs enough upside to clear a one-seed 0.10-point gate while retaining a usable hard tail under 300 counted seconds.

## Collected Ideas

- **Learnable projection shortcuts** — replace only the two stride-2 Option-A slice/pad shortcuts with 1x1 stride-2 convolution plus BatchNorm projections. This targets information loss exactly where resolution and channels change, adding little sequential depth and modest parameters.
- **Squeeze-excitation blocks** — add global-pool bottleneck channel gates after each residual branch and before addition. This lets the width-2 model recalibrate CutMix/RandAugment features by channel, but adds small per-block sequential operations and may shorten the tail.
- **Preactivation residual units** — reorder each block to BN-ReLU-Conv-BN-ReLU-Conv with a clean identity shortcut. The literature reports easier optimization and cleaner signal propagation, though a 20-layer post-activation model may be too shallow for a large benefit.
- **Zero-initialize final residual BN scales** — keep the current blocks but initialize every `bn2.weight` to zero so the network begins near identity. This is a small optimization/representation change with negligible runtime, but may slow early learning under the fixed horizon.
- **Learnable scalar residual gates** — give each residual branch a trainable scalar initialized below one. This allows gradual branch recruitment and could stabilize composite training, but introduces an unvalidated initialization hyperparameter.
- **Anti-aliased downsampling** — precede stride-2 operations with fixed low-pass filtering to preserve shift stability under crops and patches. It directly targets spatial aliasing but adds memory traffic and a nontrivial implementation surface.
- **Classifier feature normalization** — L2-normalize pooled features and classifier weights with a learned scale. This may improve angular class separation under mixed labels, but changes logit geometry and confidence calibration late in training.
- **Moonshot: shallow multi-branch residual blocks** — replace each 3x3 branch with grouped multi-scale paths. This could increase feature diversity at similar parameters, but is a broad architecture rewrite with high fixed-time and attribution risk.

## Combinations

- **Preactivation + zero residual initialization**: identity-oriented ordering and initialization reinforce each other, potentially making a broad architecture change stable enough for the short horizon. The cross has more literature coherence than either isolated tweak but weaker attribution.
- **Projection shortcuts + squeeze-excitation**: preserve downsampling information and then recalibrate the wider channel set. The combination could improve both transport and selection, but should follow isolated evidence because both add compute and parameters.
- **Projection shortcuts + accepted CutMix**: use learned skip transport specifically to retain regional features across stage changes while leaving the successful training recipe untouched. This is a direct composition with the current best rather than a second regularizer.

## Candidate Ideas

### Option-B Projection Shortcuts at Stage Transitions
**Summary**: Replace only the two stride-2 Option-A slice/zero-pad shortcuts with bias-free 1x1 stride-2 convolutions followed by BatchNorm. Keep the seven same-shape shortcuts as exact identities and preserve the complete EXP-010 recipe.

**What it targets**: At each width-doubling boundary, Option A gives half the output channels no shortcut contribution. Learned projection transports and mixes all preceding channels into every new stage channel at the same sampled spatial phase.

**Reasoning**: The original ResNet Option-B evidence favors projection at dimension changes, and width was the largest local gain. The proposal adds only 10,624 parameters (0.99%); a paired H20 diagnostic measured 1.87% step cost and projects 26,404 steps, retaining 98.17% of EXP-010.

**Sources**: `proposals/idea-01.md`; He et al. ResNet and Han et al. PyramidNet primary sources cited there; EXP-007/010.

**Estimated Effort**: medium

**Risk Assessment**: Learned random projections sacrifice exact identity gradients, add two BN states that must adapt after the augmentation switch, and still sample only one spatial phase. CIFAR-specific Option-A/B evidence is absent.

### Canonical Full Preactivation
**Summary**: Reorder the width-2 ResNet-20 to paper-faithful BN-ReLU-Conv full-preactivation blocks with no post-add ReLU, explicit first/transition shortcut semantics, and a final BN-ReLU before pooling. Keep Option A and all accepted training mechanics; do not combine zero-gamma.

**What it targets**: Clean same-shape identity propagation and direct gradients through residual sums may improve optimization/feature transport without adding parameters or expected compute.

**Reasoning**: Identity Mappings reports CIFAR gains and provides official boundary semantics. The candidate keeps exactly 1,073,962 parameters and the same counts of convolutions, BNs, and ReLUs, with a pre-registered 97% exposure gate.

**Sources**: `proposals/idea-03.md`; He et al. Identity Mappings and official preactivation implementation cited there.

**Estimated Effort**: medium-high

**Risk Assessment**: Published gains grow with depth and may vanish at 20 layers. Transition preactivation changes what Option A transports, and signed post-add features can interact unpredictably with CutMix and the hard-tail BN distribution.

### Identity-Initialized Squeeze-Excitation
**Summary**: Add an r=16 squeeze-excitation gate to each of nine residual branches before shortcut addition. Use a centered `2*sigmoid` gate with zero expansion initialization so the initial network is exactly the accepted model, then learn per-example channel recalibration.

**What it targets**: Width-2 and CutMix create diverse regional channels, but the model has no mechanism to select channels by image content. SE can suppress augmentation-specific responses and reuse useful regional features during the hard tail.

**Reasoning**: The primary SE paper reports consistent gains from channel dependencies and supports all-stage placement. The candidate adds 8,778 parameters for exactly 1,082,740 total and pre-registers a 95% exposure floor because tiny sequential kernels can dominate their nominal FLOPs.

**Sources**: `proposals/idea-02.md`; Hu et al. CVPR 2018 and official SENet implementation cited there; EXP-010/011.

**Estimated Effort**: medium-high

**Risk Assessment**: Nine gates may cost many kernel launches, global descriptors of CutMix composites may be semantically ambiguous, r=16 leaves only two hidden units in stage 1, and identity initialization delays reduction-layer learning by one step.

## Review

Claude's mandatory adversarial review completed successfully and is preserved in `01-idea-review.md`. It scored identity-initialized SE highest on mechanism fit and ceiling, but made selection explicitly conditional on a paired H20 timing gate. The review identified SE's unmeasured small-kernel launch overhead as a no-go risk, projection shortcuts as a low-ceiling mechanism on a nine-block net, and preactivation as the cleanest compute-neutral paired probe despite weak depth-specific evidence. It also required using EXP-010's 93.16% first weak checkpoint to diagnose SE descriptor mismatch if SE ran.

The required five-trial H20 timing diagnostic is recorded in `00-se-timing.md`. SE was 1.23324x slower in training, projected only 21,810 steps, had a 1.23447x p95 ratio, and was 1.15206x slower in inference. These fail the 1.0526x, 25,553-step, 1.10x p95, and 1.10x inference gates by decisive margins. The review's conditional first choice is therefore infeasible and is retired without a full experiment run.

## Idea Evaluation

- **Identity-initialized SE**: highest representational ceiling and best match to the local width/CutMix trajectory, but rejected by its mandatory feasibility gate. The stable 23.32% training overhead would remove 5,088 accepted-frontier updates and too much of the hard tail.
- **Canonical full preactivation**: selected after the review-conditioned SE no-go. It preserves parameter and operator counts, should retain nearly all exposure, and gives the cleanest attribution because shared initialization can remain aligned. Its weakness is scientific rather than operational: shallow ResNet-20 may not benefit from the identity-gradient mechanism, so the final BN-ReLU feature conditioning and signed residual-sum representation are the more relevant local hypothesis.
- **Option-B projection shortcuts**: feasible and directly timed, but not selected because its own analysis predicts near-null impact. Option A already transports every incoming channel and the residual branch creates all outgoing channels; a learned same-phase 1x1 projection has limited upside in only two transitions.

## Chosen Idea
**Selected**: Canonical Full Preactivation

**Why this idea**:
The external critic explicitly designated preactivation as the compute-neutral choice if SE failed its exposure gate. That condition occurred decisively. Preactivation changes representation geometry without adding operations or parameters, preserves the validated EXP-010 data/optimizer recipe, and offers higher measurement quality than the feasible projection alternative. The implementation will foreground final feature conditioning and unrectified residual sums rather than claiming that a shallow network needs easier optimization.

**Hypothesis**:
Paper-faithful full preactivation across the width-2 ResNet-20, including a final BN-ReLU before pooling and unchanged Option-A transitions, will improve feature conditioning and preserve signed information in residual sums while retaining at least 97% of EXP-010's update exposure, raising `best_test_acc` from 94.15% to at least 94.25% under the unchanged p=0.5 CutMix training recipe.

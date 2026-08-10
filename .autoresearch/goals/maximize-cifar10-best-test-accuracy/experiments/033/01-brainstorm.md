# Brainstorm EXP-033
**Created**: 2026-08-06

## Web Search & Literature Review

- **PyTorch initialization docs** (official `torch.nn.init` documentation)
  `kaiming_normal_` defaults to `mode='fan_in'`; `fan_in` preserves forward activation variance while `fan_out` preserves backward gradient variance. The current code relies on the default for both Conv and Linear.
- **Official torchvision ResNet source** (`torchvision/models/resnet.py`)
  Torchvision explicitly initializes every Conv2d with Kaiming normal `mode='fan_out', nonlinearity='relu'`, while BatchNorm scale/bias are 1/0. This is strong implementation precedent, though it targets ImageNet ResNets rather than this CIFAR fixed-time recipe.
- **He et al., Delving Deep into Rectifiers** (ICCV 2015)
  Variance-aware rectifier initialization supports stable signal propagation, but fan choice depends on whether forward activations or backward gradients are prioritized. BatchNorm makes raw forward-scale arguments less direct locally.
- **How to Start Training** (NeurIPS 2018)
  Initialization/architecture jointly determine signal propagation in residual networks; the paper emphasizes fan-in forward preservation and residual scaling, cautioning that canonical implementation precedent is not universal proof.

## Experimental History Review

- EXP010 remains the 94.15% frontier. The accepted `_weights_init` applies default fan-in Kaiming normal to both Conv2d and Linear layers.
- EXP012/015/024/025/031 show representation/recruitment changes can suppress strong fit or create class transients. Any initialization change requires exact-corpus output/update/concentration gates even when BatchNorm makes initial logits similar.
- EXP030/032 reject more weak-tail amplitude and boundary state reset. The next lever should affect representation formation throughout the strong phase without per-step overhead.
- Fan-out differs from fan-in only when Conv input/output widths differ: the stem and two stride-2 expansion convolutions. Same-width 3x3 blocks retain identical scale and RNG draw order. This makes Conv-only fan-out a focused stem/transition initialization experiment, not a wholesale architecture rewrite.
- EXP006 rejected fixed 16x16 Cutout as a replacement for RandAugment; probabilistic small-area Random Erasing composition remains distinct but risks compounding CutMix/N1-M7 underfit.

## Collected Ideas

- **Conv-only Kaiming fan-out** — Change `_weights_init` so Conv2d uses `mode='fan_out', nonlinearity='relu'`, while Linear keeps its exact accepted fan-in Kaiming call. This changes scale only for the 3->32 stem and 32->64/64->128 expansion convolutions, matching official torchvision ResNet without adding runtime work.
- **Small classifier initialization** — Initialize the final Linear weight with `normal_(0,0.01)` and zero its bias while leaving all convolutions accepted. Smaller initial logits may avoid early confident class geometry and improve representation learning, but it changes the classifier gradient/relative update sharply and lacks direct local evidence.
- **Zero only FC bias** — Preserve every weight draw and set the default-random classifier bias to zero. It restores class symmetry with an extremely small intervention, but BN/global pooling features dominate logits and the effect may be below the gate.
- **Probabilistic small-area Random Erasing** — Compose p=0.25 erasing with area 2-10% after N1/M7 and before normalization, retaining CutMix. It supplies local occlusion invariance without EXP006's every-view 16x16 deletion, but may compound strong underfit and alter worker throughput.
- **Correct per-channel CIFAR standard deviation** — Replace `(1,1,1)` with standard CIFAR channel std values. This normalizes input scale canonically at zero recurring complexity, but the immediately following Conv+BN largely cancels affine input scaling and the source comment intentionally follows the original recipe.
- **TrivialAugmentWide** — Replace N1/M7 with a one-op broad-magnitude policy. It is worker-side and low-tuning, but abandons the strongest locally validated augmentation setting and may change host throughput/severity.
- **Fixed LeakyReLU 0.01** — Preserve negative activation gradients across the network with adjusted Kaiming nonlinearity. It touches every block and changes kernels/compute despite weak evidence that dying ReLUs limit a BN ResNet.
- **Moonshot orthogonal transition initialization** — Orthogonalize only the stem and two widening convolutions with gain chosen to match accepted Frobenius norms. It targets channel transport but introduces a new distribution without the direct official precedent of fan-out.

## Combinations

- **Conv fan-out + small classifier**: Align both convolution and classifier initialization with common ResNet practices, but their separate effects on early gradients and class geometry would be impossible to attribute. Test Conv fan-out alone first.
- **Conv fan-out + Random Erasing**: Better backward scaling might absorb stronger local regularization, but initialization and data geometry interact across the full strong phase and each needs its own safety/effect evidence.
- **Zero FC bias + small FC weights**: Produces a canonical near-zero-logit classifier, but bias zero alone is likely inert while the combined scale change is dominated by the weights; use one clear classifier rule if pursued.

## Candidate Ideas

### Conv2d-Only Kaiming Fan-Out Initialization
**Summary**: Initialize Conv2d weights with Kaiming normal `mode='fan_out', nonlinearity='relu'`, matching official torchvision ResNet, while leaving the Linear initialization literal unchanged. Only the stem and two widening convolutions change scale; 16 same-width Convs remain numerically identical. Full specification: `proposals/idea-01.md`.

**What it targets**: Backward gradient transport and strong-phase representation formation at the stem/stage expansions, with zero recurring work.

**Reasoning**: Fan-out has direct official ResNet precedent and preserves RNG draw order. BN approximately cancels initial positive scaling, but not optimizer geometry: the stem begins at 0.306x weight scale and can see roughly 10.67x relative data updates; widening Convs begin at 0.707x with roughly 2x relative updates. Exact initialization/BN/update/concentration gates are mandatory.

**Sources**: official PyTorch init docs; torchvision ResNet source; He et al.; EXP010, EXP012/015, EXP024/025, EXP031.

**Estimated Effort**: medium-high.

**Risk Assessment**: Medium-high. Only three tensors change and BN may erase benefit, while smaller norms can amplify normalized-function steps under LR 0.1.

### Conservative Small-Area Random Erasing Composition
**Summary**: Add p=0.25, area 2-10%, mean-fill Random Erasing after N1/M7/ToTensor and before normalization/CutMix, only in the strong phase. Expected raw erased pixels average about 1.5%. Full specification: `proposals/idea-03.md`.

**What it targets**: Local absence robustness complementary to class-bearing CutMix regions.

**Reasoning**: It is much milder and compositional rather than EXP006's every-view 25%-area Cutout replacement. But CutMix already provides regional occlusion; unlabeled deletion may only deepen short-phase underfit. Exact policy/RNG/corpus, live worker, and paired timing gates make it costly.

**Sources**: Random Erasing paper; torchvision docs; EXP004/006/010/011/026/027/029/031.

**Estimated Effort**: high.

**Risk Assessment**: High. Strong-regularization redundancy and worker overhead are both plausible blockers.

### Small Final-Classifier Initialization
**Summary**: Keep all Conv initialization accepted, but initialize FC weight from `Normal(0,0.01)` and zero its bias. This makes initial classifier weights 0.08x accepted and logits closer to uniform. Full specification: `proposals/idea-02.md`.

**What it targets**: Early class confidence and backbone recruitment without runtime overhead.

**Reasoning**: A small symmetric head could act as an implicit warmup, but classifier gradients need not shrink while backbone gradients initially scale with the tiny FC. The head may overwrite rapidly or undertrain the backbone; local branch/readout failures require exact multi-step geometry gates.

**Sources**: PyTorch/He initialization context; EXP010, EXP014, EXP024/025, EXP031.

**Estimated Effort**: medium.

**Risk Assessment**: High. Direct external evidence for std0.01 is weak, and the 12.5x scale reduction is more aggressive than fan-out.

## Review

Claude's independent review (`01-idea-review.md`) selected **Conservative Small-Area Random Erasing Composition**, scoring evidence/reasoning 6/10 and impact 7/10. Conv fan-out scored 7/10 and 3/10 because its official precedent is real but BN approximately cancels the three changed tensors' forward scale, leaving transient optimizer geometry with little evidence of final-accuracy impact. Small FC initialization scored 4/10 on both axes because std0.01 is unsupported and directly enters the collapse-prone unnormalized-head family.

I adopt the on-limiter selection. Random Erasing's local counter-evidence is serious: this recipe often rejects added strong regularization. The candidate is nevertheless distinct from EXP006/011/026/027 because it deletes only about 1.5% of source pixels on average, is probabilistic, composes with rather than replaces N1/M7+CutMix, and targets absence tolerance rather than stronger target mixing. The switch fit versus 89.73% and 87.08% is the decisive underfit diagnostic, but cannot truncate a valid production run.

RNG neutrality, effective post-CutMix erased area, exact-corpus concentration/update geometry, live worker headroom, and paired <=1% full-step overhead are load-bearing. A failure blocks production without p/area/fill/placement rescue. A bare pass remains weak single-seed evidence and must be interpreted with switch, first-weak, and NLL coherence.

## Idea Evaluation

- **Conservative small-area Random Erasing** — Advance. It uniquely targets generalization with a distinct mild absence prior and retains the full accepted curriculum.
- **Conv-only Kaiming fan-out** — Defer. Its scope and safety case are strong, but BN neutralizes most forward effect and official convention is not direct accuracy evidence.
- **Small classifier initialization** — Reject. The aggressive 12.5x head-scale reduction lacks directional evidence and risks early backbone disconnection/head overwrite.

## Chosen Idea
**Selected**: Conservative Small-Area Random Erasing Composition

**Why this idea**:
It has the only direct generalization mechanism and the highest plausible ceiling: sparse mean-filled absence may complement class-bearing CutMix without replacing any validated component. Its main risks are measurable before scoring through immutable policy geometry, trajectory safety, live worker, and full-step timing gates.

**Hypothesis**:
Adding p=0.25 Random Erasing with 2-10% area, mean fill, and RNG-neutral worker semantics only to strong source views will erase about 1.1-1.9% of pixels overall, preserve at least 99% exposure and healthy switch fit, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.30%; any veto or valid miss retires this exact policy without tuning.

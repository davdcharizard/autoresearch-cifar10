# Brainstorm EXP-045
**Created**: 2026-07-27

## Web Search & Literature Review

- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): uniform target mass can reduce overconfidence, but accepted mixup already supplies soft labels.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): preserve the validated clean hard-label tail if target regularization is tested.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve accepted capacity and focus architectural change on inexpensive information flow.
- **RandAugment** (`knowledge/papers/randaugment.md`): the accepted capacity gain depended on early image invariance, motivating an architecture-side phase-stability test.

No network or new retrieval was used. This thorough pass is offline and uses the persistent knowledge base, accepted source, system understanding, and44 completed experiments.

## Experimental History Review

- EXP036 remains the94.48% frontier at130.304 passes. EXP041-044 then produced four normal-exposure misses from auxiliary loss, adaptive pooling, gradient projection, and fixed spatial dispersion; preserve the exact pooled head, GAP, classifier, objective path, and raw gradients unless a distinct mechanism is strong.
- Generalization/boundary quality remains limiting while backward dominates counted time. The learner nearly interpolates the hard tail, but additions must retain at least127 passes.
- EXP010/011 showed low-resolution capacity near-signals; EXP026's image invariance was also a near-signal; EXP027's composition improved. This is affirmative evidence that invariance can unlock accepted capacity.
- Existing downsampling shortcut projections select one spatial phase with stride-2 1x1 convolutions. Fixed averaging before only the two transition projections is an untested way to reduce phase sensitivity without masking features, adding parameters, or altering the main learned branches.
- Early-only label smoothing remains executable but redundant with accepted mixup and unsupported by a calibration diagnosis. SiLU only in the pooled head preserves state/cost but changes the sole successful recent component and is adjacent tuning.
- The goal is not saturated: 94.48% is below CIFAR model capacity, and the current downsampling bypass has a concrete phase-selection property not examined by prior experiments.

## Collected Ideas

- **ResNet-D projection shortcuts** - average each 2x2 preactivated cell before the layer2/layer3 1x1 projection and set those projection strides to1. This targets phase-selective shortcut aliasing while retaining all four sites and all learned main branches.
- **Early-only epsilon0.05 label smoothing** - smooth both component CEs only while mixup is active, then preserve exact hard CE. It is nearly free and literature-shaped but compounds existing target softness.
- **SiLU pooled-head activation** - replace only the accepted 64-unit head ReLU with default SiLU to retain moderate negative contrasts. It adds no state but changes successful-head amplitude and geometry immediately.
- **Anti-aliased main and shortcut transitions** - blur both branches before decimation for stronger shift stability. It is broader than the shortcut-only test, increases spatial backward work, and risks suppressing fine CIFAR evidence.
- **One-time hard-boundary momentum reset** - clear velocity at the mixup transition. It is causal and free but supported for only a44-update transient.
- **Decoupled weight decay** - remove decay from momentum accumulation. It changes global optimizer geometry with ambiguous effective strength and weak local support.
- **Second pooled expert** - average a second cheap residual MLP correction. It may reduce representation variance but is adjacent capacity/scale tuning around EXP036.
- **Bias-free classifier simplification** - delete10 biases to reduce class-prior fitting. Prior classifier constraints were harmful and ten parameters are not a diagnosed limiter.
- **Low-rank covariance moonshot** - add cross-channel second-order information beyond closed marginal dispersion. Its rank/normalization are arbitrary and backward cost threatens exposure.

## Combinations

- **Projection averaging + accepted early RandAugment**: fixed phase aggregation can make intermediate representations respond more consistently to the image transformations already shown to unlock capacity. This is stronger than either alone because it aligns data-side invariance with architecture-side downsampling.
- **Early smoothing + exact hard tail**: uniform entropy is removed at the same boundary as mixup, preserving terminal class calibration. This is stronger than whole-run smoothing but still redundant early.
- **SiLU + accepted pooled residual path**: signed hidden contrasts remain subordinate to exact GAP through scale0.1. This is safer than changing backbone activations, but no local evidence shows ReLU dead units.

## Candidate Ideas

### One-Shot Early-Only Epsilon-0.05 Label Smoothing
**Summary**: Add PyTorch-uniform `label_smoothing=0.05` to both accepted mixup component CEs through exactly65%, then retain exact hard CE. Model/data/RNG/optimizer and parameters remain accepted.

**What it targets**: Potential early overconfidence while preserving the validated clean boundary-refinement tail.

**Reasoning**: Offline literature and temporal regularization support mild early smoothing, but accepted mixup already gives example-aware soft targets and epsilon lacks a local bracket. Full contract: `proposals/idea-01.md`.

**Sources**: `knowledge/papers/label-smoothing.md`; `mixup.md`; `time-matters-regularization.md`; EXP004/005/020/035/041; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Redundant softness can weaken class motion, and near-zero late loss does not diagnose early overconfidence. One miss closes immediate epsilon/timing/convention rescues.

### ResNet-D Projection Shortcuts at Both Downsampling Transitions
**Summary**: At `layer2[0]` and `layer3[0]`, apply parameterless `AvgPool2d(2,2)` to the preactivated shortcut input and change the existing bias-free1x1 projection stride from2 to1. Keep the stride1 layer1 projection, all identity shortcuts, both main stride2 branches, parameter names/bytes/order, and every other accepted component exact.

**What it targets**: The accepted shortcut selects only phase `(0,0)` of every2x2 cell before channel mixing. Averaging uses all four phases and distributes bypass gradients, potentially reducing transition aliasing under the accepted image-invariance learner.

**Reasoning**: EXP026/027 supply positive invariance-composition evidence, while the intervention adds no parameters and only two small pool kernels. It is deterministic aggregation, not feature masking or worker-side padding. Full contract: `proposals/idea-02.md`.

**Sources**: EXP010/011/026/027/032/036; `02-system-understanding.md`; accepted `train.py`; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Box averaging may blur class-relevant details and alter main/shortcut amplitude without post-addition BN. The startup function changes globally and small pool backward kernels may reduce exposure. A miss closes this exact two-transition rule and immediate pooling variants.

### Signed Smooth SiLU in the Accepted Pooled MLP
**Summary**: Replace exactly the state-free `nn.ReLU()` between accepted pooled-head matrices with default non-inplace `nn.SiLU()`. Preserve all weights, initialization, scale0.1, GAP, classifier, objective, and spatial work.

**What it targets**: Moderate negative hidden channel contrasts suppressed by ReLU in the narrow64-unit pooled bottleneck.

**Reasoning**: SiLU retains negative responses and smooth gradients at negligible cost, but it also attenuates positive activations and changes the only recent successful head from step one. Full contract: `proposals/idea-03.md`.

**Sources**: EXP036/041/042/044; accepted `train.py`; `proposals/idea-03.md`.

**Estimated Effort**: low

**Risk Assessment**: Signed W2 and the direct GAP path may already express negative corrections; negative W1 preactivation may correctly represent absence. A miss closes immediate smooth/signed activation variants.

## Review

The offline reviewer selected ResNet-D projection shortcuts at4/5 evidence and4/5 impact. I adopt its corrections: the treatment only replaces bypass phase selection, not network-wide aliasing; fixed averaging may suppress fine evidence and lower shortcut variance without post-addition BN; and complete-step timing overrides arithmetic. Preflight will report accepted/candidate shortcut RMS and main/shortcut RMS ratios without gain, transition, or kernel tuning. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Shortcut averaging is the only finalist with a distinct code-demonstrable information-flow mechanism and affirmative invariance-composition context. Label smoothing remains redundant and undiagnosed; SiLU violates EXP036's preserve-the-successful-head closure and bundles positive attenuation with signed evidence.

## Chosen Idea
**Selected**: ResNet-D Projection Shortcuts at Both Downsampling Transitions

**Why this idea**:
Replace only stride2 shortcut phase selection at layer2[0]/layer3[0] with fixed2x2 average aggregation before the unchanged pointwise channel maps. This retains every spatial site with deterministic quarter weight, preserves all learned parameter bytes/order/RNG, leaves main branches and the accepted objective/head exact, and plausibly composes architecture-side phase stability with the validated early image-invariance learner.

**Hypothesis**:
If single-phase projection shortcuts materially limit translation/phase stability, exact average-pool-before-projection at both downsampling transitions will retain at least127 projected and realized passes and raise fixed-seed `best_test_acc` from94.48% to at least94.58%. A normal-exposure miss closes this exact two-pool rule and immediate one-transition/kernel/order/scale variants without claiming all anti-aliasing methods fail.

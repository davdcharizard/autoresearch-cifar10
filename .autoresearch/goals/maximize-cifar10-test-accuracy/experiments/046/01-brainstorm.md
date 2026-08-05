# Brainstorm EXP-046
**Created**: 2026-07-27

## Web Search & Literature Review

- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can leave a durable generalization effect after removal, supporting preservation of the accepted exact clean tail.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild uniform targets can reduce overconfidence, but the source explicitly warns against stacking redundant soft-target methods without a calibration rationale.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): interpolation regularizes behavior between examples; feature-space interpolation is a distinct representation constraint only if it replaces rather than merely duplicates input mixing.
- **RandAugment** (`knowledge/papers/randaugment.md`): consistent fill semantics matter for geometric/color transforms; the accepted worker-safe policy already uses approximate CIFAR-mean RGB fill.

No network or new retrieval was used. This thorough pass uses only the persistent offline knowledge base, accepted source, system understanding, and 45 completed experiments.

## Experimental History Review

- The 94.48% frontier combines `(2,2,3)` WRN capacity, early worker-safe RandAugment, accepted input mixup, and a cheap pooled residual MLP. It retains 130.304 passes and nearly interpolates the hard tail, so generalization and boundary quality rather than memory, I/O, or wall time remain limiting.
- EXP041-045 produced five consecutive normal-exposure misses across auxiliary supervision, adaptive/statistical pooling, gradient projection, and fixed shortcut anti-aliasing. Preserve the exact pooled readout, raw gradients, sole refined-path CE, and learned downsampling paths unless a new mechanism supplies direct evidence.
- Mixup strength/duration/pair coefficient structure, classifier decay/radii, late averaging, tail LR, feature masking, spatial summary statistics, and shortcut averaging are locally closed. Immediate variants are post-result search rather than new hypotheses.
- The accepted crop pads raw PIL images with black before subtracting the CIFAR mean, whereas EarlyRandAugment fills transformed pixels with approximate dataset-mean RGB. This untested inconsistency creates deterministic high-contrast crop borders without providing useful image content and can be removed without state, GPU compute, or RNG changes.
- Feature-space interpolation, early target smoothing, pooled-head activation shape, transition-time optimizer state, and batch-normalization population structure remain formally untested. Only the first three have credible low-cost single-treatment definitions; the latter two lack a local diagnosis or non-arbitrary setting.
- The goal is not saturated: 94.48% is below the capacity of CIFAR classifiers, 98.9% of H20 allocation remains free, and recent misses span mechanisms rather than establishing an accuracy ceiling. New experiments still need near-zero spatial backward cost because backward consumes about 74% of each counted step.

## Collected Ideas

- **Mean-neutral crop padding** - set `RandomCrop`'s constant PIL fill to `[125,123,114]`, matching the accepted RandAugment fill and making padded pixels approximately zero after normalization. It removes a code-demonstrable border artifact, consumes no RNG/state/GPU work, and composes directly with the successful image-invariance learner.
- **Fixed stage-2 manifold mixup** - feed ordinary augmented images through the early backbone, interpolate a fixed intermediate feature tensor with the accepted batch-shared beta coefficient/permutation, and keep the same mixed labels through 65%. It targets representation linearity at lower spatial resolution without a second forward, though it abandons input-space convexity and changes where augmentation interactions occur.
- **Early-only epsilon-0.05 label smoothing** - apply PyTorch-uniform smoothing to both accepted mixup component CEs through exactly 65%, then retain byte-identical hard CE. It is nearly free and literature-shaped, but likely overlaps the already soft mixed target and has no measured calibration diagnosis.
- **SiLU inside only the pooled residual MLP** - replace the state-free hidden ReLU with default SiLU while preserving direct GAP, weights, scale 0.1, classifier, and backbone. Signed smooth hidden contrasts may improve the successful cheap representation, but positive attenuation also changes that accepted component from step one.
- **Ghost batch normalization** - reshape only batch statistics into fixed sub-batches while retaining affine parameters and inference buffers. It could add useful gradient noise at batch 256 without convolution cost, but convolutional BN already aggregates large spatial populations and changed running-stat semantics need an arbitrary ghost size.
- **One-time momentum reset at the clean boundary** - clear optimizer momentum buffers exactly when mixup ends, so clean labels establish a fresh descent direction rather than inheriting soft-target velocity. It is free and causal, but only a roughly 44-update transient separates mixup and worker-side RandAugment exhaustion and no logged instability supports it.
- **Mean-preserving random erasure** - apply a small early-only neutral-fill rectangle after crop/flip to encourage occlusion robustness without mixed spatial labels. It differs from failed CutMix, but probability/area/aspect settings are a large arbitrary search space and early feature masking has repeatedly hurt.
- **Low-rank covariance pooled expert moonshot** - project final channels to a tiny fixed rank and expose cross-channel second moments to an exact-neutral residual head. It could add information absent from both GAP and failed marginal dispersion, but rank, normalization, initialization, and backward cost are underdetermined.
- **Bias-free classifier simplification** - remove the ten classifier biases for a balanced dataset, preventing learned class-prior offsets at no compute cost. Prior classifier geometry constraints harmed and ten degrees of freedom are not a diagnosed limiter, so this is mainly a completeness control.

## Combinations

- **Mean-neutral crop padding + accepted early RandAugment**: both geometric pipelines use the same semantic neutral color, avoiding black-border examples immediately followed by mean-filled policy transforms. The cross is stronger than changing RandAugment because it fixes an inconsistency while preserving the already validated policy and RNG isolation.
- **Stage-2 manifold mixup + exact hard tail**: intermediate linearity regularizes the learned representation early, while the accepted last 35% restores ordinary examples and boundary refinement. The temporal cross is safer than full-run manifold mixing and more distinct than stacking two simultaneous interpolation sites.
- **Early label smoothing + exact hard tail**: a time-limited entropy prior avoids perturbing terminal calibration. This is stronger than full-run smoothing but still compounds accepted example-aware softness during the early window.
- **SiLU + accepted scaled pooled residual path**: signed corrections remain subordinate to exact GAP through scale 0.1. The combination is safer than changing backbone activations, though no local measurement demonstrates dead hidden units.

## Candidate Ideas

### CIFAR-Mean Constant Fill for Random Crops
**Summary**: Change only the accepted `RandomCrop(32,padding=4)` fill from raw black to the fixed quantized CIFAR mean `(125,123,114)`. Preserve constant padding, crop/flip draws, transform order, worker-isolated RandAugment, batch-shared mixup, model, optimizer, and every schedule. The candidate pad becomes approximately zero after accepted mean subtraction instead of roughly `(-0.49,-0.48,-0.45)`.

**What it targets**: A frequent artificial input boundary under the generalization bottleneck. Uniform crop offsets touch padding in 80/81 crops and expose about 13.4% synthetic pixels on average, while the accepted RandAugment already uses the proposed mean-colored fill.

**Reasoning**: This is the only candidate with a code-demonstrable mismatch in a successful mechanism and no GPU/state/RNG cost. It is distinct from EXP032 reflection padding because it retains the accepted constant PIL path and changes only boundary color, not geometry or implementation. The counter-hypothesis is that black borders supply useful occlusion/contrast regularization.

**Sources**: `knowledge/papers/randaugment.md`; EXP026/027/032/036; accepted `train.py`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: Mean fill is still textureless and can reduce useful augmentation diversity; active RandAugment can spread or recolor border pixels. Persistent-worker timing must be qualified because the prior reflection treatment was unstable, even though tuple fill stays on the accepted constant-padding path.

### Early Pooled-Feature Manifold Mixup
**Summary**: During exactly the first 65%, replace input-pixel interpolation with one post-GAP/pre-pooled-MLP interpolation of the 128-dimensional accepted features, using the exact accepted batch-shared Beta draw, permutation, paired CE, and cutoff. Each spatial backbone sees an ordinary independently augmented image; the final 35% remains the exact clean path.

**What it targets**: Representation and decision-boundary linearity without added spatial backward cost. It trains the successful nonlinear pooled head directly on between-example feature segments while keeping spatial BatchNorm statistics on real augmented inputs.

**Reasoning**: EXP002 validates the coefficient/target/time prior, EXP027 validates early invariance plus capacity, and EXP036 validates a nonlinear pooled decision representation. Fixed post-GAP placement is cheap and interpretable, but this is a replacement of strongly successful input mixup rather than an orthogonal addition and has no saved manifold-mixup paper.

**Sources**: `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP002/004/005/015/020/027/035/036/041; `proposals/idea-02.md`.

**Estimated Effort**: high

**Risk Assessment**: Late feature interpolation may be too easy or off-manifold, removes useful low-level convex-image invariance, and bundles clean spatial BN statistics with the placement change. One miss closes immediate placement and compound-mix rescues.

### Early-Only Epsilon-0.05 Uniform Label Smoothing
**Summary**: Add PyTorch-uniform `label_smoothing=0.05` to both accepted component CEs only while mixup is active, then retain literal accepted hard CE. Model, inputs, mixup draws, state, RNG, optimizer, schedule, and inference remain unchanged.

**What it targets**: Possible harmful early class overconfidence while preserving the validated exact-label boundary-refinement tail. The uniform prior distributes mass beyond the one or two classes represented by mixup.

**Reasoning**: Saved literature supports mild smoothing and early regularization, and the implementation is nearly free. Local evidence is weak: accepted mixup already supplies soft targets, alpha/duration are bracketed, near-zero late loss does not diagnose early calibration, and epsilon 0.05 is a prospective convention rather than a local optimum.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP004/005/020/035/041/045; `proposals/idea-03.md`.

**Estimated Effort**: low

**Risk Assessment**: Redundant target entropy may blunt useful class motion and weaken the critical early period. A valid miss closes immediate epsilon, convention, ramp, placement, and confidence-penalty rescues without a calibration diagnosis.

## Review

The offline critic selected CIFAR-mean crop fill at 4/5 evidence and 3/5 impact. I adopt its significant cautions: the fill mismatch is concrete but does not prove black borders are harmful; mean fill can remove useful occlusion/contrast and is still textureless; RandAugment may spread the altered pixels; and loader stability, not source-path inference, decides feasibility. The treatment remains fixed `(125,123,114)` constant padding with accepted crop geometry/RNG and a strict no-alternate-fill closure. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Mean fill is the only finalist that removes a quantified source-level artifact while preserving every validated learning mechanism and all stochastic decisions. Pooled-feature mixup has larger theoretical upside but deletes EXP002's strongest positive regularizer and bundles BN-distribution with representation mixing. Label smoothing is nearly free but lacks a calibration diagnosis and redundantly softens a bracketed mixup target.

## Chosen Idea
**Selected**: CIFAR-Mean Constant Fill for Random Crops

**Why this idea**:
Change exactly one `RandomCrop` fill argument so synthetic border pixels are approximately neutral after the accepted mean subtraction and consistent with the already validated RandAugment fill. This affects about 13.4% of pixels on average in 80/81 crop offsets, yet retains constant PIL padding, transform order, crop/flip/RandAugment decisions, GPU graph, state, optimizer, mixup, hard tail, and evaluator. It is a new input-distribution mechanism rather than a rescue of failed reflection padding.

**Hypothesis**:
If the frequent high-magnitude normalized black crop border materially worsens train/test boundary-statistic alignment, replacing only that border with fixed quantized CIFAR-mean RGB will preserve at least 127 realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. A valid normal-exposure miss closes this exact always-on fill and immediate color/mode/width/schedule variants without claiming all boundary augmentation fails.

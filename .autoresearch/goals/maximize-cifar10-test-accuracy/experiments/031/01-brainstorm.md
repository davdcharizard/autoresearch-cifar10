# Brainstorm EXP-031
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): wider, moderately deep residual models can spend CIFAR compute more effectively than thin deep networks, but local tests show allocation and tensor shape matter more than parameter count alone.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): Beta interpolation is the strongest low-cost local generalizer; alpha changes the mixture endpoint distribution without adding GPU work.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can preserve its effect after removal, supporting retention of the accepted 65%/35% split.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): carefully selected late windows can improve generalization cheaply, but EXP013's local whole-state EMA result is a strong contrary prior.

No network source was consulted. This offline thorough pass uses the accumulated local knowledge base and complete experiment history.

## Experimental History Review

- EXP027 is the 94.32% baseline: the `(2,2,3)` model and early `N=1,M=5` RandAugment compose positively even though each missed alone. This interaction, alpha-0.2 batch-shared mixup through 65%, full FP32 gradients, the 0.002 LR floor, continuous matrix decay, and the clean tail are protected.
- Compute, not memory or I/O, limits exposure. Backward is about 74% of step time, yet EXP028 proved that saving 35.9% of hard-step time by freezing high-resolution layers loses essential adaptation. EXP029 showed batch 128 cannot preserve the full-model image/update regime.
- Regularization changes form a strongly negative neighborhood: stronger alpha, shorter/longer mixup windows, CutMix, per-example coefficients, broad dropout, targeted drop-path, and late SAM all regressed. Alpha 0.1 remains the only clean unmeasured batch-shared strength bracket.
- Low-resolution capacity has the best positive history, but exact width/depth increments missed alone and attention variants were either weaker or outside their protected exposure envelope. Any new capacity idea must preserve the accepted depth/RandAugment composition and pass direct shape-specific timing.
- Untested gaps include larger batches that may exploit the H20, tensor memory format, crop padding geometry, weaker batch-shared mixup, normalization/optimizer conditioning, and architecture changes that preserve full residual participation.

## Collected Ideas

- **Batch 512 with linear LR scaling** - double the examples per optimizer step and set `LR=0.4`, `MIN_LR=0.004`, retaining the time schedule and all accepted regularization. It attacks the 74%-backward compute bottleneck by seeking higher H20 utilization and far more image exposure, with the risk that fewer optimizer decisions and large-batch generalization outweigh the gain.
- **Channels-last FP32 execution** - convert the model and each input batch to channels-last memory format while retaining the exact architecture and FP32 SGD. This directly targets convolution throughput without dropping gradients or changing the objective; H20 tensor-layout kernels may improve exposure, though fixed-seed convolution numerics will define a different trajectory and small CIFAR tensors may not accelerate.
- **Weaker alpha-0.1 batch-shared mixup** - change only the Beta concentration while preserving batch coherence and the 65% cutoff. It tests whether the deeper-plus-RandAugment learner needs more endpoint-heavy mixtures after both stronger and additive feature regularization failed, but all neighboring mixup perturbations are negative.
- **Reflection-padded random crops** - change `RandomCrop` from zero padding to reflection padding, preserving crop size, flip, RandAugment, worker RNG isolation, and GPU training. It targets boundary artifacts at effectively no counted cost; the downside is a new worker-side stochastic image distribution with little direct local evidence.
- **Classical momentum instead of Nesterov** - retain momentum 0.9 but disable Nesterov, testing whether less anticipatory terminal motion improves the best/final stability of the deeper learner. It is compute-neutral and orthogonal, but canonical WRN training supports momentum SGD broadly and there is no local overshoot diagnosis.
- **Reduced full-run weight decay** - lower matrix decay from `5e-4` to `3e-4` throughout, hypothesizing that the composed deeper/augmented model is more regularized than the learner on which the value was established. EXP007's late removal and the current generalization gap oppose this, so it is a weak bracket rather than a lead.
- **Late parameter-only uniform averaging** - average only trainable parameters over a short predetermined final window while leaving live BN buffers, addressing EMA lag and approximate BN-state averaging from EXP013. It is cheap and distinct in mechanics, but remains post hoc smoothing in a neighborhood where lower loss did not move enough top-1 decisions.
- **Compute-efficient full-participation final block** - replace the added block's two dense 3x3 convolutions with a structured grouped-plus-pointwise transformation sized to retain receptive field and channel mixing. It attacks backward cost while honoring EXP030's evidence that the branch must participate on every example, but represents a high-risk architecture change with no local equivalence evidence.
- **Moonshot: channels-last plus selective stage-3 width** - spend any measured channels-last speedup on wider low-resolution channels rather than extra passes. This combines the only consistently promising capacity location with a format-level efficiency lever; joint numerical and architectural changes make attribution and exposure risk substantial.

## Combinations

- **Batch 512 + channels-last**: larger batches may make layout-optimized H20 kernels more efficient than either change alone, potentially converting memory headroom into substantial image exposure. The combination is only defensible after each lever's isolated timing because it compounds altered batch optimization with altered convolution numerics.
- **Channels-last + selective stage-3 width**: a verified format speedup could pay for extra low-resolution capacity while holding the accepted exposure regime. This is stronger than format alone if extra passes saturate, and stronger than width alone if layout recovers its cost, but it requires a later composition experiment rather than an unisolated first run.
- **Reflection padding + alpha 0.1**: cleaner crop boundaries and more endpoint-heavy interpolation could reduce two different input-space distortions while keeping the hard tail intact. The cross might outperform either if the accepted composed learner is slightly over-regularized, but current evidence does not justify changing both axes in one score.
- **Short uniform averaging + channels-last**: a format speedup could absorb averaging-copy overhead, while a shorter uniform window could avoid EXP013's long-horizon EMA lag. The two mechanisms are largely independent, so isolated results should precede any composition.

## Candidate Ideas

### Batch 512 With a Fully Scaled LR Curve
**Summary**: Change only `BATCH_SIZE=512`, `LR=0.4`, `MIN_LR=0.004`, and the image-equivalent safety cap `MAX_STEPS=32000`. Preserve the accepted model, FP32 Nesterov SGD, time schedule, early alpha-0.2 batch-shared mixup/RandAugment, full gradients, and evaluator. Advance only if balanced complete-body timing shows at least 1.10x accepted image rate, projecting at least 146.308 passes.

**What it targets**: The measured 98% forward/backward share and enormous unused H20 memory. Larger batches may improve convolution utilization and process materially more images, while doubled LR approximately preserves cumulative first-order displacement per example.

**Reasoning**: This is the symmetric unexplored counterpart to infeasible batch 128, but the mechanism is not exposure alone: it jointly tests larger, lower-noise decisions and better hardware utilization. The strict 1.10x gate is necessary because BF16, fixed-MAC redistribution, and freezing all proved that exposure gains cannot compensate for a damaged learner.

**Sources**: `proposals/idea-01.md`; `02-system-understanding.md`; EXP009, EXP016, EXP027-EXP029; `project-notes/project-insights.md`.

**Estimated Effort**: medium

**Risk Assessment**: Half as many optimizer/BN/mixup decisions per image, fewer stochastic gradients, a changed dropped tail, and linearly doubled LR may degrade generalization or late refinement. The score defines an indivisible batch/LR/BN/RNG operating point and cannot justify adjacent batch or LR tuning.

### End-to-End FP32 Channels-Last Training
**Summary**: Convert the accepted model and every training input to channels-last memory format while preserving FP32 storage, accepted cuDNN/TF32 flags, optimizer, stochastic stream, model logic, and data policy. Add an idempotent forward-entry conversion so the frozen evaluator's NCHW inputs use the same candidate layout. Require at least 1.02x exact fixed-time throughput and 135.67 projected passes before one score.

**What it targets**: The convolution-dominated forward/backward bottleneck without removing gradients, reducing precision, or altering capacity. It attempts to turn an H20 layout preference into more accepted-style full-model exposure.

**Reasoning**: H2D and non-model work are under 2%, so only convolution-level acceleration can materially help. This is safer than BF16 in storage precision and safer than freezing in adaptation, but it is still a distinct deterministic numerical path; balanced full-body timing and exact candidate replay are required.

**Sources**: `proposals/idea-02.md`; `02-system-understanding.md`; EXP009, EXP016, EXP027-EXP029; local PyTorch/CUDA environment.

**Estimated Effort**: medium

**Risk Assessment**: Small 32x32 tensors may make layout conversion neutral or slower, different deterministic cuDNN kernels change low-bit optimization numerics, and extra exposure may not improve a generalization-limited learner. A stable timing miss or one valid score closes exact channels-last without layout or precision rescue variants.

### Weaker Alpha-0.1 Batch-Shared Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2 ->0.1`, retaining a single scalar coefficient shared across batch 256, the 65% cutoff, accepted deeper-plus-RandAugment model, full FP32 training, schedule, worker policy, seed 42, and evaluator. Beta(0.1,0.1) makes early mixtures substantially more endpoint-heavy while preserving symmetry and mean 0.5.

**What it targets**: Boundary quality at near-zero tail loss, testing whether the accepted composed learner receives slightly too much interpolation softness without adding a new regularizer or masking useful computation.

**Reasoning**: Alpha 0.4 closed the stronger side, and EXP030 says to preserve the added block's full residual contribution. Alpha 0.1 is the only unmeasured one-axis batch-shared strength bracket and has near-zero cost, but the accepted run shows no direct over-regularization symptom and every neighboring mixup perturbation regressed.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/mixup.md`; EXP004, EXP005, EXP015, EXP020, EXP027, EXP030.

**Estimated Effort**: low

**Risk Assessment**: Likely under-regularization and an alpha-dependent CUDA Beta/permutation trajectory may lose more than the required margin. A valid normal-exposure miss closes immediate mixup-strength tuning, including all adjacent alphas, seeds, coefficient mappings, and cutoff combinations.

## Review

The reviewer selected channels-last at 3/5 evidence and 2.5/5 impact. I adopt the decision-changing corrections: the proposal now uses reciprocal time-phase rates rather than an arithmetic weighted step time; it claims preservation only of logical state and equal-prefix stochastic decisions, not accepted numerical trajectory; and final accuracy/loss remain separate corroboration signals. The independent accepted oracle, bounded cross-layout checks, exact candidate replay, evaluator NCHW bridge, FP32 audit, >=1.02 speed gate, one score, and no-rescue closure remain mandatory. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Channels-last is the cleanest systems extension because it targets measured convolution cost while retaining full gradients, capacity, FP32 storage, and equal-prefix stochastic decisions. Batch 512 has more upside but changes the Nesterov/BN/mixup decision regime without positive local evidence; alpha 0.1 has clean attribution but the strongest negative neighborhood.

## Chosen Idea
**Selected**: End-to-End FP32 Channels-Last Training

**Why this idea**:
The accepted learner is compute-bound in convolutional forward/backward, and channels-last is the finalist least likely to damage its established depth/RandAugment/full-gradient mechanism. It receives a score only after independent semantic checks and a corrected fixed-time timing gate demonstrate a material H20 opportunity.

**Hypothesis**:
If end-to-end channels-last FP32 execution preserves the accepted logical learner and equal-step stochastic decisions while improving exact fixed-time complete-body rate by at least 1.02x, then it will project and realize at least 135.67 passes and raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%. Final accuracy at least 94.32% and loss relative to 0.2523 will be reported as mechanism corroboration only. A stable timing miss closes the format without scoring; a valid scored miss closes it without layout, precision, batch, or cuDNN-flag rescue variants.

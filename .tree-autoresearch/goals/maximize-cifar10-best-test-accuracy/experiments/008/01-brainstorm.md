# Brainstorm EXP-008
**Created**: 2026-08-05

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (`experiments/008/papers/eca-net.md`)
  Local one-dimensional channel interaction adds negligible parameters and published compute while improving residual backbones, but the evidence is mostly ImageNet-scale and standard sigmoid gating is not identity-preserving.
- **RandAugment: Practical Automated Data Augmentation with a Reduced Search Space** (`experiments/008/papers/randaugment.md`)
  Direct CIFAR WRN results support a large augmentation prior, but operation count and magnitude were validation-selected and may overlap the existing front-loaded CutMix.
- **DropBlock: A Regularization Method for Convolutional Networks** (`experiments/008/papers/dropblock.md`)
  Structured spatial dropout can outperform independent dropout and benefits from scheduled strength, but EXP-004 already uses residual drop path and another regularizer risks mechanism overlap.
- **How to Scale Your EMA** (`knowledge/papers/how-to-scale-your-ema.md`)
  EMA decay must be expressed through an effective horizon when update count is wall-clock-dependent.
- **Averaging Weights Leads to Wider Optima and Better Generalization** (`knowledge/papers/stochastic-weight-averaging.md`)
  Late trajectory averaging improves CIFAR residual models cheaply when the trajectory remains diverse and BatchNorm state is handled faithfully.
- **Residual stochastic regularization** (`knowledge/papers/shakedrop.md`, `knowledge/papers/time-matters-regularization.md`)
  Residual-branch noise can improve CIFAR models, but phase and mechanism matter more than simply increasing regularization strength.

## Experimental History Review

- The accepted lineage is BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23% -> EXP-004 95.40%. A wider preactivation model, front-loaded CutMix, and clean-tail period-two SAM are independently validated and should remain intact.
- EXP-003 showed 0.14-0.29-point selected gains can reverse on confirmation. EXP-006/007 tail variation reinforces a practical effect prior near 0.3 points for a worthwhile one-run candidate.
- EXP-005 preserved steps but halved new-image identities through overlapping batches and lost 0.12 points. EXP-006 substituted manifold mixup for validated CutMix and remained flat. EXP-008 should be additive and preserve the independent-image stream and full validated doses.
- EXP-007's published-default ASAM package preserved compute but reached 95.34%. It changed actual Euclidean perturbation magnitude by about 9x, so it does not support further narrow optimizer tuning from the observed score.
- EXP-004 uses only 1,190.5 MiB on a 97,871 MiB H20 and completes 25,560 steps. The active limiter is detectable generalization gain from an already strong trajectory, not memory; small single-pass modules can exploit headroom if they preserve at least about 25,000 steps (`02-system-understanding.md`).
- Untried high-level gaps include additive channel recalibration, horizon-defined weight averaging, stronger input policies, spatial activation regularization, anti-aliased downsampling, and architecture reallocation.

## Collected Ideas

- **Identity-preserving late-stage ECA** - Add local channel recalibration to only the 128- and 256-channel residual branches. A zero-initialized short 1D kernel and `2*sigmoid` gate make every new gate exactly one at construction while retaining nonzero gate gradients; this targets representation selectivity at negligible parameter and arithmetic cost.
- **Horizon-matched full-state EMA** - Maintain shadow parameters and BatchNorm buffers after every parent update with decay derived from a fixed effective sample/time horizon. Evaluate the shadow state once per epoch without adding validation calls, targeting the observed late-checkpoint variance while preserving all training inputs and optimizer work.
- **Fixed CIFAR RandAugment policy** - Insert a preregistered torchvision policy between crop/flip and tensor conversion, with no metric-driven search. The direct CIFAR WRN evidence offers a large prior, but CPU throughput and overlap with CutMix must be gated before launch.
- **Scheduled residual DropBlock** - Apply contiguous spatial masks to residual activations during the early regularized phase and decay them away in the clean tail. It targets correlated feature co-adaptation without another model pass, but may duplicate current per-example drop path.
- **Late-stage ShakeDrop upgrade** - Replace Bernoulli residual dropping with signed/random-amplitude residual perturbations in the later blocks using dedicated RNG streams. This imports stronger CIFAR residual regularization, but it is a substitution for a working mechanism and requires several coupled schedule choices.
- **Anti-aliased transition downsampling** - Move stage-transition stride into a fixed low-pass downsample and keep the learned convolution stride one. It targets aliasing and shift sensitivity rather than regularization, though extra memory traffic may cost optimizer steps and its CIFAR effect is uncertain at 32x32 resolution.
- **Parameter-neutral stage reallocation** - Trade some early 64-channel capacity for an additional 128/256-channel block or wider late features at roughly fixed parameters. This targets class-selective high-level representation, but changes depth, stochastic-depth layout, throughput, and initialization together.
- **Auxiliary intermediate classifier** - Attach a small head after the 128-channel stage during only the first 75% and train it with the same CutMix labels. This supplies deeper supervision without changing inference, but adds a loss weight and may compete with the primary head rather than improving its boundary.
- **Moonshot: multi-resolution residual branch** - Replace one late 3x3 branch with parallel standard and dilated/depthwise context paths fused by an identity-initialized gate. It could expand receptive-field diversity but is weakly matched to tiny CIFAR images and carries a larger implementation surface.

## Combinations

- **ECA + horizon-matched EMA**: ECA could improve the representation while EMA reduces late-iterate variance in its new gates. The cross attacks both class-boundary quality and checkpoint stability, but combining them would prevent clean attribution and is therefore better reserved for two separately validated parents.
- **Stage reallocation + ECA**: Moving capacity toward late stages gives channel attention richer high-level features to select. The combination may outperform either alone, but simultaneous topology and attention changes would make a first test too broad.
- **RandAugment + clean-tail EMA**: Stronger early view diversity can widen the learned basin, while late averaging consolidates clean iterates. This has a coherent phase story but risks excessive regularization on top of CutMix and introduces two new packages.

## Candidate Ideas

### Horizon-Matched Full-State EMA
**Summary**: Deep-copy a detached full-state shadow at 75% progress, then update parameters and floating BatchNorm buffers from every restored post-update online state with a 15-second charged-time constant. Copy integer buffers exactly and switch the existing once-per-epoch evaluator from online to EMA at initialization; never evaluate both. Parent optimization remains unchanged apart from charged fused state maintenance (`proposals/idea-02.md`).

**What it targets**: Late-checkpoint variance and abundant memory headroom, without removing any images, CutMix batches, optimizer updates, or SAM pulses (`02-system-understanding.md`).

**Reasoning**: CIFAR residual models benefit from weight averaging, and horizon-defined EMA avoids importing a decay that changes meaning with wall-clock throughput. Full-state evaluation avoids an uncharged BatchNorm recalibration pass. However, EXP-004 already finished at its best checkpoint and the strongly decayed tail may lack enough trajectory diversity for a 0.30-point gain.

**Sources**: `knowledge/papers/how-to-scale-your-ema.md`; `knowledge/papers/stochastic-weight-averaging.md`; `experiments/008/proposals/idea-02.md`; EXP-004/006/007 analyses.

**Estimated Effort**: medium

**Risk Assessment**: Expected gain is only +0.05 to +0.20; averaging BatchNorm state is approximate; the fixed switch hides late online checkpoints; state update ordering around SAM is correctness-sensitive. It is a low-cost variance intervention but a weak match to the effect-size requirement.

### Identity-Preserving Residual ECA
**Summary**: Add a short ECA channel gate after `conv2` and before drop path in each of the six residual branches. Use independent raw zero-valued kernels `[3,3,5,5,5,5]` and `2*sigmoid(logits)`, making every gate exactly one at initialization without consuming RNG while retaining nonzero first-step gate gradients. All EXP-004 data, CutMix, drop path, SAM, optimizer, and evaluation behavior stay intact. The fixed package adds 26 parameters and is described in `proposals/idea-01.md`.

**What it targets**: The measured limiter is class-boundary generalization from a strong but compact representation, while memory is only 1.2% utilized. Per-image cross-channel selection adds representational flexibility without substituting validated exposure (`02-system-understanding.md`).

**Reasoning**: ECA reports clear residual-backbone gains with negligible parameters and FLOPs. The scale-corrected zero initialization avoids halving every residual branch, preserves all shared parent initialization and RNG, and lets attention learn immediately. Six small gates could still be launch-bound, so actual parent/candidate BF16 ordinary and SAM latency must project a sufficiently large step horizon before the run.

**Sources**: `experiments/008/papers/eca-net.md`; `experiments/008/proposals/idea-01.md`; `experiments/004/04-analysis.md`; `02-system-understanding.md`.

**Estimated Effort**: medium

**Risk Assessment**: Channel adjacency may have little meaning in a shallow CIFAR WRN; the `(0,2)` identity-centered gate is an evidence-backed adaptation rather than the paper's exact `(0,1)` gate; CutMix descriptors may diffuse attention; tiny operations may cost more latency than their FLOPs imply. A valid sub-95.50 result rejects this fixed package, and 95.50-95.69 remains below the 0.30-point mechanism target.

### Fixed Torchvision RandAugment Before CutMix
**Summary**: Add the installed torchvision `RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, nearest, fill=0)` after crop/flip and before tensor conversion for every training image. Wrap it in a per-worker private torch RNG state so parent crop/flip, shuffle, CutMix, and CUDA streams remain isolated. The full CutMix and SAM doses remain; no policy scalar, phase, or seed is searched (`proposals/idea-03.md`).

**What it targets**: Generalization through stronger per-image photometric and geometric invariance while preserving all independent identities and the full validated training path. It attacks input diversity rather than optimizer geometry (`02-system-understanding.md`).

**Reasoning**: RandAugment has direct CIFAR WRN evidence with gains large enough to clear this protocol's noise. The fixed torchvision defaults avoid a test-driven search and add no GPU model work. Loader throughput must be audited because the current charged timer excludes batch-yield wait, and the experiment evaluates a combined RandAugment+CutMix+SAM package rather than RandAugment in isolation.

**Sources**: `experiments/008/papers/randaugment.md`; `experiments/008/proposals/idea-03.md`; `experiments/004/04-analysis.md`; EXP-005/006 learnings.

**Estimated Effort**: medium

**Risk Assessment**: Strong crop/flip, RandAugment, CutMix, drop path, and SAM can over-regularize; the default policy is not matched to this WRN; PIL operations may inflate total runtime or starve the GPU; private worker RNG handling is delicate. No alternate magnitude, operation count, phase, or interpolation may be chosen after the result.

## Review

Claude selected RandAugment on evidence (7/10) and impact (8/10), ahead of ECA (6/10, 5/10) and EMA (2/10, 2/10). Two critiques are adopted as essential: the package is changed from generic torchvision defaults to the paper's low-capacity CIFAR WRN operating point, and RandAugment is disabled with CutMix at progress 0.75 so the validated clean SAM tail remains unchanged. Loader feasibility will measure the complete RNG-isolation wrapper and both charged and total time.

Claude's ECA descriptor and weight-decay suggestions are retained as future design notes, not combined with this experiment. Its EMA rejection is accepted because the proposed horizon, measured parent behavior, and self-estimated effect all point below the current resolution floor. Full feedback is in `01-idea-review.md`.

## Idea Evaluation

The randomized verdict is adopted. RandAugment is the only finalist with direct CIFAR WRN evidence and a plausible effect above 0.30 points. Planning found that the paper's WRN-28-2 magnitude 2 is not semantically portable to torchvision 0.24.1: on 32x32 inputs it quantizes translation and posterization to exact no-ops. Before any accuracy run, the fixed implementation was therefore corrected to `num_ops=1, magnitude=5`, the lowest mild audited torchvision bin with nonzero two-pixel translations and seven-bit posterization. This is an implementation-semantics correction, not an accuracy-driven search. The policy applies only while `progress < 0.75`, matching the lineage's regularized phase; crop/flip, CutMix, independent identities, and the complete clean SAM tail remain.

## Chosen Idea
**Selected**: Implementation-Audited Mild RandAugment in the Regularized Phase

**Why this idea**:
This candidate has the strongest task- and architecture-matched effect evidence and changes input invariance rather than another narrow optimizer geometry. The fixed mild torchvision `N=1, M=5` package is audited to produce nonzero operation magnitudes, while the 0.75 cutoff preserves the exact data regime in which late SAM was validated. A private per-worker generator state prevents the transform from shifting crop/flip, shuffle, CutMix, or CUDA randomness. No operation count, magnitude, phase, interpolation, or seed will be selected after implementation review or from the result.

**Hypothesis**:
Adding one mild published CIFAR-WRN RandAugment operation to each training image only during the first 75% of charged time will preserve at least 25,000 optimizer steps and improve EXP-004's `best_test_acc` from 95.40% to at least 95.70% in one fixed-seed physical-GPU-0 run. The formal necessary threshold remains 95.50%; a valid result below 95.70% falsifies the preregistered mechanism-sized expectation, and no scalar or phase retry is allowed.

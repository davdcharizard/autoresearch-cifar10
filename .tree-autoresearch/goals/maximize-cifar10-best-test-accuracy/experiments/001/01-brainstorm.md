# Brainstorm EXP-001
**Created**: 2026-08-05

## Web Search & Literature Review

- **Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness** (`papers/regmixup.md`)
  Clean-example cross-entropy plus Mixup can outperform using Mixup as the sole objective on CIFAR-10/100, but the extra clean forward pass is costly under a fixed wall-clock budget.
- **Time Matters in Regularizing Deep Networks** (`papers/time-matters-regularization.md`)
  Augmentation, weight decay, and Mixup matter most during an early critical period, supporting strong early regularization followed by a less constrained late optimization phase.
- **StackMix: A Complementary Mix Algorithm** (`papers/stackmix.md`)
  Complementary mixed-sample augmentations can stack, including a reported 0.5-point CIFAR-10 gain over CutMix, though altered input geometry and throughput make it a later exploratory option.
- **A Unified Analysis of Mixed Sample Data Augmentation** (`papers/mixed-sample-analysis.md`)
  Mixup and CutMix impose distinct, efficient input-gradient regularizers; hybrids can capture both without significant arithmetic overhead.
- **ShakeDrop Regularization for Deep Residual Learning** (`papers/shakedrop.md`)
  Stochastic residual-branch scaling can regularize ResNet-family models, but stability and short-budget convergence are material risks.

## Experimental History Review

- First experiment under this goal - no prior experiment history, successful siblings, or failed children exist.
- The BASE node is the unchanged ResNet-20 at 91.51% (`tree.sh log ... BASE`), with 269,722 parameters, 330.1 MiB peak VRAM, 34,435 steps, and 89 epochs in 300.0 training seconds.
- The primary limiter is an outdated budget mismatch: the step scheduler holds LR at 0.1 until step 32,000, so nearly the entire run trains at the initial LR and only about 2,435 steps see the first decay; the second milestone is never reached.
- The hardware is heavily underused: 330.1 MiB peak VRAM on a 97,871 MiB H20 leaves room for a larger batch and substantially higher-capacity model, provided throughput remains high enough to anneal and converge within five minutes.
- The baseline has only crop/flip augmentation, no soft-target regularization, no model averaging, and no wall-clock-aware optimization. These are the main untried gaps.

## Collected Ideas

- **Wall-clock cosine with warmup** - replace unreachable fixed-step milestones with a learning-rate schedule computed from elapsed training fraction, including a short warmup and cosine decay to a small final LR. This directly targets the schedule/budget mismatch and guarantees a full optimization arc regardless of model throughput.
- **H20 utilization through width and batch size** - widen the residual stages, increase batch size, and use mixed precision or channels-last only where stable. The baseline leaves over 99% of VRAM unused, so a moderate 2-4x width increase could add representation capacity while a larger batch improves accelerator utilization.
- **Pre-activation Wide ResNet** - replace the narrow post-activation ResNet-20 with a pre-activation wide residual network whose depth/width is chosen for the five-minute envelope. This attacks model under-capacity and improves residual optimization, but it risks fewer effective epochs.
- **Efficient mixed-sample regularization** - apply one-pass Mixup/CutMix with soft-target cross-entropy, optionally switching between the two, rather than doubling work for RegMixup. It targets overfitting and limited data diversity using literature-backed regularization with negligible arithmetic overhead.
- **Cutout or lightweight RandAugment** - add cheap image-space corruption after crop/flip, potentially front-loaded during the early critical period. This broadens invariances without changing the loss, but CPU transform overhead and overly strong corruption could slow or underfit the short run.
- **EMA or late SWA** - maintain an exponential moving average of weights and evaluate the averaged model once per epoch, or average only late checkpoints. This targets noisy late optimization and can improve generalization at low compute cost, though batch-normalization buffers must be handled correctly.
- **Fixed input rescaling** - apply conventional CIFAR per-channel standard-deviation scaling inside the model so training and the frozen evaluator receive identical normalization without modifying `prepare.py`. This corrects the baseline's unit standard deviation cheaply, but batch normalization may already absorb much of the scale.
- **ShakeDrop/PyramidNet moonshot** - gradually increase channels across residual blocks and apply depth-dependent stochastic residual scaling. It combines capacity-efficient representation growth with strong regularization, but has the highest implementation and convergence risk for a first short-budget experiment.

## Combinations

- **Wall-clock cosine + wider high-throughput model**: schedule progress follows elapsed compute rather than assumed steps, so increased model cost cannot strand the run at its initial LR. Capacity and optimization alignment together are more likely to beat either a wider model with the stale schedule or a schedule-only change on the tiny baseline.
- **Mixed-sample regularization + EMA**: soft-target augmentation improves example diversity while EMA stabilizes the noisier parameter trajectory it creates. The combination addresses both bias and variance at low extra cost, whereas either alone leaves the other failure mode untouched.
- **Pre-activation width + early regularization + cosine**: a higher-capacity residual model needs stronger early regularization and reliable late annealing to convert capacity into test accuracy. The combination is more coherent than dropping a large model into the baseline recipe unchanged.
- **Input rescaling + mixed-sample augmentation**: conventional channel scaling improves conditioning while Mixup/CutMix improves generalization. Both are cheap and orthogonal, though the expected contribution from rescaling is uncertain because of batch normalization.

## Candidate Ideas

### Time-Normalized Wide-Batch BF16 ResNet-20
**Summary**: Replace the unreachable step milestones with a 5%-warmup, wall-clock cosine schedule, widen every ResNet stage by 2x, raise batch size from 128 to 512, and use BF16 autocast plus channels-last tensors. The schedule is indexed by accumulated charged training time, so the wider model still completes a deliberate high-to-low LR trajectory. The architecture remains recognizably the baseline ResNet and keeps crop/flip, SGD momentum, and weight decay fixed. See `proposals/idea-01.md`.

**What it targets**: The BASE schedule spends about 93% of its 300-second budget at LR 0.1 and never reaches the second decay, while the 269,722-parameter network uses only 330.1 MiB of a 97,871 MiB H20. This package repairs the optimization horizon and converts unused accelerator capacity into model capacity.

**Reasoning**: Wall-clock normalization directly matches the benchmark resource. Width 2 adds roughly 4x parameters, while larger BF16/channel-last kernels should recover much of the compute cost. The proposal deliberately defers `torch.compile` and mixed-sample augmentation to keep its execution risks bounded.

**Sources**: `proposals/idea-01.md`; `papers/time-matters-regularization.md`; BASE summary in `01-brainstorm.md`.

**Estimated Effort**: medium

**Risk Assessment**: Batch 512 may reduce update diversity, width may cost more throughput than BF16 recovers, and BF16/channel-last may be neutral on tiny convolutions. Warmup, conservative peak LR 0.20, and explicit throughput diagnostics mitigate but do not eliminate these risks.

### Time-Aware Pre-Activation WRN-16-4 with Restrained Stochastic Depth
**Summary**: Replace the narrow post-activation ResNet-20 with a six-block pre-activation WRN-16-4 (about 2.75M parameters), batch 256, BF16 autocast, channels-last execution, Nesterov SGD, and a piecewise schedule indexed entirely by charged training time: warm up to LR 0.20 over the first 5%, then cosine-decay to a small terminal LR over the remaining 95%. Add expectation-preserving stochastic depth with a maximum rate of 0.08, then anneal it to zero during the final quarter. Retain baseline weight decay `1e-4`, crop/flip, clean cross-entropy, and input scaling to isolate the architecture package. See `proposals/idea-03.md`, refined by `01-idea-review.md`.

**What it targets**: The strongest hypothesized limiter is representational capacity: BASE is a 270K-parameter 2016 model with negligible H20 memory use. A shallow-wide pre-activation network adds capacity and Tensor-Core-friendly channel dimensions without the serial depth cost of a much deeper model.

**Reasoning**: Pre-activation improves residual optimization, width exploits the available accelerator, and literature supports residual-branch stochastic regularization. The corrected piecewise time schedule reaches its stated peak and cannot expand warmup when throughput changes. Baseline weight decay is retained because the original proposal did not justify its 5x increase. Fixed input rescaling is excluded because the first BatchNorm already absorbs much of the scale and adding it would create another unmeasured variable. The proposal has the highest plausible accuracy ceiling of the finalists.

**Sources**: `proposals/idea-03.md`; `papers/shakedrop.md`; `papers/time-matters-regularization.md`; BASE summary in `01-brainstorm.md`.

**Estimated Effort**: high

**Risk Assessment**: Tenfold parameter growth may reduce optimizer exposure, stochastic depth may slow short-horizon fitting, and architecture plus precision plus schedule create a larger debugging surface. The larger model also makes each epoch-end evaluation slower; the plan must estimate total runtime from BASE's 74.3 seconds of aggregate evaluation overhead and enforce the 600-second cap. Restrained drop rates and documented WRN-16-3/no-drop fallbacks contain the main training failure modes.

### Front-Loaded CutMix with Time-Normalized Cosine and EMA
**Summary**: Keep the small ResNet-20 and batch 128, replace fixed-step decay with wall-clock cosine from 0.1 to 0.001, apply CutMix with probability 0.5 only during the first 70% of charged training time, and evaluate a sparsely updated EMA after its 10% activation point. Clean batches remain interleaved early and become exclusive late; label smoothing is intentionally excluded to avoid redundant regularization. See `proposals/idea-02.md`.

**What it targets**: The baseline combines a poorly matched LR horizon with weak data diversity and noisy single-checkpoint weights. This recipe attacks those quality gaps without increasing architecture cost or materially reducing the number of completed updates.

**Reasoning**: The mixed-sample literature supports CutMix as an efficient input-gradient regularizer, while critical-period evidence supports front-loading regularization and relaxing it late. EMA complements CutMix by smoothing the resulting trajectory, and copying online BatchNorm buffers avoids the main implementation trap.

**Sources**: `proposals/idea-02.md`; `papers/mixed-sample-analysis.md`; `papers/regmixup.md`; `papers/time-matters-regularization.md`.

**Estimated Effort**: medium

**Risk Assessment**: CutMix plus a steadily decaying LR can underfit the tiny model; EMA can lag late improvements or mismatch BatchNorm state; and the three-part package makes attribution imperfect. Conservative probability, a clean final 30%, and sparse EMA updates reduce those risks.

## Review

The independent review (`01-idea-review.md`) selected the pre-activation WRN candidate but identified four material refinements. The chosen version uses time-indexed rather than step-indexed warmup, a piecewise warmup-then-cosine formula that actually reaches LR 0.20, baseline weight decay `1e-4` instead of the unsupported `5e-4` change, and an explicit evaluation-overhead/runtime check against the 600-second cap. The review also noted the omitted fixed input rescaling; it remains excluded because BatchNorm plausibly absorbs the unit-std input scale and the first experiment should not add another unmeasured intervention. Stochastic depth remains the first ablation if the model underfits.

## Idea Evaluation

The review scored Time-Aware Pre-Activation WRN-16-4 highest on potential impact (9/10) and found it uniquely addresses both the schedule mismatch and extreme accelerator underuse. Time-Normalized Wide-Batch ResNet-20 had the strongest internal polish but a lower ceiling, while CutMix/Cosine/EMA left the diagnosed capacity limit untouched. The reviewer pick is adopted with the refinements above.

## Chosen Idea
**Selected**: Time-Aware Pre-Activation WRN-16-4 with Restrained Stochastic Depth

**Why this idea**:
It makes the strongest capacity bet that remains plausible in the measured hardware envelope and couples it to an optimization schedule that is invariant to throughput. The 10x parameter increase still occupies a small fraction of H20 memory, while a shallow-wide topology avoids excessive serial depth. The corrected schedule and unchanged weight decay remove the reviewer's main internal-consistency concerns without diluting the core hypothesis.

**Hypothesis**:
A pre-activation WRN-16-4 trained with batch 256, BF16 channels-last execution, restrained early stochastic depth, and a piecewise time-normalized warmup/cosine schedule will complete within 600 total seconds on GPU 0 and achieve `best_test_acc >= 91.61%`, with a likely result above 93.0%, because it converts unused H20 capacity into representation quality while guaranteeing late low-LR refinement.

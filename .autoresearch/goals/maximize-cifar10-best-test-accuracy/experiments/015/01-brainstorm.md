# Brainstorm EXP-015
**Created**: 2026-08-06

## Web Search & Literature Review

- **When, Where and Why to Average Weights?** (`experiments/015/papers/when-where-why-average-weights.md`; https://proceedings.mlr.press/v267/ajroldi25a.html)
  ICML 2025 finds that checkpoint averaging gives efficiency and mild generalization gains at low implementation cost, and that averaging works best together with learning-rate annealing rather than replacing it.
- **Reconciling Modern Deep Learning with Traditional Optimization Analyses: The Intrinsic Learning Rate** (`experiments/015/papers/intrinsic-learning-rate.md`; https://papers.nips.cc/paper/2020/hash/a7453a5f026fb6831d68bdc9cb0edcae-Abstract.html)
  NeurIPS 2020 explains BatchNorm SGD through LR/decay-controlled trajectory dynamics; its supplemental CIFAR evidence shows that checkpoints on one local trajectory can average well even while their parameter distance grows.
- **Existing weight-averaging distillation** (`knowledge/papers/weight-averaging.md`)
  The persistent knowledge entry emphasizes late averaging plus annealing and the main local implementation hazard: coherent BatchNorm buffer handling.
- **PyTorch Automatic Mixed Precision guidance** (https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html; https://docs.pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
  Official guidance keeps master model/optimizer state in FP32 and autocasts forward/loss operations; BF16 retains FP32-like dynamic range, while gradient scaling is chiefly needed to prevent FP16 underflow. Local dtype and speed behavior still require measurement.
- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** (https://arxiv.org/pdf/1706.02677)
  Goyal et al. report a modest ResNet gain from zero-initializing each residual branch's final BN scale. Transfer is limited because their deeper projection-shortcut network lacks this CIFAR model's dead padded-channel hazard.

## Experimental History Review

- EXP-010 remains the 94.15% frontier: width-2 postactivation ResNet-20, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, then a hard weak cosine tail. It retained 26,898 steps, switched at 89.73%, reached 93.16% at the first weak checkpoint, and finished at its best with NLL 0.1934.
- Stronger CutMix and full preactivation lowered strong fit. Preactivation still reached 94.22%, but its +0.07 missed the formal gate; preserve postactivation and avoid compounding strong-view underfit.
- Decay changes failed twice. Keep coupled all-parameter `1e-4`; neither more decay nor excluding BN/bias is a live direction.
- Batch 256 was stable but supplied only 1.189x image throughput, below its 1.20x mechanism floor. Pure batch scaling is retired at that operating point, while lower-precision convolution remains untested.
- EXP-014 proved that exact initial logits are insufficient protection for a new branch: raw max features produced a 4.10x classifier-gradient norm, one-step loss explosion, a 3.96x weight ratio, and 10.00% throughout. Any additive representation must gate first-update scale, not only output at construction.
- The system decomposition remains current at `7c1e7d8`: model backward is 75.46% of counted step time, forward 22.11%, loader/transfer/optimizer overhead small, and memory only 598.7 MiB of 97,871 MiB. The live accuracy limiter is generalization within a short strong phase; the live systems opportunity is faster convolution/BN math.
- Untried gaps include mixed-precision exposure, online trajectory averaging with coherent buffers, isolated Nesterov on the accepted schedule, and postactivation identity initialization that deliberately leaves transition blocks active.

## Collected Ideas

- **Weak-tail online EMA** — clone the accepted model at the 80% switch and maintain an exponential moving average of parameters plus floating BatchNorm buffers during the already annealed weak tail, evaluating the EMA at the existing once-per-epoch calls. This targets terminal trajectory variance without perturbing the validated strong phase; ICML 2025 directly supports averaging plus annealing, but state coherence and counted update overhead need strict gates.
- **H20 BF16 autocast exposure** — run accepted forward/loss/backward under installed BF16 autocast while retaining FP32 master parameters and ordinary SGD. This directly targets the measured 97.57% model forward/backward cost and could process substantially more seed-identical batches in 300 seconds; H20 tensor cores and BF16 range make it plausible, but kernel selection, BN/CE dtypes, numerical drift, and altered update count require fresh paired validation.
- **Channels-last convolution layout** — convert model and batches to `channels_last` while retaining FP32 and every algorithmic choice. It attacks convolution/BN backward without changing the mathematical architecture or parameter count, and can expose H20-optimized kernels; tiny CIFAR maps and channel widths may instead make layout conversion or kernel choice slower.
- **Isolated Nesterov momentum** — enable Nesterov on the accepted 80%-hold schedule while preserving momentum 0.9, decay, augmentation, and all timing. EXP-001 bundled Nesterov with the failed 15% hold, so it never isolated the standard lookahead-gradient update on the successful schedule. It is cheap and evidence-backed generally, but likely has a small ceiling and changes effective early steps.
- **Same-width zero-gamma residual initialization** — zero only `bn2.weight` in the six stride-1 equal-width blocks, leaving all three transition blocks accepted. Ordinary blocks begin as exact postactivation identities with a trainable gamma gradient, while active transitions prevent the Option-A padded-channel deadlock that retired all-block zero-gamma. The risk is delayed residual learning and the same strong underfit pattern seen in full preactivation.
- **Anti-aliased transition shortcuts** — replace Option-A stride-two slicing with a fixed 2x2 average downsample before zero-padding, retaining parameter count and postactivation residual branches. This targets information loss/shift sensitivity at the only resolution changes and is cheap relative to convolutions, but it deviates from the original CIFAR Option-A semantics and could blur already-small features.
- **Classifier-only weak-tail averaging** — maintain an online average of only `fc.weight` and bias during the cosine tail, avoiding BatchNorm-state mismatch and reducing averaging overhead to 1,290 values. It is a simplification of full-model EMA that targets readout noise, but the accepted best/final gap is zero and most useful trajectory diversity may live in convolutional features.
- **BF16-funded width-3 moonshot** — combine lower-precision convolution with width 3 so tensor-core throughput pays for additional representation capacity while memory remains trivial. Width 2 previously bought +1.25 despite fewer steps, making a larger capacity/exposure frontier plausible; the cross is high-risk because two mechanisms change together and BF16 may not offset the roughly quadratic convolution cost.

## Combinations

- **BF16 + width 3**: mixed precision could convert H20 tensor-core headroom into a larger postactivation representation rather than merely extra width-2 updates. The combination has a higher ceiling than BF16 or width alone if width-3 BF16 retains near accepted exposure, but it requires a stronger timing gate and weakens attribution.
- **Weak-tail EMA + isolated Nesterov**: Nesterov could improve the live trajectory while EMA smooths its terminal iterates. The cross might outperform either alone if Nesterov adds useful late diversity, but it should not be the first test because a miss would be uninterpretable and the accepted schedule has not isolated Nesterov.
- **Same-width zero-gamma + anti-aliased transitions**: identity-initialized within-stage blocks would preserve activations while active averaged transitions reduce aliasing and keep new padded channels learnable. The package addresses both initialization and downsampling, but combines two representation changes and could deepen strong-phase underfit.

## Candidate Ideas

### H20 BF16 Autocast Exposure
**Summary**: Wrap only the accepted training forward and cross-entropy in CUDA BF16 autocast, leaving model parameters, gradients, BatchNorm persistent state, SGD momentum/decay, and the untouched evaluator in FP32. Do not use a GradScaler, manually cast the model/inputs, change memory format, or combine capacity. Advance only if fresh H20 pairs prove at least 1.15x synchronized speedup rooted in forward/backward kernels, with aligned finite hard/soft updates and at least 30,932 projected steps. Full proposal: `proposals/idea-02.md`.

**What it targets**: The measured model forward/backward consumes 97.57% of CUDA-stage time, with backward alone at 75.46%. BF16 Tensor Core kernels could process more accepted batch-128 updates and images inside the fixed 300 seconds without the update-noise loss that invalidated batch 256.

**Reasoning**: PyTorch officially supports BF16 autocast for lower-precision-eligible convolution/linear work while retaining FP32 master state and FP32-policy loss reductions. The H20 supports BF16 and memory is unconstrained. This is the only finalist that attacks the dominant systems bottleneck directly, though additional exposure has not yet been shown to improve the already-flat accepted trajectory.

**Sources**: `02-system-understanding.md`; EXP-010 and EXP-013; PyTorch AMP documentation linked above; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Small CIFAR kernels, BN/add launches, or existing TF32 may erase speedup. BF16 rounding changes gradients and BN-observed activations; extra time-aligned steps also apply more decay and may over-optimize. Numeric, dtype, loader, stage-attribution, evaluation-count, and total-wall gates must all pass without FP16/scaler/layout fallback.

### Same-Width Residual Identity Initialization
**Summary**: Set `bn2.weight` exactly zero only in the six non-entry stride-1 equal-width blocks; retain gamma one in all three stage-entry blocks, including both Option-A padded transitions. The accepted postactivation topology, state shapes, parameter count, optimizer, and entire EXP-010 recipe remain unchanged. The six ordinary blocks start as exact forward identities and recruit their normalized residual branches after gamma's first update. Full proposal: `proposals/idea-03.md`.

**What it targets**: A compute-neutral representation/conditioning opportunity left by EXP-012's 94.22 near miss. It aims to improve residual optimization without canonical preactivation's global reorder or the all-block zero-gamma deadlock in newly padded channels.

**Reasoning**: Goyal et al. report directional ResNet evidence for last-BN zero initialization. Locally, the scoped rule preserves active stage-entry feature creation: no transition gamma is zero, so the 32 and 64 new channel halves receive gradients. Explicit first-update gamma, replay-loss, class-concentration, branch-recruitment, 64-step fit, and 99%-exposure gates address EXP-014's lesson that initial functional identity alone is insufficient.

**Sources**: Goyal et al. linked above; EXP-012/014 analyses; `03-experiment-learnings.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Six residual convolutions receive no data gradient on step one and may stay weak, compounding the known strong-phase underfit. Literature transfer from deep ImageNet projection ResNets is indirect. A large first gamma update can still destabilize the replay even with normalized features; strict safety gates are mandatory.

### Late Weak-Tail Online Parameter EMA
**Summary**: Preserve the online EXP-010 trajectory and, starting at 90% pre-step progress, maintain a detached FP32 EMA of every trainable parameter with a fixed one-epoch half-life (`decay = 0.5 ** (1/390)`). At each existing eligible endpoint, temporarily install EMA parameters, use current online BN buffers for exactly one evaluation, and restore online values in `finally`. All shadow updates and swaps are charged to the 300-second timer; no BN recalibration, second model evaluation, or alternate window is allowed. Full proposal: `proposals/idea-01.md`.

**What it targets**: Terminal weak-tail trajectory variance/generalization while leaving the validated strong and early weak adaptation phases untouched. The accepted run finished at its best, so the narrow aim is to find a flatter nearby parameter estimate without changing representation or regularization.

**Reasoning**: ICML 2025 finds averaging plus LR annealing performs best across modern workloads, and NeurIPS 2020 supplies local-trajectory CIFAR/BatchNorm rationale. A 90% start excludes rapid strong-to-weak adaptation; a derived one-epoch half-life avoids an accuracy-tuned constant. Fused foreach updates should retain at least 99% of accepted steps and consume only about 8 MiB with backups.

**Sources**: EXP015 paper distillations; `knowledge/papers/weight-averaging.md`; EXP-010; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: EMA can lag a trajectory that is still improving; adjacent steps may be too correlated; current online BN buffers are only an approximation for averaged parameters. Evaluating EMA instead of online after 90% may forfeit the accepted endpoint, while evaluating both would bias observation count. Timer accounting and bitwise restoration must be proven.

## Review

Mandatory external Claude review completed successfully in `01-idea-review.md`; no fallback reviewer was used. It selected same-width zero-gamma at 8/10 evidence and 6/10 impact. The key distinction was objective alignment: this candidate targets optimization/generalization without changing exposure, whereas BF16 lacks evidence that more accepted updates improve accuracy and EMA targets a tail whose local best/final variance is only about 0.01 points.

I adopted the reviewer's stronger framing that the likely risk is neutrality at ResNet-20 depth, not material underfit from a one-step recruitment delay. I retained every first-update safety gate and the active padded-transition rule. I did not adopt the optional suggestion to also zero safe `layer1[0]`: the reviewed finalist and detailed proposal predeclare six ordinary non-entry blocks, and keeping all stage entries active gives a cleaner symmetric mechanism with less risk of weakening initial feature creation. Any bare 94.25% pass remains formally valid but weak single-seed causal evidence.

## Idea Evaluation

- **Same-width zero-gamma**: selected. It has the strongest local mechanism, clean compute-neutral attribution, and gates that directly address EXP-014's update-scale collapse.
- **BF16 autocast**: second. It can be a future systems experiment after evidence that extra accepted exposure moves accuracy; this loop should not spend its only run on an unproven exposure-to-generalization link.
- **Late EMA**: third. Strong external averaging evidence does not overcome the accepted run's monotonic final-at-best trajectory, current-buffer mismatch, and forced loss of online endpoint observations.

## Chosen Idea
**Selected**: Same-Width Residual Identity Initialization

**Why this idea**:
Zero exactly the six `bn2.weight` tensors in non-entry, stride-1, equal-width postactivation blocks and leave all three stage-entry blocks at gamma one. This preserves the accepted model graph, parameter count, exposure, transitions, optimizer, data, schedule, and RNG while testing a canonical identity-initialization mechanism in the only blocks where it is structurally safe. The strict first-update and short-fit gates prevent repeating EXP-014's mistaken assumption that zero initial output is automatically optimization-safe.

**Hypothesis**:
The six ordinary residual branches will recruit smoothly after one update, retain at least 99% of EXP-010's 26,898 steps, keep the 80% strong checkpoint above the 87.08 underfit marker, and raise `best_test_acc` from 94.15% to at least 94.25%. A first-step gamma above 0.25, replay loss above 2x control/pre-update, one-class concentration, failed second-step branch recruitment, or 64-step fit above 1.5x control vetoes the full run.

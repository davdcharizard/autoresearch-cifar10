# Brainstorm EXP-029
**Created**: 2026-08-06

## Web Search & Literature Review

- **Gradient Centralization** (`knowledge/papers/gradient-centralization.md`; ECCV 2020)
  Per-output-filter gradient centering is a projected-gradient regularizer with broad vision evidence, no forward change, and no second gradient; the local pre-decay variant must preserve accepted coupled decay explicitly.
- **PyTorch Channels-Last Memory Format** (`knowledge/references/pytorch-channels-last.md`; official PyTorch tutorial)
  CUDA Conv2d and BatchNorm can propagate channels-last layout when both model and inputs are converted, but official headline gains emphasize reduced precision. FP32 32x32 behavior is unresolved and must be timed locally.
- **Deep Networks with Stochastic Depth** (`knowledge/papers/stochastic-depth.md`; ECCV 2016)
  Random residual-block bypass can regularize while reducing expected training compute, with direct CIFAR evidence; transfer from 110/1202-layer networks to nine local blocks is the central risk.
- **CutMix** (`knowledge/papers/cutmix.md`; ICCV 2019)
  Regional mixing is the strongest validated local data lever. Its alpha controls area severity without changing target type or adding GPU work, but the accepted alpha-1 point already balances strong fit and regularization well.

## Experimental History Review

- EXP010 remains the 94.15% frontier: width-2 depth-20, all-parameter decay `1e-4`, 80% LR hold, N1/M7, probability-0.5 alpha-1 CutMix, then an abrupt low-LR weak hard tail. EXP005/027 protect the simultaneous data/LR boundary.
- Global optimizer paths are now a recurring high-importance failure. EXP020/022/028 each produced candidate-only class concentration; EXP028 did so despite exact first-step and constant-gradient scale, with a 12.35x stochastic update spike. Preserve ordinary momentum.
- Architecture/recruitment changes often lose strong fit or destabilize early geometry: preactivation, zero-gamma, asymmetric width, raw-max readout, pooled shortcuts, and ECA failed. Any global representation change needs exact-corpus output/update gates.
- EXP012 and EXP026 each reached 94.22%, only 0.03 below the formal gate, showing modest representational/data improvements can get close but often trade switch fit or NLL. A small isolated regularizer is plausible if it preserves the full accepted curriculum.
- Backward remains 75.46% of counted step time and memory is abundant. EXP013 found batch 256 supplied 18.91% more image throughput but missed its preregistered 20% premise; channels-last and stochastic depth are untested ways to attack compute without linear-LR large-batch behavior.

## Collected Ideas

- **Conv2d-weight-only gradient centralization** — Subtract each convolution filter's data-gradient mean after backward, then use unchanged ordinary SGD and all-parameter coupled decay. It targets generalization while preserving the accepted forward function, optimizer state path, data, and schedule; direct ECCV evidence and the reviewed EXP028 fallback make it the cleanest post-PNM probe.
- **FP32 channels-last training** — Convert initialized model weights and each transferred batch to channels-last so cuDNN may use NHWC kernels for the convolution/BN-dominated forward/backward. It targets the measured 75.46% backward bottleneck without changing logical shapes, capacity, batch noise, targets, or optimizer, but official evidence is strongest for reduced precision rather than this tiny FP32 workload.
- **Linear stochastic depth** — Give the nine residual blocks linearly declining survival probabilities with a conservative final survival of 0.9, bypassing whole residual branches per batch during training and using survival-scaled full branches at evaluation. It combines regularization with expected backward savings, but each dropped block is a large perturbation in ResNet-20 and may compound the locally protected strong-phase underfit.
- **Batch 192 with sqrt LR scaling** — Seek a better image/update tradeoff than EXP013 batch 256 by using batch 192 and LR scaled by `sqrt(1.5)`, with fixed elapsed evaluation checkpoints. It targets unused H20 parallelism while limiting gradient-noise and update-count change, but lacks a direct accuracy precedent and needs a new throughput premise rather than relaxing EXP013.
- **CutMix alpha 0.5 at probability 0.5** — Keep the accepted number of mixed batches but bias boxes toward smaller/larger regions than alpha 1, reducing the frequency of near-half composites while retaining regional class evidence. This is cheap and isolated, yet the direction of regularization is ambiguous and post-result alpha tuning would be an overfit hazard.
- **Standard stride-2 Option-B shortcuts** — Replace only the two padded subsampling identities with direct stride-2 1x1 learned projections, avoiding the pool-first aliasing path rejected by EXP017/021. It could improve channel transport without extra pooling, but canonical CIFAR Option A historically regularizes well and another transition change risks NLL regression.
- **Short-horizon weak-tail EMA** — Maintain an exponential average only after the 80% transition and evaluate it on the existing schedule without BN recalibration, copying online BN buffers. It could reduce late noise without SWA's backward-looking uniform bias, but EXP010 ends at its best and parameter averaging adds counted optimizer work.
- **Fixed slope-0.01 LeakyReLU** — Preserve negative activation gradients with matched Kaiming gain. It is parameter-free and cheap, but BN weakens the dying-ReLU premise and the change touches every block plus final pooled features; it remains lower priority after repeated strong-phase representation suppression.
- **Moonshot channels-last plus width redistribution** — Spend a demonstrated layout speedup on modest stage-specific capacity while retaining depth. The combination could turn systems headroom into representational headroom, but it destroys attribution unless channels-last first proves both throughput and numerical safety alone.

## Combinations

- **Channels-last + gradient centralization**: Layout savings could pay for GC's 38 small reductions while GC supplies an accuracy mechanism. The cross is plausible only after measuring each component alone; combining now would obscure whether any gain came from exposure or projection.
- **Stochastic depth + extra depth**: Add blocks while dropping a subset during training, then evaluate the deeper ensemble-like full network. This is closer to the paper's successful regime and could preserve expected training cost, but it changes capacity, RNG, and inference depth simultaneously and is too broad before local shallow stochastic depth is known safe.
- **Batch 192 + channels-last**: NHWC efficiency could improve the batch-scaling knee and increase examples without batch 256's update loss. It may outperform either alone on throughput, but batch noise, LR scaling, and layout numerics would be confounded.

## Candidate Ideas

### Conservative Linear Stochastic Depth
**Summary**: Apply batchwise whole-branch stochastic depth to all nine residual blocks with fixed linear survivals from 1.0 to 0.9. Training executes surviving branches unscaled and bypasses dropped branches; evaluation executes every branch scaled by its survival probability. A dedicated CPU generator supplies exactly nine draws per batch. Full specification: `proposals/idea-03.md`.

**What it targets**: Both limiters at once: expected residual-branch compute falls about 5%, potentially increasing fixed-budget steps, while sampled effective depth supplies a literature-backed generalization regularizer.

**Reasoning**: The ECCV paper reports CIFAR improvements and shorter training graphs, giving the candidate higher upside than a pure layout change. The 0.9 final survival is intentionally conservative because the accepted network has only nine blocks and local representation changes repeatedly suppress strong fit. Exact forced-transition, RNG, mask-frequency, BN, output-geometry, and timing tests are load-bearing.

**Sources**: `knowledge/papers/stochastic-depth.md`; Huang et al. arXiv 1603.09382; EXP010, EXP012/015, EXP021/024/025; `02-system-understanding.md`.

**Estimated Effort**: high.

**Risk Assessment**: High. Published gains are concentrated in much deeper ResNets; dropping local transition branches zeroes newly padded channels, skipped BN/decay/momentum create depth-dependent state, and CutMix/RandAugment already stress strong fit. Control-flow variability may also miss even a 1% timing gate.

### Conv2d-Weight-Only Data-Gradient Centralization
**Summary**: After every backward pass, subtract each of the 19 Conv2d weights' per-output-filter gradient mean over input/spatial dimensions, then run unchanged PyTorch momentum SGD. FC, BN, parameters, forward graph, initialization, all-parameter coupled decay, curriculum, schedule, and evaluator remain accepted. Full specification: `proposals/idea-01.md`.

**What it targets**: The accuracy limiter is generalization without losing EXP010's strong-phase fit. GC removes common-mode convolution data-gradient drift while preserving the accepted initial function and ordinary optimizer state path; this directly avoids the recurring global-optimizer failure family.

**Reasoning**: ECCV evidence reports improved vision generalization and low-resolution convolution-only use. Locally, the intervention is narrower than recent failures: it cannot increase instantaneous Conv data-gradient norm, does not add alternating state or parameter pullback, and preserves validated `1e-4` decay by applying projection before SGD adds decay. Its 38 small operations still need strict timing because the accepted step is only about 10.9 ms.

**Sources**: `knowledge/papers/gradient-centralization.md`; `experiments/028/proposals/idea-02.md`; EXP003, EXP008/009, EXP020/022/028; `02-system-understanding.md`.

**Estimated Effort**: medium.

**Risk Assessment**: Medium scientific/runtime risk. The local `P(g)+lambda*w` operator is narrower than the paper's projected-decay theorem, published schedules are longer, and small reductions could cost more than 1% exposure. Momentum can still integrate a different trajectory, so exact-corpus concentration/update gates remain mandatory.

### FP32 Default-TF32 Channels-Last Training
**Summary**: Initialize the accepted model identically, then restride its 4-D weights and every counted training input to `torch.channels_last`; add a forward-boundary conversion so the immutable contiguous-input evaluator uses the same layout. Keep FP32/default-TF32, logical NCHW shapes, batch 128, model/data/SGD/schedule unchanged, and equalize production to the accepted 19 evaluation looks. Full specification: `proposals/idea-02.md`.

**What it targets**: The measured systems limiter: convolution/BN backward consumes 75.46% of GPU-stage time. A faster legal layout could add optimizer exposure without changing batch noise, targets, model capacity, or optimizer decisions.

**Reasoning**: Official PyTorch documentation confirms CUDA Conv2d/BatchNorm channels-last propagation and possible faster kernels. This is the most direct untried systems intervention compatible with Python 3.14, but official GPU headline gains emphasize reduced precision rather than tiny FP32 CIFAR tensors. Seven fresh paired trials and profiler checks therefore require a real 3% full-step speedup including input restride, with no hidden layout repairs.

**Sources**: `knowledge/references/pytorch-channels-last.md`; official PyTorch memory-format tutorial; EXP013; `02-system-understanding.md`.

**Estimated Effort**: medium-high.

**Risk Assessment**: High feasibility/attribution risk. NHWC may be neutral or slower at these shapes, Option-A slice/pad may cause conversions, and different legal cuDNN kernels change numerics. The proposal's exact-19-look guard is essential to prevent faster epochs from gaming the maximum-over-evaluations metric.

## Review

Claude's independent adversarial review (`01-idea-review.md`) selected **Conv2d-weight-only data-gradient centralization**, scoring it 8/10 on evidence/reasoning and 6/10 on impact. Channels-last scored 4/10 on both axes because tiny FP32 speedup and exposure-to-accuracy transfer are two unproven links. Stochastic depth scored 4/10 and 5/10 because nine blocks are far outside its strongest literature regime and branch bypass re-enters the locally failed identity-oriented/transition-fragility family.

Two GC caveats are adopted as first-class interpretation constraints. First, the exact local recurrence is `P(g)+lambda*w`, not the paper's `P(g+lambda*w)`, so published evidence is directional rather than proof for this operating point; preserving locally validated all-parameter decay remains more important than forcing theorem alignment. Second, every Conv feeds BN, which may make common-mode filter gradients partly redundant. Preflight and production diagnostics must therefore report removed-mean norm as a fraction of raw gradient norm by stage and phase. A tiny fraction plus a null result means the mechanism was near-inert, not that another GC strength/layer subset should be tried inside EXP029.

The reviewer correctly retained the recurring optimizer-path warning: instantaneous projection cannot increase Conv data-gradient norm, but this does not bound momentum-integrated trajectory geometry. The exact-corpus concentration and update-spike gates—not the norm theorem—are the safety basis.

## Idea Evaluation

- **Conv2d-weight-only data-gradient centralization** — Advance. It is the only finalist whose primary payoff directly targets generalization at fixed forward function/exposure, and it preserves ordinary SGD state while offering clean attribution. The decay-order and BN-redundancy gaps lower the expected effect but do not undermine the falsifiable test.
- **FP32 channels-last** — Defer. It directly attacks backward cost, but official evidence is weak for tiny FP32 tensors and even a 3% speedup may not improve accuracy; changed cuDNN numerics further blur exposure attribution.
- **Conservative stochastic depth** — Reject for EXP029. Direct CIFAR evidence does not transfer cleanly from 110-1202 layers to nine blocks, and droppable transition branches compound strong-phase and Option-A fragility already observed locally.

## Chosen Idea
**Selected**: Conv2d-Weight-Only Data-Gradient Centralization

**Why this idea**:
GC is the best mechanism-to-limiter match: it tests a literature-backed generalization projection without changing the accepted forward graph, initialization, ordinary momentum state, data curriculum, LR schedule, or evaluation process. Unlike PNM/Nesterov/Lookahead, it adds no alternating or displaced optimizer state and cannot increase the instantaneous Conv data-gradient norm. Its remaining trajectory and runtime risks are directly measurable before production.

**Hypothesis**:
Applying full-strength per-output-filter centralization to all 19 Conv2d loss gradients immediately before unchanged coupled-decay momentum SGD will remove a nontrivial common-mode gradient component without suppressing the protected strong phase, preserve at least 99% of accepted update exposure, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.25-94.30%. Any safety/timing veto invalidates this exact point; a valid lower result rejects it, and neither outcome authorizes strength, layer, phase, decay-order, or corpus tuning within EXP029.

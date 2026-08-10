# Brainstorm EXP-024
**Created**: 2026-08-06

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (`knowledge/papers/eca-net.md`; CVPR 2020)
  Global spatial descriptors plus a tiny channel-axis convolution can recalibrate residual features with negligible parameters and FLOPs. The transfer to this shallow, CutMix-trained network is uncertain, so exact-function initialization and real H20 timing are essential.
- **Generalizing Pooling Functions in Convolutional Neural Networks** (`knowledge/papers/mixed-pooling.md`; AISTATS 2016)
  Smooth or learned mixtures of spatial statistics can outperform one universal pooling rule. EXP-014 showed that an unconstrained max branch is optimization-discontinuous here, narrowing a credible revisit to bounded, scale-controlled aggregation.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`; BMVC 2016)
  Width can improve CIFAR representation efficiency, but EXP-023 showed that globally trading away depth loses at the current frontier. A stage-specific allocation that preserves all ResNet-20 blocks is a materially different test.
- **CutMix** (`knowledge/papers/cutmix.md`)
  Regional label mixing rewards distributed, area-sensitive class evidence. Any attention or pooling change must avoid turning a small high-response patch into an area-insensitive shortcut during the strong phase.

## Experimental History Review

- The moving frontier remains EXP-010 at 94.15%: width-2 postactivation ResNet-20, standard momentum and all-parameter decay `1e-4`, N1/M7 plus alpha-1 CutMix on half of strong-phase batches through 80%, then a hard weak tail. It reached 89.73% at the switch, 93.16% at the first weak checkpoint, and 26,898 updates.
- Validated gains are compositional but narrow: the long high-LR plateau, phase-bounded RandAugment, width 2, and conservative CutMix. New representation ideas should preserve that recipe and isolate one mechanism.
- Repeated failures rule out changing the standard momentum path without warm/alignment safeguards, changing width-2 decay, identity-oriented residual suppression, and pool-first transitions. These are not useful finalists for this loop.
- EXP-014's raw average-plus-max classifier collapsed from a 4.10x first gradient. A pooling revisit is credible only if the statistic is smooth and bounded and an exact-corpus probe checks first-update scale and concentration.
- EXP-023's 1.54M-parameter width-3 ResNet-14 ran efficiently and entered the weak tail well but peaked at 94.00%. The result rejects sacrificing one block per stage, not depth-preserving conditional or stage-specific capacity.
- The measured systems bottleneck remains model backward at 75.46% of step time; the accuracy constraint is to preserve the accepted 89.73% strong-phase switch fit while raising weak-tail generalization. Extra parameters are affordable in memory, but every sequential kernel or wider convolution must justify lost updates.

## Collected Ideas

- **Identity-scale final-stage ECA** — insert tiny channel-axis attention after the final BN in each `layer3` residual branch and multiply by `2*sigmoid(g)`, with the gate convolution initialized to zero. This preserves the exact initial network function while learning input-conditional channel allocation only at 8x8 resolution; it targets representation quality with little arithmetic, though three new sequential descriptor/gate paths can reduce exposure and CutMix can confound global descriptors.
- **Fixed GeM-3 global pooling** — replace global average pooling over nonnegative final features with a clamped cubic generalized mean. It targets compact semantic evidence while spreading gradients over all positive spatial sites, unlike EXP-014's raw max branch; its altered initial logits and potential feature-scale amplification require aligned first-update diagnostics.
- **Bounded average/max interpolation** — mix average and RMS-normalized max features through one global scalar constrained to a small range and initialized at pure average. It directly revisits the failed pooling family with a controlled optimization path, but exact identity plus usable gradient is technically delicate and target-area semantics remain weak under CutMix.
- **Depth-preserving final-stage widening** — retain all nine residual blocks and widths 32/64 through `layer2`, then widen `layer3` and its classifier above 128 channels. This spends capacity at 8x8 resolution, preserves early/middle refinement, and exploits EXP-023's sublinear H20 width scaling; the transition and three widened blocks may still cost too many updates.
- **Channels-last accepted model** — convert model parameters and every input batch to channels-last physical layout, leaving mathematical training unchanged. This attacks the 75.46% convolution-backward bottleneck and could buy more accepted-recipe exposure, but tiny FP32 CIFAR kernels may see no speedup and layout conversions could erase gains.
- **Batch-96 high-noise exposure point** — reduce batch size while preserving LR and the accepted recipe, seeking more optimizer updates and beneficial gradient noise rather than image throughput. It is a simple orthogonal operating-point test, but H20 under-utilization may reduce images seen and the unscaled LR could destabilize the soft-target phase.
- **Plateau-only stochastic depth on late identity blocks** — randomly skip only non-transition residual branches during strong training and disable skipping in the weak tail. This simplifies training-time computation and regularizes representation while restoring the full model for refinement, but changing effective depth and BatchNorm sampling can suppress the already-short strong fit.
- **Moonshot: conditional mixture of average and GeM descriptors** — compute average and smooth high-order descriptors and use an input-conditioned channel gate to blend them before the classifier. It could adapt evidence concentration to each image and CutMix composition, but combines two unvalidated mechanisms, adds sequential kernels, and has poor attribution for one scored run.

## Combinations

- **ECA + channels-last**: memory-format speedup could fund the sequential attention kernels, making conditional channel allocation closer to exposure-neutral. The cross beats either part only if channels-last has an independently verified speed margin; otherwise it obscures whether representation or systems behavior caused the result.
- **Final-stage widening + ECA**: extra late channels provide capacity and ECA decides which channels matter per image. This could outperform static width or gating alone, but EXP-023 and the fixed-time budget make the combined compute risk too high before either isolated mechanism succeeds.
- **GeM + weak-tail-only activation**: keep average pooling during CutMix training, then enable smooth evidence concentration only when targets return to hard labels. This avoids CutMix area ambiguity, but introducing a new readout at the phase boundary creates an abrupt function change and leaves too little time to adapt.

## Candidate Ideas

### Identity-Initialized Final-Stage ECA
**Summary**: Add ECA gates to all three `layer3` residual branches only. Each gate globally averages the 128-channel 8x8 residual, applies a bias-free length-5 channel-axis `Conv1d`, and multiplies the residual by `2*sigmoid(logits)` before the unchanged shortcut addition. Zero kernels give exact unit gates and preserve the accepted initial function; only 15 parameters are added, for 1,073,977 total. Full proposal: `proposals/idea-01.md`.

**What it targets**: Conditional allocation of mature semantic channels without removing ResNet-20 depth or globally widening convolution. It aims to raise weak-tail generalization while adding minimal arithmetic to a system whose backward path already consumes 75.46% of step time.

**Reasoning**: Width-2 capacity was highly valuable in EXP-007, while EXP-023 showed static width cannot compensate for deleted blocks. ECA supplies input-dependent capacity at the cheapest spatial stage and begins at the exact accepted function, avoiding EXP-012/015's initial branch suppression. The main uncertainty is sequential launch overhead and whether CutMix-blended global descriptors learn useful rather than area-confounded gates.

**Sources**: `knowledge/papers/eca-net.md`; EXP-007, EXP-010, EXP-012, EXP-015, EXP-023; `proposals/idea-01.md`.

**Estimated Effort**: high — small production code change, but exact identity/RNG, immutable-corpus recruitment, and five-pair timing gates are necessary.

**Risk Assessment**: Tiny attention modules can be launch-bound; EXP-012 measured 1.233x cost for nine SE gates. Zero initialization guarantees only step zero, so gate movement, gradient scale, and candidate-only class concentration must be vetoed before production. Channel adjacency has no guaranteed semantic ordering, and CutMix descriptors may combine incompatible regions.

### Depth-Preserving Final-Stage Widening to 160
**Summary**: Preserve all nine ResNet-20 blocks and accepted widths 32/64 through `layer2`, but widen `layer3` and the classifier from 128 to 160 channels. The candidate keeps 19 convolutions and unchanged Option-A logic, has exactly 1,507,818 parameters, and raises estimated MACs only 17.18% because added work is at 8x8. Full proposal: `proposals/idea-03.md`.

**What it targets**: Static semantic capacity without EXP-023's loss of iterative depth. It spends extra channels where spatial cost is lowest, while fresh timing tests whether H20's sublinear width scaling retains at least 24,000 projected updates.

**Reasoning**: EXP-007 established a strong local width benefit, and EXP-023 retained healthy optimization but lost after deleting three blocks. A 32/64/160 allocation isolates the report's recommended depth-preserving, stage-specific alternative and changes no optimizer or data mechanism. It has similar parameter count but substantially fewer MACs than EXP-023.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-007, EXP-010, EXP-023; `proposals/idea-03.md`.

**Estimated Effort**: medium — parameterized architecture change plus immutable-corpus safety and paired timing controllers.

**Risk Assessment**: The second Option-A shortcut would preserve only 64 of 160 channels and zero-pad 96, weakening identity propagation. Extra static parameters may only reduce train loss, and a 12% step-time allowance still costs meaningful weak-tail exposure. The result jointly tests capacity, transition geometry, initialization stream, and exposure.

### Scale-Matched Bounded Average–GeM-3 Pooling
**Summary**: Replace pure GAP with a convex mixture of GAP and per-example norm-matched GeM-3. A single learned scalar starts at 1/64 and is structurally capped at 1/8; the GeM arm is scale-matched with a detached norm ratio, preserving bounded pooled-feature norm and avoiding any independent classifier. One parameter is added, for 1,073,963 total. Full proposal: `proposals/idea-02.md`.

**What it targets**: The final spatial aggregation error mode: GAP may dilute compact class evidence, while EXP-014 proved that raw max has an unsafe first-update scale. The bounded smooth arm attempts a small salience correction without unbounded feature or classifier growth.

**Reasoning**: Mixed-pooling and GeM literature supports learnable interpolation and smooth average-to-max statistics, and the proposal supplies analytic forward-norm and initial-gradient bounds. It is materially different from EXP-014 because it has one classifier, no raw max, no zero-output branch, dense spatial gradients, and a capped contribution. Evidence in this exact classification/CutMix regime remains indirect.

**Sources**: `knowledge/papers/mixed-pooling.md`; EXP-010 and EXP-014; `proposals/idea-02.md`.

**Estimated Effort**: high — implementation is compact, but numerical reference checks, immutable-corpus scale diagnostics, and timing of several reduction kernels are substantial.

**Risk Assessment**: Equal feature norms do not preserve feature direction or logits. The detached input-dependent scale has unusual gradient semantics, the scalar can saturate under LR 0.1, and even bounded GeM can overweight a compact CutMix region. Multiple reductions and pointwise kernels may cost more exposure than their FLOPs imply.

## Review

The fallback adversarial critic ranked depth-preserving final-stage widening first (evidence/reasoning 8/10, impact 7/10), ahead of bounded GeM (5/10, 3/10) and ECA (4/10, 4/10). It judged 32/64/160 to be the direct follow-up to EXP-023's depth-loss diagnosis and the only finalist that adds substantial representation capacity with a locally validated width prior. I adopted its material refinements: lower the point estimate from 94.35% to 94.25-94.30%, treat the 96-of-160 zero-padded transition as the main architectural diagnostic, and base causal interpretation on switch/first-weak/NLL trajectories rather than a bare ten-example pass.

One critic concern was factually rejected: the 1.23324x nine-gate SE timing result is not misattributed. It is explicitly recorded in `experiments/012/00-se-timing.md`, `experiments/012/03-execute.md`, and `experiments/012/04-analysis.md`; EXP-012's scored fallback was preactivation only because SE failed that gate. This correction does not change the winning pick because ECA still supplies conditional reallocation rather than the static added capacity supported by EXP-007, and its tight launch-overhead gate remains a high no-go risk. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the critic's winner, **Depth-Preserving Final-Stage Widening to 160**. It preserves the exact depth that EXP-023 implicated, spends new compute only at 8x8, and turns the strongest historical architecture signal—width—into a lower-cost asymmetric allocation. ECA remains the best fallback if static late capacity fails with healthy transition behavior; bounded GeM remains interesting but has insufficient direct evidence that its deliberately small contribution can clear the 0.10-point gate.

## Chosen Idea
**Selected**: Depth-Preserving Final-Stage Widening to 160

**Why this idea**:
It is the cleanest actionable consequence of EXP-023: preserve all nine accepted blocks while allocating 25% more channels only to the final 8x8 stage. EXP-007 gives strong local evidence that extra width can beat a substantial exposure penalty, while EXP-023 shows H20 width scaling is sublinear and that deleted depth—not immediate optimization collapse—was the likely bound. Compared with the other finalists, this candidate has both a larger representational effect and a clearer local evidence chain. The new 64-to-160 Option-A transition is a genuine confound, so immutable-corpus safety, paired timing, and first-weak trajectory checks are mandatory.

**Hypothesis**:
An FP32 postactivation ResNet-20 with stage widths 32/64/160, unchanged accepted training recipe, and at least 24,000 projected updates will preserve healthy strong-phase fit and use low-resolution semantic capacity to reach 94.25-94.30% `best_test_acc`. A weak first-tail conversion after an otherwise healthy switch would implicate the diluted 64-of-160 identity path; a healthy transition followed by a lower peak would reject this static late-width allocation as a generalization improvement.

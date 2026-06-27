# Brainstorm EXP-064
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **ACNet: Asymmetric Convolution Blocks** (knowledge/papers/acnet-structural-reparameterization.md; arXiv 1908.03930, ICCV 2019; impl https://github.com/DingXiaoH/ACNet)
  Train-time parallel 3x3 ∥ 1x3 ∥ 3x1 branches (each with its own BN, summed before ReLU) fold EXACTLY into one 3x3 conv at eval. CIFAR-10 fixed-epoch gains +0.35–1.11 across VGG/ResNet-56/WRN-16-8/DenseNet-40 — gains hold on a WIDE ResNet (WRN-16-8), the closest published analog to our 4x-wide net. Mechanism is optimization-geometry (per-branch BN gives the optimizer adaptive scales for kernel skeleton vs corners), not regularization and not eval-time capacity. The same group's DBB (CVPR 2021) reproduces the family mechanism with different branch sets.
- **Knowledge base re-scan** (knowledge/README.md): no prior reparameterization entry anywhere in the 64-experiment record — this axis is genuinely unmeasured.

## Experimental History Review

State after EXP-063 (57 consecutive closures; baseline 96.71 @ 1990397, family mean 96.57, σ 0.16, bar 96.81):

- **Multiplicity closed at the FUNDING level** (EXP-063): steps (−0.9 dilution), kernel width (2.8×), and concurrency (two streams serialize, ratio 1.81 — the binding resource is the serial kernel-dispatch chain; idle SMs are unreachable). Function-space gain (+0.3–0.5) is real but unaffordable in every currency.
- **Closed axes** (do not revisit): heat (both directions), noise (level + schedule), loss (per-sample/target-dist/logit-scale/fc-WD), numerics (fp16, faster-but-different arithmetic EXP-021), weight averaging (EMA/SWA/schedule-free), reg dose AND type, allocation (stage depths both directions EXP-017/061), depth ("closed both ways" EXP-034: 12-block nets launch-bound ~+2.8ms/block AND converge to a LOWER plateau 96.01), width lattice {64,128,256} with the >256 cliff, BN constants, tail distribution (both data and parameter currencies), step-time engineering (EXP-021/048: charged step already overhead-free; extra steps at plateau add nothing — EXP-048's null).
- **Critical structural fact from EXP-048**: numerics-identical de-overheading delivered +87 steps and read the mean EXACTLY — extra steps at the plateau are worthless. Any winning idea must raise the PLATEAU LEVEL, not buy more steps.
- **What raises plateau level and is unmeasured**: the record has tested capacity, regularization, loss geometry, averaging, schedule, and ensembling as plateau-raisers — all closed. Train-time over-parameterization with eval-time identity (structural reparameterization) is absent from the record and is mechanistically distinct from all six: eval capacity unchanged, train loss typically drops (not a regularizer), single model (not multiplicity), anneal untouched (not schedule).
- **Pricing law for the toll** (EXP-034/040/044/045): extra small kernels price at LAUNCHES, not FLOPs, on this box — an ACB adds 2 convs + 2 BNs per site; with 19 sites (18 block convs + stem) the dt toll is the launch count, only measurable by probe. The ~90s probe with a pre-registered inequality (EXP-063 pattern) prices it at zero charged cost.

## Candidate Ideas

### 1. ACNet asymmetric convolution blocks (structural reparameterization), probe-gated
**Summary**: Replace all 19 3x3 convs (18 block convs + stem) with ACBs: `out = BN_a(conv3x3(x)) + BN_b(conv1x3(x)) + BN_c(conv3x1(x))`, summed before the existing ReLU; the block's standalone BN is absorbed into the per-branch BNs (paper-faithful: each branch carries its own BN, no second BN after the sum). All recipe constants byte-identical. Eval the branched module directly — in eval mode it is mathematically identical to the folded plain net (BN-eval is affine; fold is exact algebra), so Eval.evaluate is untouched and the trained system is evaluated as-is; no folding machinery needed. Launch gated by an uncharged GPU probe with a pre-registered dt inequality: epochs(P) = 300000/P/97.65; dilution ≈ 0.014/ep × (139 − epochs(P)); LAUNCH only if the published-gain band minus dilution clears the +0.24 bar-over-mean requirement with margin.

**Reasoning**: The only unmeasured plateau-raising mechanism class left. Published +0.35–1.11 on CIFAR-10 at fixed epochs including WRN-16-8 (wide ResNet — our regime's closest analog); mechanism (per-branch adaptive scaling of kernel skeleton vs corners) is something heavy augmentation cannot supply, surviving the absorption screen's mechanism test. Evades the capacity closure (eval params unchanged) and the reg-dose closure (not a regularizer). Worst case is the familiar pair: probe NO-LAUNCH (zero charged cost) or a measured no-improvement closing the reparameterization axis.

**Sources**: knowledge/papers/acnet-structural-reparameterization.md; EXP-034/040 launch-pricing; EXP-048 plateau null; EXP-063 probe-gate pattern; goal-learnings § Failed Approaches (no reparam entry).

**Estimated Effort**: medium (ACB module + sanity checks + probe + composite run).

**Risk Assessment**: (a) dt toll: +38 small kernels/step on a launch-bound box could be +4–8ms → NO-LAUNCH (acceptable: cost-closure at zero charge). (b) torch.compile may fuse the 1D convs poorly or well — probe measures truth. (c) Absorption: published baselines use light aug at the 94–95% level; 0-for-18 transfer prior — but every remaining idea faces this, and the mechanism argument is stronger than CutMix/Schedule-Free had. (d) BN-per-branch triples BN params/stats — VRAM trivial, init standard.

### 2. Augmentation-strength ramp (progressive regularization, EfficientNetV2-style)
**Summary**: Disable TrivialAugment + RandomErasing during the warmup phase (first ~15% of budget), enable at full strength after — a time-profile change at constant endpoint pressure. dt unchanged (aug is CPU-side; charged step is GPU-bound). Implemented via a probability gate keyed to elapsed-budget fraction passed into the transform pipeline (epoch-boundary switch to respect persistent workers).

**Reasoning**: The unmeasured MIRROR of the tail-lightening closures (EXP-025/033 lightened the tail and lost; this lightens the HEAD while keeping full tail pressure — consistent with the two-sided tail-pressure law). EfficientNetV2's progressive learning validates aug ramping at scale. High-LR early phase needs less regularization (reg matters most near convergence); easing early aug banks faster early progress that the anneal compounds.

**Sources**: EXP-025/033 (tail side measured); EXP-031 (the RESIZE half of progressive learning measured ZERO conversion on CIFAR — bad omen for the family); EfficientNetV2 (arXiv 2104.00298).

**Estimated Effort**: low.

**Risk Assessment**: Effect size marginal (+0.1–0.3 at best — EfficientNetV2's gains are throughput-at-scale, and the resize half already read zero here); reg-dose axis is peaked AT the recipe so the early-phase dose drop may simply under-regularize the most plastic phase (EXP-018's inverted-init precedent: early-phase help must ADD learning, and weak-aug data is also lower-information). Likely family-band null.

### 3. Second literature excavation round (budget-creating mechanisms)
**Summary**: Another /lit-search sweep targeting 2024–2026 work, explicitly filtered to mechanisms that are budget-creating or parameterization-side (the EXP-062 excavation pattern), feeding a future loop.

**Reasoning**: Excavations have produced clean closures (Schedule-Free, CutMix, Smith et al.), but the transfer record is 0-for-18 and the highest-value unmeasured axis (reparameterization) is already in hand as Candidate 1 — searching before measuring it would defer the best available experiment.

**Sources**: knowledge/README.md; project-insights absorption law.

**Estimated Effort**: low (but produces no measurement this loop).

**Risk Assessment**: Opportunity cost only.

## Idea Evaluation

**Evidence strength**: Candidate 1 dominates — peer-reviewed ICCV results on CIFAR-10 spanning four architectures including a wide ResNet, reproduced as a family by DBB/RepVGG (three CVPR/ICCV papers, same mechanism). Candidate 2's nearest evidence half (progressive resizing) measured ZERO conversion here (EXP-031), and its effect size fails the +0.3 screen. Candidate 3 produces no measurement.

**Mechanism clarity**: Candidate 1's mechanism is precise: per-branch BNs give SGD independently adaptive learning-rate-like scales for the kernel skeleton vs corners — a reparameterization of optimization geometry that is orthogonal to every closed axis (capacity at eval unchanged, not a regularizer, anneal untouched, single model). Candidate 2's mechanism ("ease early, bank progress") is the vaguest of the three and brushes the peaked reg-dose axis.

**Expected impact**: Candidate 1's published band (+0.35–1.11) minus realistic dilution (−0.2–0.4 at a 24.5–26.5ms probe read) leaves net +0.1–0.9 — the only candidate whose mid-estimate clears the bar. Candidate 2's ceiling is ~+0.3 before absorption compression.

**Risk profile**: Candidate 1 fails gracefully twice over: the probe NO-LAUNCH branch costs zero charged seconds (EXP-063 pattern), and a launched null closes the last unmeasured plateau-raising axis — high information either way. Its worst outcome equals Candidate 2's expected outcome.

**Feasibility**: ACB is ~30 lines (module + integration); the probe/composite/verification machinery is all reusable from EXP-063/061.

## Chosen Idea
**Selected**: ACNet asymmetric convolution blocks (structural reparameterization), probe-gated

**Why this idea**:
It is the only remaining unmeasured mechanism class that raises the plateau level (the binding constraint per EXP-048's extra-steps null), it carries the strongest external evidence of any untested candidate (+0.35–1.11 on CIFAR-10 at fixed epochs, including a wide ResNet), and its failure modes are the cheap kind: an uncharged probe NO-LAUNCH or an axis-closing null. The launch decision is made BEFORE spending charged time via a pre-registered dt inequality, per the validated EXP-063 gate pattern and the standing launch-pricing law (EXP-034/040: small kernels price at launches, not FLOPs).

**Hypothesis**:
Replacing all 19 3x3 convs with ACBs (3x3 ∥ 1x3 ∥ 3x1, per-branch BN, summed pre-ReLU) raises best_test_acc above the 96.81 bar: the probe will read dt ≤ ~26ms (epochs ≥ 118, dilution ≤ ~0.29), and the reparameterized optimization geometry will deliver ≥ +0.5 of its published +0.35–1.11 band over the family mean — net ≥ +0.24 over mean, i.e., pair-mean ≥ 96.81 under the EXP-052 replicate protocol. If the probe prices the branches above the inequality, the pre-registered NO-LAUNCH branch closes the reparameterization axis on cost grounds at zero charged seconds.

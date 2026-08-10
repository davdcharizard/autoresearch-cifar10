# Brainstorm EXP-005
**Created**: 2026-08-05

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention** (`papers/eca-net.md`)
  A global-pooling plus short channel-wise 1D convolution can add image-conditioned channel selection to residual blocks at negligible parameter and FLOP cost. The paper reports strong ImageNet residual-backbone gains, but no matched CIFAR-10 result.
- **A Unified Analysis of Mixed Sample Data Augmentation** (`papers/hmix.md`)
  HMix interpolates between Mixup and CutMix by using a smaller pure pasted box and global interpolation outside it. It modestly but consistently improves over CutMix across several CIFAR-100 backbones with negligible arithmetic overhead.
- **Self-Distillation from the Last Mini-Batch** (`papers/dlb.md`)
  Half-overlapping consecutive batches allow cached detached logits to supervise a fresh view one update later with no extra model forward. The paper reports 0.37-1.01-point CIFAR-10 gains and direct compatibility gains when composed with CutMix.
- **Existing goal knowledge** (`knowledge/README.md`)
  Phase-dependent regularization, mixed-sample analysis, SWA, SAM, label smoothing, ShakeDrop, and prior goal results were reviewed before selecting new directions.

## Experimental History Review

- The accepted lineage is BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23% -> EXP-004 95.40%. EXP-004 is the sole successful branch tip and requires at least 95.50% for a new improvement.
- EXP-001 established the BF16 time-scheduled PreAct WRN-16-4. EXP-002 added front-loaded probabilistic CutMix for +0.61 points. EXP-004 added period-two SAM only in the final clean quarter for +0.17 points.
- EXP-003's CutMix/drop-path sweep produced selected search gains that fell by 0.14-0.29 points in confirmation. Narrow scalar tuning around the accepted regularization stack is locally saturated and too vulnerable to selection noise.
- EXP-004 ends at its best checkpoint with 0.1654 test loss, 25,560 optimizer steps, 2.75M parameters, and only 1,190.5 MiB peak allocation. Generalization is the limiter; memory is not, and expensive extra passes must justify lost optimizer exposure.
- The highest-value gap is an orthogonal, low-cost representation or consistency mechanism that can plausibly move more than 0.10 points without disturbing the validated CutMix/SAM phase boundary.

## Collected Ideas

- **Last-mini-batch self-distillation:** repeat half of each raw-index batch on the next step with a fresh crop/flip, then train those examples against detached temperature-three logits from one update earlier. This directly targets prediction instability and has matched CIFAR-10 plus CutMix evidence, but reduces the rate of new unique images and needs identity-safe gating around mixed batches.
- **Identity-initialized ECA residual gates:** place a three-tap channel-attention kernel in each residual branch with `2*sigmoid` and a zero kernel so the initial parent function and shared RNG draw order are preserved exactly. It uses representation capacity rather than more regularization and costs only 18 parameters, though the neutral gate is an adaptation of the published ECA design.
- **Front-loaded HMix:** preserve the accepted mixing probability and cutoff, but replace selected CutMix batches with a fixed `r=0.5` hybrid whose outside region is globally interpolated. It attacks input-gradient structure rather than scalar strength, yet remains close to a mechanism already tested extensively in EXP-003.
- **Stochastic Mixup/CutMix dual policy:** split the existing early mixed-batch gate between the two augmentation types while keeping total mixed exposure at 0.5 and the final quarter clean. This is simpler than HMix and externally supported, but adds a policy choice whose likely effect is close to the observed noise floor.
- **RecursiveMix history replay:** paste a resized prior input into the current image and add a prior-prediction consistency term. This combines spatial and temporal regularization with negligible state, but is more complex than DLB and overlaps awkwardly with current CutMix.
- **LookSAM-style reused sharpness direction:** compute occasional full SAM gradients and reuse a decomposed sharpness-aware component on intervening updates. This could strengthen the successful EXP-004 mechanism without another second pass each time, but it is a larger optimizer departure and risks stale gradient geometry under momentum.
- **Late-stage width reallocation:** spend H20 headroom on wider 128/256-channel stages while narrowing the first stage to control convolution cost. This is a representation-focused architectural bet, but changes parameter initialization and throughput broadly without a strong matched recipe.
- **Remove stochastic depth under attention:** simplify the stack by replacing drop path with ECA, reasoning that CutMix plus SAM may already supply sufficient regularization. This could regain fit and throughput, but EXP-003 did not establish that lower drop path is beneficial and bundling removal with attention weakens attribution.
- **Moonshot compact ConvNeXt-style blocks:** replace the six preactivation residual blocks with depthwise large-kernel blocks and pointwise expansion to spend memory on a modern representation. It offers a high ceiling but abandons the validated architecture and is too risky for a single 300-second fixed-seed experiment without profiling.

## Combinations

- **ECA + DLB:** ECA improves channel-selective representation while DLB stabilizes predictions across successive augmented views. The mechanisms are complementary, but combining two unvalidated changes would make a failure uninterpretable and DLB's data-order change is already high-risk.
- **HMix + DLB:** the paper reports DLB compatibility with CutMix, while HMix supplies richer mixed-sample geometry. Applying DLB only across clean-clean transitions keeps targets well-defined, but leaves DLB sparse during the early phase and bundles two data-path interventions.
- **ECA + HMix:** HMix could produce richer part/global features for ECA to select, while ECA costs little enough to preserve the mixed-sample step horizon. Both are individually low-cost, but the stricter 95.50% gate does not justify sacrificing attribution before either mechanism succeeds alone.
- **ECA + reused sharpness direction:** channel attention changes representation, while more frequent approximate sharpness steps could regularize the new gates and backbone. This may be stronger than either alone, but optimizer correctness and architecture validation should be staged separately.

## Candidate Ideas

### Clean-Gated Last-Mini-Batch Self-Distillation
**Summary**: Emit half-overlapping batches of raw sample indices, obtain a fresh crop/flip for each repeated occurrence, and add the paper's `tau=3`, `alpha=1` teacher-to-student KL only on aligned clean-clean transitions. Mixed batches retain the exact parent CutMix objective and invalidate the cache; clean-tail SAM recomputes the same CE+DLB objective on its perturbed pass but publishes only primary-forward logits. Full design: `proposals/idea-03.md`.

**Reasoning**: DLB has the strongest matched evidence: 0.37-1.01-point CIFAR-10 gains across backbones and additive gains with CutMix, including 0.60 points on WRN-20-8. It adds only a tiny logits KL and no model pass, directly targeting temporal/view consistency rather than revisiting regularization strength. Its main cost is statistical: half-overlap roughly halves new unique-image introduction and epoch-end evaluation frequency, while the clean gate limits early DLB coverage to roughly clean-clean transitions.

**Sources**: `papers/dlb.md`; `proposals/idea-03.md`; EXP-004 analysis.

**Estimated Effort**: high

**Risk Assessment**: Sample/logit misalignment is a silent correctness hazard, CutMix suppresses much of early DLB, repeated examples reduce data diversity, and applying DLB inside both SAM passes may over-regularize the clean tail. Runtime identity assertions, cache-transition counters, and fixed paper hyperparameters bound but do not remove those risks.

### Front-Loaded HMix with the Validated SAM Tail
**Summary**: Replace only selected early CutMix masks with fixed-`r=0.5` HMix, using a smaller pure paired-image box and global interpolation outside it. Preserve the 0.5 gate, uniform lambda stream, 75% cutoff, clean batches, architecture, and SAM tail; compute label weights from the exact clipped mask and blend coefficients. Full design: `proposals/idea-02.md`.

**Reasoning**: HMix provides a mechanistically different input-gradient regularizer with negligible extra compute. It improves over CutMix across CIFAR-100 residual backbones and preserves the parent's successful early-mixing/clean-tail schedule without adding another forward or stochastic decision. Unlike EXP-003, it changes mask structure rather than tuning CutMix probability or drop path.

**Sources**: `papers/hmix.md`; `proposals/idea-02.md`; EXP-003 and EXP-004 analyses.

**Estimated Effort**: low to medium

**Risk Assessment**: Evidence transfer from CIFAR-100 is uncertain, global interpolation may over-regularize CIFAR-10, and the expected 0.10-0.30-point effect overlaps EXP-003's observed selection variability. Incorrect clipping or source aliasing would silently mismatch image and label mixtures.

### Identity-Initialized ECA Residual Gates
**Summary**: Add a fixed three-tap ECA kernel to every residual branch after the second convolution and before stochastic depth. Use zero-initialized bare parameters and `2*sigmoid` gates, yielding exact unit gates at initialization, no extra RNG consumption, and only 18 added parameters. Preserve every CutMix, drop-path, SAM, optimizer, and timing setting. Full design: `proposals/idea-01.md`.

**Reasoning**: ECA is an orthogonal representation change with negligible theoretical compute and ample published evidence on residual backbones. Identity initialization makes the experiment a strict parent extension at step zero and avoids confounding attention with a 0.5 residual-amplitude reduction. It attacks the remaining generalization gap through learned channel interaction while retaining nearly the full optimizer horizon.

**Sources**: `papers/eca-net.md`; `proposals/idea-01.md`; EXP-004 analysis.

**Estimated Effort**: medium

**Risk Assessment**: Direct evidence does not cover this CIFAR-10 regime; six tiny kernels may be launch-bound despite low FLOPs; local channel adjacency may be weak; and the identity-preserving `2*sigmoid` gate differs from the paper. The likely effect may be near the 0.10-point gate.

## Review

Claude's cross-model review selected clean-gated DLB and found no hard-constraint, reward-hacking, or unchanged-retry issue. It judged DLB the only candidate with matched CIFAR-10 and CutMix-composition evidence at a magnitude likely to survive heavy transfer discounting. Three mandatory refinements are adopted: preserve the natural 389-step DLB epoch and explicitly prohibit cadence-oriented epoch redefinition; acknowledge that the fixed seed-42 overlapping permutation changes parent data order; and diagnose flat accuracy with final loss worse than 0.1654 as a likely over-regularized CE+KL SAM tail. The expected effect is discounted from the paper's +0.60 WRN/CutMix anchor to 0.10-0.30 points because early clean-clean gating supplies only about 25% coverage and the parent is already strongly regularized. Full review: `01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. DLB scored 4/5 for both evidence/reasoning and potential impact. HMix scored 3/5 and 2/5: exact math and RNG hygiene are strong, but the effect is likely trapped in EXP-003's saturation/noise region. ECA scored 2/5 on both dimensions because its identity gate departs from the ImageNet evidence and its 18 weight-decayed parameters may remain near identity. Under the preregistered single-run protocol, DLB offers the highest expected true effect despite the largest implementation burden.

## Chosen Idea
**Selected**: Clean-Gated Last-Mini-Batch Self-Distillation

**Why this idea**:
DLB directly targets temporal and augmentation-view prediction consistency with no extra model forward. It has the strongest relevant evidence, including a 0.60-point CutMix-composition gain on a CIFAR-10 WRN, while HMix and ECA are more likely to produce real effects below the 0.10-point acceptance gate. The implementation will keep the paper's `tau=3`, `alpha=1`, fixed seed 42, natural half-overlap epoch, clean-only cache semantics, and full CE+KL SAM objective.

**Hypothesis**:
Clean-gated DLB will retain at least 24,500 charged optimizer steps and improve EXP-004's 95.40% `best_test_acc` to at least 95.50% in one fixed-seed run on physical GPU 0. After discounting the paper's stronger result for partial early coverage and the already-regularized parent, the expected gain is 0.10-0.30 points. The sampler's changed order is an explicit confound; epochs will not be redefined to increase evaluation count; and flat accuracy with final loss worse than 0.1654 will be interpreted as an over-regularized DLB+SAM tail rather than grounds for a retry.

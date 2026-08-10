# Brainstorm EXP-010
**Created**: 2026-08-05

## Web Search & Literature Review

- **Deep Pyramidal Residual Networks** (`experiments/010/papers/deep-pyramidal-residual-networks.md`)
  Gradual rather than abrupt channel growth improved CIFAR residual-network generalization, making macro-level channel allocation a credible lever.
- **Aggregated Residual Transformations for Deep Neural Networks** (`experiments/010/papers/aggregated-residual-transformations.md`)
  Cardinality can outperform added depth or width at matched complexity, but grouped-convolution efficiency must be measured on this compact workload.
- **Gradually Updated Neural Networks for Large-Scale Image Recognition** (`experiments/010/papers/gradually-updated-neural-networks.md`)
  Changing how channels are computed can increase effective representational depth without raising nominal computation, supporting existing-convolution-path ideas.
- **Residual-network scaling evidence** (official ICLR 2024 OpenReview abstract: https://openreview.net/forum?id=6pfCFDPhy6)
  Joint depth/width changes alter optimization parameterization; a depth change cannot assume the parent's learning-rate optimum transfers automatically.

## Experimental History Review

- BASE reached 91.51; EXP-001's time-aware PreAct WRN-16-4 made the dominant +3.11-point gain; EXP-002's front-loaded CutMix added +0.61 to reach the chosen parent at 95.23.
- EXP-004 added +0.17 with periodic late SAM and remains the 95.40 global best, but its four subsequent children failed to improve and its two-pass dose becomes a confounder when throughput changes.
- From EXP-002, EXP-003's regularization sweep did not confirm and EXP-009's four SE gates were rejected before accuracy because they added 20.7% median latency. EXP-009 specifically leaves macro architecture reallocation untested.
- The current limiter is a detectable generalization gain near 0.3 points, not memory: the parent family uses roughly 1.2% of H20 memory while single-run tail variation is at least 0.15 points (`02-system-understanding.md`; `03-experiment-learnings.md`).
- Untried gaps include depth/width scaling, stagewise block allocation, gradual width schedules, grouped residual transformations, and cheap changes to residual-block parameterization. Every finalist must preserve the EXP-002 data/optimizer semantics and pass a same-harness parent-relative latency gate on physical GPU 0.

## Collected Ideas

- **Back-loaded 1-2-3 stage depth** - Move one residual block from the 32x32/64-channel stage to the 8x8/256-channel stage, retaining six blocks and approximately equal convolutional FLOPs while allocating about one million more parameters to high-level features. This directly targets representation capacity with the existing convolution path and should preserve step exposure.
- **Deeper WRN-22-4** - Increase each stage from two to three blocks at the existing widths. Standard WRN scaling suggests meaningful accuracy upside, but nine blocks will lower the number of optimizer updates under 300 seconds; a latency gate must show enough exposure remains.
- **Wider WRN-16-5** - Change stage widths from 64/128/256 to 80/160/320 while retaining six blocks. Width is a proven CIFAR lever and H20 memory is abundant, though dense convolution cost grows roughly quadratically and the inherited learning rate may be suboptimal.
- **Gradual six-block pyramid** - Replace abrupt 64/128/256 stage widths with monotonically growing widths across all six blocks. It imports PyramidNet's generalization principle, but changing dimensions at every block adds shortcut projections and risks repeating EXP-009's small-kernel overhead failure.
- **ResNeXt-style grouped bottlenecks** - Replace dense two-convolution residual branches with 1x1/grouped-3x3/1x1 transformations at matched measured latency. It introduces cardinality as a new representation axis, but changes three coupled properties and grouped kernels may be inefficient at CIFAR feature sizes.
- **Final-stage width-only expansion** - Keep the first two stages at 64/128 and widen only the last stage to 320 or 384. It spends capacity where spatial cost is lowest, but the transition and last two blocks still increase FLOPs and the abrupt bottleneck may limit use of the added channels.
- **Residual branch scaling and zero-gamma start** - Identity-center the existing residual blocks via a learned or fixed branch scale and zero initialization, without extra forward kernels. It targets optimization stability, but batch normalization already stabilizes this shallow network and a cold residual path may waste the short time budget.
- **Variance-aware classifier head** - Concatenate global channel mean and standard deviation before the classifier, adding almost no parameters or convolutions. It offers an orthogonal texture/statistics signal, but extra reductions add launches and the likely effect is below the required detectable margin.
- **Moonshot: compute-neutral channel ordering** - Implement a gradually updated channel-group convolution inside each block, using structured partial reuse to increase effective depth at nominally similar FLOPs. The literature reports CIFAR gains, but custom slicing/concatenation kernels create high implementation and latency risk.

## Combinations

- **Back-loaded depth + modest final width**: use a 1-2-3 block allocation with a smaller final width increase (for example 288 channels). The reallocation preserves major-kernel count while the width increase adds capacity at the cheapest resolution, plausibly offering more upside than either a pure topology move or full 1.25x widening, but it introduces two variables and is unsuitable for the first isolation test.
- **Deeper WRN + depth-aware residual scaling**: combine WRN-22-4 with a fixed residual scale to stabilize the additional blocks. This could reduce optimization degradation from depth, but the parent's BN PreAct design is already stable and the scale adds an unvalidated hyperparameter, so depth alone is cleaner.
- **Back-loaded depth + clean-finish SAM later**: first establish whether 1-2-3 improves EXP-002; if successful, add EXP-004's SAM as a child. Sequential testing is stronger than bundling because architecture throughput changes alter the number of SAM pulses.

## Candidate Ideas

### Wider WRN-16-5
**Summary**: Widen the six-block network from 64/128/256 to 80/160/320 channels, retaining its block count, operator topology, and EXP-002 recipe. The candidate has 4,289,754 parameters and reuses the same 16-aligned dense convolution path without adding attention, grouped kernels, or projections.

**What it targets**: It attacks representation capacity using abundant H20 memory while preserving the shallow optimization path that produced EXP-001's dominant gain.

**Reasoning**: Wider residual networks are a strong CIFAR prior, and the current model occupies only about 1.2% of GPU memory. Yet 609,930,368 MACs/image are 1.554x the parent, predicting only 18,000-20,500 steps. The parent-relative preflight requires <=1.60x median latency and >=17,500 projected steps, with width fixed before accuracy.

**Sources**: `experiments/010/proposals/idea-03.md`; EXP-001/EXP-002 reports; `experiments/010/papers/aggregated-residual-transformations.md`.

**Estimated Effort**: low implementation, medium verification.

**Risk Assessment**: Fewer unique views and evaluation opportunities can outweigh capacity; width 80 kernels may be less efficient than powers of two; inherited LR and decay may not be width-optimal; extra width may overfit a capacity-sufficient parent.

### Back-loaded 1-2-3 stage depth
**Summary**: Reallocate the parent's six residual blocks from 2-2-2 to 1-2-3 across widths 64/128/256. This removes one same-width 64-channel block before the first downsample and adds one 256-channel block after the last downsample, retaining twelve block 3x3 convolutions, three projections, and exactly 392,612,352 Conv/Linear MACs per image. Parameters rise from 2,748,890 to 3,855,578, and the final receptive field grows from roughly 53 to 65 pixels.

**What it targets**: The measured limiter is a detectable generalization gain rather than memory or raw throughput (`02-system-understanding.md`). The candidate spends equal major-convolution work on more class-specific late features while reducing early BN/ReLU activation traffic.

**Reasoning**: EXP-009 recommends an existing-convolution representation change after multi-launch SE cost 20.7% latency. PyramidNet makes channel allocation a demonstrated CIFAR lever, while this version avoids its per-block projection overhead. Equal MACs do not prove equal latency, so the proposal requires a same-harness GPU-0 gate at <=1.05x parent median and >=26,500 projected steps.

**Sources**: `experiments/010/proposals/idea-01.md`; `experiments/010/papers/deep-pyramidal-residual-networks.md`; EXP-002 and EXP-009 reports.

**Estimated Effort**: low implementation, medium verification.

**Risk Assessment**: Removing early spatial processing may irreversibly weaken local features; extra late weights may overfit; 8x8 dense kernels may be slower despite equal MACs. The exact allocation has no direct literature validation and observed metric noise can obscure a marginal effect.

### Deeper WRN-22-4
**Summary**: Increase each stage from two to three residual blocks while retaining widths 64/128/256 and the complete EXP-002 CutMix recipe. The standard 3-3-3 layout has nine blocks and 4,298,970 parameters, adding one dense two-convolution transformation per spatial scale with no auxiliary operator family.

**What it targets**: It attacks representational depth, betting that more nonlinear transformations improve class boundaries enough to overcome fewer samples and optimizer updates under the fixed 300-second budget.

**Reasoning**: Residual depth is a proven capacity axis and preactivation shortcuts make the implementation straightforward. However, its 619,104,768 MACs/image are 1.577x the parent and plausibly reduce exposure from 27,950 to 17,700-19,300 steps. A parent-relative GPU-0 preflight therefore requires <=1.50x median latency and >=18,500 projected steps before accuracy is tested.

**Sources**: `experiments/010/proposals/idea-02.md`; official ICLR 2024 residual scaling abstract; `experiments/010/papers/deep-pyramidal-residual-networks.md`.

**Estimated Effort**: low implementation, medium verification.

**Risk Assessment**: The model may be undertrained in roughly 95 epochs, inherited LR/drop-path settings may not transfer, and near-zero parent training loss weakens a raw-capacity explanation. A result is package-level because depth, RNG use, throughput, and per-block drop rates all change.

## Review

Claude verified every candidate's parameter/MAC arithmetic and found no hard-constraint, reward-hacking, or exact-retry issue. It identified three material refinements for the selected 1-2-3 proposal: ground the selection in preserved compute/exposure rather than the unsupported 95.53-95.80 forecast; log early training trajectory so local-feature starvation is diagnosable; and state that moving blocks also redistributes per-position drop-path dose, making attribution package-level. Those concerns are adopted. The full review is in `experiments/010/01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. It scored 1-2-3 at 8/10 evidence and 7/10 impact, ahead of WRN-16-5 at 6/10 and 6.5/10 and WRN-22-4 at 5/10 and 6/10. Width and depth have plausible raw capacity upside, but both must recover a roughly one-third loss of optimizer/data exposure and about forty max-over evaluation opportunities. The selected reallocation instead tests an untried representation axis with exact parent MACs and a genuinely binding 1.05x latency gate.

## Chosen Idea
**Selected**: Back-loaded 1-2-3 stage depth

**Why this idea**:
It gives the cleanest high-value test of architecture allocation under the fixed wall-clock protocol: six blocks, twelve block 3x3 convolutions, three projections, and 392,612,352 MACs/image are preserved while about 1.1M parameters and 12 pixels of receptive-field diameter move to late semantic processing. Its case is bounded downside and preserved exposure, not a claimed literature-proven effect size. The plan must retain the <=1.05x parent-median gate, record early loss/accuracy behavior, and interpret any result as the full allocation plus redistributed drop-path package.

**Hypothesis**:
On one fixed-seed GPU-0 run, the 1-2-3 allocation will pass the parent-relative latency gate, retain at least 26,500 optimizer steps, and reach at least 95.53% best test accuracy (+0.30 points over EXP-002 and +0.13 over the current global best) because equal compute is spent on a larger late-stage representation. A valid 95.33-95.52 is a formal tree improvement but falsifies the preregistered detectable-effect hypothesis; below 95.33 or a preflight failure is no improvement.

# Brainstorm EXP-009
**Created**: 2026-08-05

## Web Search & Literature Review

- **ECA-Net** (`experiments/008/papers/eca-net.md`)
  Local 1D channel interaction adds only a handful of parameters and negligible published FLOPs, but published gains are calibrated on deeper ImageNet residual models.
- **Squeeze-and-Excitation Networks** (`experiments/009/papers/se-net.md`)
  Bottleneck channel excitation has strong residual-backbone evidence but more compute than ECA and requires an identity-preserving adaptation for this short run.
- **Claude's prior ECA audit** (`experiments/008/01-idea-review.md`)
  The raw zero kernel, identity-centered gate, RNG neutrality, fixed kernel sizes, and gradient path are sound; descriptor normalization and weight-decay treatment need explicit resolution.

## Experimental History Review

- The chosen base lineage is BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23%. WRN-16-4 and front-loaded CutMix are validated, and EXP-002 retains 27,950 steps with 1,178.9 MiB peak VRAM.
- EXP-002 has one failed scalar-regularization child (EXP-003) and one successful orthogonal optimizer child (EXP-004, +0.17 SAM). ECA is a different representation mechanism; a successful fork can later compose the validated SAM package.
- Four distinct children at EXP-004 failed or stopped, motivating this interior fork rather than a fifth tip sibling.
- EXP-008 showed the CPU input path has only modest headroom. EXP-009 must leave the DataLoader byte-for-byte unchanged and spend only small charged GPU work.
- The limiter remains detectable class-boundary generalization from a compact model. Memory is abundant, but the effect prior should approach 0.3 points because smaller deltas sit within observed protocol noise.

## Collected Ideas

## Combinations

## Candidate Ideas

### Late-Stage Identity-Centered ECA
**Summary**: Use the same normalized-descriptor, zero-kernel, `2*sigmoid`, zero-weight-decay design only in the four 128/256-channel blocks. This adds 20 parameters and leaves both 64-channel blocks bitwise parent-identical.

**What it targets**: High-level class-selective features where channels are wider and spatial maps smaller, reducing launch and reduction cost while avoiding low-level attention on mixed CutMix textures.

**Reasoning**: Later stages have more semantic channels and cheaper spatial reductions. The restriction is an a priori architectural choice, not a post-hoc placement search, and may improve effect-to-cost relative to all-block ECA.

**Sources**: `experiments/008/papers/eca-net.md`; `02-system-understanding.md`; prior Claude ECA audit in `experiments/008/01-idea-review.md`.

**Estimated Effort**: medium

**Risk Assessment**: Four gates may be too weak in a six-block network, and the claimed semantic-stage advantage is plausible rather than directly measured. It also sacrifices the paper's all-stage coverage.

### All-Block Identity-Centered ECA
**Summary**: Add an independent ECA gate to each of the six residual blocks. Derive its channel descriptor from the normalized nonnegative `relu(bn2(conv1(...)))` tensor, apply the gate to `conv2` output before drop path, and use a raw zero kernel with `2*sigmoid` so the parent function and RNG are exact at initialization. Kernel sizes `[3,3,5,5,5,5]` add 26 parameters; ECA kernels receive zero weight decay.

**What it targets**: Per-image feature selectivity across every residual scale, directly expanding representation behavior without removing images or changing the validated CutMix path.

**Reasoning**: This incorporates Claude's two prior corrections: a normalized descriptor and no identity-restoring weight decay. All-block coverage offers the highest effect ceiling, while raw zero parameters preserve every shared parent initialization draw.

**Sources**: `experiments/008/papers/eca-net.md`; `experiments/008/proposals/idea-01.md`; `experiments/008/01-idea-review.md`; `experiments/002/04-analysis.md`.

**Estimated Effort**: medium

**Risk Assessment**: Six reductions/Conv1d/sigmoid/multiply paths may be launch-bound; channel adjacency may be weak in a shallow model; ImageNet gains may compress below the local 0.30-point target. A parent-relative GPU latency gate must pass before launch.

### Late-Stage Identity-Centered SE
**Summary**: Add SE bottleneck gates to the four 128/256-channel residual branches with reduction 16. Initialize the final excitation matrix and bias to zero and use `2*sigmoid`, so gates start exactly one; construct zero tensors without RNG consumption and exclude only the zero-final excitation parameters from weight decay.

**What it targets**: Full cross-channel dependencies in high-level stages, with greater representational capacity than local ECA while still exploiting abundant memory.

**Reasoning**: SE is the canonical channel-recalibration mechanism and does not assume useful channel locality. Late-stage placement controls spatial cost, and identity initialization isolates learned recalibration from a residual-scale reset.

**Sources**: `experiments/009/papers/se-net.md`; `02-system-understanding.md`; `experiments/002/04-analysis.md`.

**Estimated Effort**: medium

**Risk Assessment**: The MLP adds more kernel launches and parameters; zero-final initialization initially blocks gradients into the first excitation layer until the final layer moves; reduction 16 is a transferred scalar; deeper ImageNet evidence may not transfer.

## Review

Claude selected late-stage identity-centered SE at 7/10 evidence and 8/10 impact, ahead of all-block ECA at 6/10 and 5/10 and late-stage ECA at 4/10 and 3/10. The decisive argument is that EXP-002 has abundant memory and tolerated 840 fewer steps while gaining 0.61 points, so ECA optimizes a nonbinding cost constraint while limiting channel-interaction capacity.

The significant corrections are adopted: raw SE parameters use a dedicated seed-42 CPU generator without touching global initialization; descriptor vectors are standardized per sample across channels and gate the same `conv2` output they describe; hidden width is fixed at 16; all gate parameters receive zero decay and a fixed 5x LR scale while every parent parameter remains exact; a 200-step GPU-0 smoke must exceed `max|gate-1|=0.02`; and feasibility is parent-relative. CutMix-induced diffuse gates are preregistered as the leading null mechanism. Full feedback is in `01-idea-review.md`.

## Idea Evaluation

The randomized Claude verdict is adopted. SE has the largest mechanism ceiling and avoids ECA's arbitrary channel-adjacency assumption. The chosen experiment remains the reviewed late-four-block package rather than conditionally expanding placement after a latency result; placement is fixed before execution and latency can only accept or reject it.

## Chosen Idea
**Selected**: Late-Stage Identity-Centered SE with Isolated Initialization

**Why this idea**:
Add full cross-channel recalibration to the four 128/256-channel residual outputs without altering the parent function, shared initialization, data stream, or evaluator. Each raw-parameter SE module standardizes its pooled channel descriptor, uses a 16-unit ReLU bottleneck, and applies `2*sigmoid` with a zero final matrix/bias. A dedicated generator initializes the first matrix, while the fixed gate optimizer group supplies 5x parent LR and zero decay to escape the BF16 identity dead zone. No placement, width, LR scale, or initialization will change after the live-gate and latency gates.

**Hypothesis**:
Four identity-centered SE gates in the 128/256-channel residual blocks will become functionally active within 200 fixed preflight steps, retain at least 26,000 optimizer steps, and improve EXP-002's `best_test_acc` from 95.23% to at least 95.53% in one fixed-seed physical-GPU-0 run. The formal tree threshold remains 95.33%; 95.33-95.52 is a formal improvement below the preregistered 0.30-point evidentiary target. The leading null mechanism is that early CutMix composites train diffuse rather than class-selective gates.

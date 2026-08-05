# Brainstorm EXP-009
**Created**: 2026-07-24

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): shallow width is an effective CIFAR capacity allocation, but the prior full width-3 preflight lost too much throughput.
- **RandAugment** (`knowledge/papers/randaugment.md`): a small shared-magnitude transform space improves CIFAR generalization and is available in torchvision, with CPU cost and strength calibration as the local risks.
- **Weight averaging** (`knowledge/papers/weight-averaging.md`): late averaging is cheap and can mildly improve generalization, but the accepted stable endpoint and BatchNorm-state mismatch limit headroom here.
- **Label smoothing** (`knowledge/papers/label-smoothing.md`): mild soft targets can reduce overconfidence, but stacking them with the already successful mixup risks repeating the observed over-regularization pattern.

No network search was performed because this session is explicitly offline. The accumulated local knowledge base covers the relevant architecture, augmentation, and averaging mechanisms.

## Experimental History Review

- EXP-001 and EXP-002 remain the only improvements: WRN-16-2 plus counted-time cosine reached 93.38%, then alpha-0.2 mixup through 65% reached the current 94.07% best with 141.9 passes.
- Six controlled follow-ups all preserved normal exposure but regressed: CutMix, shorter mixup, stronger mixup, block dropout, late decay removal, and cosine-to-zero. This fixes alpha 0.2, its 65% cutoff, continuous `5e-4` matrix decay, and the 0.002 LR floor for now.
- The prior WRN-16-3 preflight retained only 56.8% of width-2 throughput and projected about 80.6 passes, so uniform widening is compute-bound despite enormous memory headroom.
- The accepted FP32 WRN uses about 1.1 GiB on a 97.9-GiB H20 and processes roughly 24k images/s. Accuracy still improves through the late tail, while reducing late update magnitude worsens results. The strongest untested gap is therefore useful optimization/data exposure per counted second, followed by selectively better capacity or orthogonal augmentation.

## Collected Ideas

- **BF16 tensor-core training** — run forward and loss under CUDA BF16 autocast while keeping FP32 parameters, gradients, optimizer state, schedule, and batch 256. This targets low H20 utilization and could increase passes without the dynamic-loss-scaling complexity of FP16.
- **BF16 plus larger batches** — combine autocast with batch 512 or 1024 and a preregistered throughput/update-count gate. The H20 has vast memory headroom; more images per step may unlock tensor cores and data exposure, though batch-size-dependent optimization requires careful LR policy.
- **Fused SGD and channels-last layout** — retain numerical precision and batch 256 while using library-native fused optimizer kernels and NHWC memory format. This is a simplification/implementation lever with low scientific risk but probably smaller upside on a tiny model.
- **Selective final-stage widening** — widen only the 64/128-channel third residual stage and classifier rather than every stage. This spends capacity at the semantic end of the network while avoiding the rejected 2.24x uniform width-3 cost.
- **Low-magnitude RandAugment before tensor conversion** — add a single mild torchvision RandAugment operation during the mixup phase or throughout training. This imports a proven CIFAR augmentation family distinct from failed CutMix, but CPU transform time and stacked regularization are concrete risks.
- **Evaluator-consistent channel standardization** — divide mean-centered inputs by fixed CIFAR channel standard deviations inside the model. This is cheap and orthogonal, but immediate stem-output BatchNorm cancels much of its plausible benefit.
- **Late checkpoint averaging with explicit BN policy** — average weights only over a short terminal window and average or copy BN buffers without an uncounted recalibration pass. It targets iterate variance, but accepted final stability gives little evidence that variance is limiting.
- **Moonshot: compact ConvNeXt-style CIFAR blocks** — replace residual basic blocks with depthwise-separable blocks and modern normalization to exploit H20 throughput at greater effective capacity. Upside is substantial but implementation and optimization confounds are too broad for a first plateau probe.

## Combinations

- **BF16 autocast + batch 512**: autocast unlocks tensor-core arithmetic while a larger batch raises per-kernel work and reduces loader/launch overhead. Together they may increase passes more than either alone, but the proposal must separate feasibility preflight from the one scored configuration and preserve the time-based schedule.
- **BF16 autocast + selective final-stage widening**: recovered tensor-core throughput can pay for targeted semantic capacity, potentially retaining accepted exposure with more representational power. This combination is stronger than raw widening if BF16 offsets the extra convolution cost, but it introduces two treatment mechanisms and should follow isolated throughput evidence.
- **Mild RandAugment + accepted mixup cutoff**: diverse early invariances plus the validated 35% clean tail could outperform interpolation alone. The combination retains late unaugmented margin refinement, but it risks the additive-regularization mechanism already seen with dropout.

## Candidate Ideas

### Isolated BF16 Autocast at Batch 256
**Summary**: Keep FP32 master parameters, gradients, SGD state, mixup interpolation, batch 256, every accepted hyperparameter, and FP32 evaluation, but execute the training forward and cross-entropy under CUDA BF16 autocast. A local evaluator-free H20 benchmark measured 14.7% higher image throughput than FP32 at batch 256; BF16 batch 512 added only 1.5% more and is explicitly rejected to avoid batch/LR/BN confounds.

**What it targets**: The accepted model uses only about 1.1 GiB and processes roughly 24k images/s on an H20 while late hard-label updates remain useful. BF16 targets useful optimizer/data exposure per counted second rather than another regularization change.

**Reasoning**: Calibrating the measured 1.147x step ratio to EXP-002 projects about 162.8 passes and 4.1k additional updates within the same time-based LR curve. BF16 keeps FP32-like exponent range and needs no GradScaler; local smoke evidence found FP32 parameters/gradients/loss and BF16 logits. Autocast consumes no RNG and preserves the equal-length stochastic prefix.

**Sources**: `proposals/idea-01.md`; EXP-002, EXP-006, and EXP-008 analyses; local H20 proposal-development benchmark.

**Estimated Effort**: low.

**Risk Assessment**: BF16 activation rounding changes the trajectory and may outweigh additional exposure. Loader overhead may reduce the full-loop gain, and more updates at each time-based LR can over-optimize. A fresh evaluator-free production preflight must require at least 1.10x throughput before the single scored run.

### Low-Magnitude RandAugment
**Summary**: Insert exactly one torchvision RandAugment operation at magnitude 5/30 after crop/flip and before tensor conversion, using bilinear interpolation and mean-colored fill. Keep it active for the full run while preserving alpha-0.2 mixup through 65% and the accepted hard-label tail.

**What it targets**: Local geometric and photometric invariances not supplied by crop, flip, or convex mixup, using a literature-backed augmentation family distinct from failed shared-rectangle CutMix and feature dropout.

**Reasoning**: RandAugment improved CIFAR models with a compact policy space and is already available locally. `N=1, M=5` is deliberately weaker than conventional policies to limit compounding with mixup; an immutable worker transform avoids an ambiguous prefetched time cutoff.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/randaugment.md`; EXP-002, EXP-003, EXP-005, and EXP-006 analyses.

**Estimated Effort**: low.

**Risk Assessment**: It still adds regularization after several over-regularization failures and remains active during clean-label refinement. PIL work may inflate total wall time even if excluded from counted step time; a real-loader preflight must project at most 500 seconds total and a valid negative run must not trigger a stronger policy retry.

### Selective 160-Channel Final Stage
**Summary**: Replace stage widths `[32, 64, 128]` with `[32, 64, 160]`, leaving the expensive high-resolution stages and the entire accepted training recipe unchanged. This adds 39.0% parameters but only 17.0% convolution/linear MACs, unlike rejected uniform width 3's 124% parameters and 120.6% MAC increase.

**What it targets**: Additional abstract class-separating capacity at 8x8 resolution, allocated where each learned parameter has relatively low spatial compute cost. It addresses capacity without repeating the compute-prohibitive uniform-width preflight.

**Reasoning**: The topology has 961,562 parameters and an affine timing prior of about 90.3% accepted throughput / 128.1 passes, materially above width 3's measured 56.8% / 80.6 passes. Wide-residual literature supports width, and the final stage directly feeds global pooling and classification.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/wide-residual-networks.md`; EXP-006 `proposals/idea-01.md`.

**Estimated Effort**: medium.

**Risk Assessment**: The final stage may not be the capacity bottleneck, the abrupt 64-to-160 transition may optimize poorly, and exposure loss can still dominate. A matched two-topology preflight must retain at least 85% throughput and project at least 120 passes before scoring.

## Review

The offline blind critic selected isolated BF16 autocast and identified two significant refinements. First, the intervention must be described as a joint numerical/throughput treatment: BF16 changes convolution numerics from the first update, and more wall-time-aligned updates do not prove exposure alone was limiting. Second, the fresh preflight must include pinned-host-to-device copies inside the same timed region as production, because the proposal-development synthetic CUDA-input benchmark omitted copy/compute interaction. The scored run remains authoritative for realized exposure. I also adopt the caution that faster epochs create more legal every-fifth-epoch observations: the 94.17% rule remains formal, while a transient best spike with worse final accuracy/loss will be treated as weak mechanistic evidence. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. BF16 scored 8.5/10 for evidence and 7.5/10 for impact because repeated local H20 timing and dtype checks directly address the compute/exposure gap without changing batch or regularization. Selective stage-3 width has a slightly higher theoretical impact score but weaker evidence that final-stage capacity is limiting and necessarily loses exposure. RandAugment is downgraded after six negative regularization/schedule follow-ups and because `M=5` does not bound every sampled operation.

## Chosen Idea
**Selected**: Isolated BF16 Autocast at Batch 256

**Why this idea**:
It is the only finalist with measured local H20 upside against the diagnosed fixed-time limiter while preserving the accepted architecture, batch size, optimizer, seed, regularization, and FP32 evaluator. Batch 512 was rejected from the same local benchmark because its 1.5% incremental throughput cannot justify the optimization confounds. A production-copy preflight will verify at least 1.10x matched step throughput without touching test accuracy before the single scored run.

**Hypothesis**:
CUDA BF16 autocast with FP32 master weights at batch 256 will jointly provide a sufficiently accurate numerical path and at least 10% denser optimizer/data exposure on the counted-time schedule to raise `best_test_acc` from 94.07% to at least 94.17%. The run should retain final accuracy near best and final test loss near or below 0.2432; any valid score below 94.17% rejects this deployed BF16 treatment regardless of throughput.

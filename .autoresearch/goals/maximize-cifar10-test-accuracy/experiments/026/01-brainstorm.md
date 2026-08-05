# Brainstorm EXP-026
**Created**: 2026-07-26

## Web Search & Literature Review

- **RandAugment** (`knowledge/papers/randaugment.md`): a standard small operation-count/shared-magnitude policy adds CIFAR invariances without a policy-search dependency; CPU cost and magnitude-independent operations are the local risks.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): augmentation and mixed-target effects can persist after early removal, supporting an augmentation-free late refinement phase.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild target softness can address overconfidence, but stacking soft-target mechanisms needs calibration.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): shallow width can allocate CIFAR compute effectively, though local width/depth near-misses show the operating point matters.

No network access was used; this offline brainstorm relies on the persistent local knowledge base and completed experiment artifacts.

## Experimental History Review

- The accepted WRN-16-2 plus early batch-shared alpha-0.2 mixup remains 94.07% after 23 rejected or preflight-aborted follow-ups. The exact 65% mixup duration is locally bracketed; stronger alpha, CutMix, dropout, per-example coefficients, and late decay/schedule changes regress.
- The strongest near-misses are full two-gate SE at 94.16%, one extra final block at 94.15%, selective width 160 at 94.11%, and late EMA at 94.10%. Full-SE placement/function ablations lose its signal, while the exact diagnostic-free path still misses its exposure gate; that family is closed.
- More exposure alone is not sufficient: BF16 and fixed-MAC depth redistribution delivered 159-172 passes but scored 93.81-93.82. Lower terminal loss likewise failed to guarantee top-1 gains under EMA or SAM.
- The remaining limiter is a generalization/noise-scale boundary, not raw convergence. The accepted model lacks color, contrast, mild rotation/shear, and other image-space invariances beyond crop/flip; the prior reviewer specifically reserved a worker-safe early-only RandAugment design.
- Persistent workers complicate temporal CPU transforms because prefetched batches cross a shared cutoff. A valid early-only design must bound and audit this lag, or change loaders at an explicit epoch boundary without touching evaluator behavior.
- Untried gaps include a temporal image-invariance intervention, batch-size/noise-scale operating points, a cheaper final-stage width between 128 and 160, isolated weaker mixup, and low-cost target regularization. All must preserve the fixed H20 budget and single fixed seed.

## Collected Ideas

- **Worker-safe early one-op RandAugment** - insert torchvision `RandAugment(num_ops=1, magnitude=5)` after crop/flip, controlled by shared worker state and disabled at the first epoch boundary after 65% counted time. This targets missing photometric/geometric invariances while preserving a genuinely clean terminal phase; local literature and reviewer feedback support the temporal version, but prefetch lag and CPU wall time need fail-closed checks.
- **Early on-device Cutout** - apply one vectorized 8x8 mask per image on CUDA only before 65%, then remove it with mixup. This avoids worker state and CPU wall overhead while targeting occlusion invariance, but it stacks information removal with accepted mixup and inherits adverse regularization evidence.
- **Batch-128 noise-scale retune** - halve batch size and peak/floor LR proportionally while keeping the time schedule, architecture, mixup law, and data pipeline. More parameter updates and higher gradient noise may improve boundary refinement, but fewer examples per second could erase the benefit and the paired LR change complicates attribution.
- **Batch-384 utilization retune** - raise batch and LR proportionally to exploit H20 throughput and see more examples within 300 seconds. It attacks fixed-time data exposure, but history shows exposure alone is weak and fewer optimizer updates plus lower gradient noise may hurt generalization.
- **Selective stage-3 width 144** - widen only the final low-resolution stage from 128 to 144, an intermediate point between accepted and EXP-010's 160 channels. It seeks a better capacity/exposure balance than the 94.11 width-160 near miss, with shape-specific H20 timing required because MACs do not predict throughput.
- **Early label smoothing instead of mixup** - replace early mixup with epsilon-0.05 label smoothing through 65%, retaining the hard-label tail. It targets overconfidence with almost no input cost, but abandons the only accepted regularizer and has a lower evidence prior than sample-aware mixup.
- **Weaker alpha-0.1 mixup** - change only the accepted Beta concentration, keeping duration and all other semantics. This completes the isolated strength bracket but endpoint-heavy coefficients probably under-regularize after alpha 0.4 already showed the opposite side is worse.
- **Moonshot frequency-domain perturbation** - apply a tiny random low-frequency amplitude perturbation early while preserving phase and labels. It could add texture/style invariance absent from RandAugment, but custom FFT augmentation is expensive, weakly grounded locally, and too easy to miscalibrate in one run.

## Combinations

- **Early RandAugment + accepted temporal mixup**: image-space nuisance invariance and between-example linearity are complementary; synchronizing both removals preserves the validated hard-label tail better than always-on RandAugment, while one mild operation limits additive regularization.
- **Width 144 + accepted mixup**: modest extra low-resolution capacity can exploit the already validated regularizer, and the smaller width may retain enough exposure to outperform width 160's +0.04 near miss. This is stronger than width alone only if the accepted mixup remains correctly calibrated for the added parameters.
- **Batch-128 retune + accepted mixup**: a higher-noise optimizer trajectory may complement mixup's smooth decision constraint and yield more updates, but the combination changes both sample exposure and LR/noise scale, so matched timing and preregistered scaling are essential.

## Candidate Ideas

### Selective 144-Channel Stage 3
**Summary**: Replace only the final stage width, `[32,64,128] -> [32,64,144]`, keeping the accepted FP32 batch-256 mixup recipe. Construct the accepted graph first, then replace the full tail under a restored CPU RNG fork with fixed local seed 26026 so upstream weights and post-construction CPU/CUDA RNG remain exact. Score only if matched H20 timing projects >=137 passes.

**What it targets**: A possible capacity/exposure balance between the accepted tail and EXP-010's directionally positive but insufficient width-160 treatment.

**Reasoning**: Width 144 adds 18.5% parameters and 8.1% MACs, roughly half the width-160 increment, and could recover exposure while retaining some low-resolution class-separation capacity. Shape-specific H20 timing, not MAC estimates, is authoritative.

**Sources**: `proposals/idea-03.md`; EXP-010 plan/analysis; `knowledge/papers/wide-residual-networks.md`; project H20 shape insight.

**Estimated Effort**: medium

**Risk Assessment**: This is likely an unjustified adjacent retry: EXP-010 explicitly closed neighboring width tuning and offers no evidence for an interior optimum at 144. Capacity and exposure move together, and a new tail seed prevents a clean response curve. It should rank below genuinely new mechanisms even if technically feasible.

### Batch-128 With a Proportionally Scaled LR Curve
**Summary**: Change batch 256 to 128, scale the full LR curve from `0.2->0.002` to `0.1->0.001`, and double only the nonbinding step cap to preserve its example-domain allowance. Keep WRN-16-2, transforms, batch-shared alpha-0.2 mixup through 65%, momentum/Nesterov/decay, time schedule, seed, and evaluator unchanged.

**What it targets**: The remaining noise-scale/generalization boundary through roughly twice as many, noisier optimizer decisions and mixup draws per unit example exposure, rather than simply increasing data passes.

**Reasoning**: The LR-per-sample scaling is preregistered and interpretable, and both batch sizes drop the same 80 examples per epoch. More exposure alone has failed, whereas this changes update granularity and BatchNorm/mixup-group statistics. A matched H20 gate requires at least 120 projected passes and 46,875 updates.

**Sources**: `proposals/idea-02.md`; EXP-001/002 accepted recipe; EXP-009/016 exposure failures; `03-experiment-learnings.md`.

**Estimated Effort**: low

**Risk Assessment**: Four constants change as one coupled operating point, and halving LR may under-update within 300 seconds despite more steps. BatchNorm and batch-shared mixup semantics co-vary, so a miss cannot isolate gradient noise. No alternative LR, batch, momentum, or floor is permitted after results.

### Worker-Safe Early-Only One-Operation RandAugment
**Summary**: Add torchvision `RandAugment(num_ops=1, magnitude=5, num_magnitude_bins=31, bilinear, mean-color fill)` after accepted crop/flip and before tensor conversion. A top-level transform wrapper reads a shared multiprocessing byte. RandAugment stays active through the complete epoch in which counted time reaches 65%, then the parent flips the flag only after exhausting that iterator, guaranteeing that every subsequent prefetched batch is RandAugment-free while preserving one persistent loader.

**What it targets**: Missing photometric and mild geometric invariances in the accepted crop/flip/mixup pipeline, while preserving the useful late hard-label refinement distribution that made always-on RandAugment unattractive.

**Reasoning**: RandAugment and time-local regularization provide direct local literature support. Prior reviews repeatedly reserved an early-only worker-safe treatment, and the attention/capacity/optimizer search has plateaued. Epoch-exhaustion creates a bounded sub-epoch lag but avoids prefetch leakage, loader reconstruction, sampler drift, or a custom augmentation engine. A real-data multi-epoch loader benchmark must project <=500 seconds total wall time before the sole score.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/003/005/006; prior RandAugment reviews in EXP-009/010/017.

**Estimated Effort**: medium

**Risk Assessment**: Even one operation can compound accepted mixup, and magnitude-independent Equalize/AutoContrast are not uniformly mild. The epoch-boundary cutoff trails the mixup transition by up to one epoch. Shared-state visibility, deterministic sampler isolation, CPU starvation, and wall time need fail-closed preflights; a valid miss closes this exact policy without tuning.

## Review

The idea reviewer selected worker-safe early-only RandAugment at 4.5/5 for both evidence and impact. I adopt its fixed `N=1,M=5`, no operation filtering, epoch-exhausted cutoff, and conservative wall gate. The later plan review identified worker-stream drift as avoidable attribution noise, so RandAugment receives an isolated per-worker RNG stream while accepted crop/flip RNG is restored exactly; no seed is added or searched. Full reviews: `01-idea-review.md` and `02-plan-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. RandAugment uniquely targets the diagnosed missing invariance axis while its CPU cost remains outside counted optimizer time. Batch 128 is retained only as a later granularity fallback after correcting its noise-scale claim; width 144 is rejected as an adjacent retry of a family EXP-010 explicitly closed.

## Chosen Idea
**Selected**: Worker-Safe Early-Only One-Operation RandAugment

**Why this idea**:
It is the strongest orthogonal generalization lever remaining after attention, capacity, exposure, and optimizer refinements plateaued. The standard torchvision policy adds photometric/geometric invariances missing from crop/flip/mixup, while a shared worker flag flipped only after an exhausted epoch prevents prefetch leakage into all later epochs. The epoch-boundary lag is preregistered, and a <=500-second projected-wall gate protects the hard timeout without consuming a scored run.

**Hypothesis**:
Adding exactly one torchvision RandAugment operation at magnitude 5 during the early phase, then disabling it after the first fully exhausted epoch whose counted time reaches 65%, will retain normal accepted optimizer exposure, complete below 600 seconds wall time, and raise the fixed-seed `best_test_acc` from 94.07% to at least 94.17% by learning missing image-space invariances without corrupting the later clean refinement epochs.

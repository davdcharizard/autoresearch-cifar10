# Brainstorm EXP-002
**Created**: 2026-07-24

## Web Search & Literature Review

- **When, Where and Why to Average Weights?** (`experiments/002/papers/weight-averaging.md`): carefully windowed parameter averaging can mildly improve generalization with little memory or compute cost.
- **When Does Label Smoothing Help?** (`experiments/002/papers/label-smoothing.md`): mild soft targets can reduce overconfidence and improve classification generalization.
- **mixup** (`knowledge/papers/mixup.md`): convex image/label interpolation improves CIFAR generalization with one forward pass.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization is most valuable early and can be removed for late hard-label convergence.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): width remains a viable capacity lever, though EXP-001 already validated a moderate WRN.

## Experimental History Review

- EXP-001 improved `best_test_acc` from 91.54% to the current 93.38% baseline using WRN-16-2, batch 256, selective decay, and a time-aligned cosine schedule.
- The successful run completed 147 epochs/28,540 steps, used only 1.1 GiB VRAM, and ended at 93.34%, only 0.04 points below its best. This indicates stable convergence rather than a lucky isolated checkpoint.
- Persistent DataLoader workers and periodic-plus-final evaluation are validated protocol requirements; non-persistent workers caused a 600-second timeout.
- No generalization regularizer beyond crop/flip and selective weight decay has yet been tried on the successful WRN baseline. EMA, mixup, label smoothing, Cutout, and additional width remain open.

## Objective Diagnosis

The dominant gap is now generalization, not basic optimization or memory capacity. EXP-001 drove smoothed training loss from roughly 0.5 after warmup to about 0.005 late in training while test accuracy plateaued near 93.3%. Its final checkpoint retained the best score, so schedule completion is working; simply extending convergence is unlikely to yield a large gain. The H20 still has over 96 GiB free, but a larger model may reduce data exposure and does not directly address the near-zero-loss generalization gap. The highest-value next interventions should smooth the learned solution or regularize targets/examples early, then preserve EXP-001's clean low-LR tail. Because the moving threshold is 93.48%, small but genuine improvements are sufficient, making focused low-overhead changes preferable to another broad bundle.

## Collected Ideas

- **Late exponential moving average** — maintain a shadow average only after the model enters the lower-LR half of training, then evaluate averaged parameters with current BN buffers. This targets residual SGD iterate variance while avoiding bias from early under-trained weights.
- **Final-window stochastic weight average** — uniformly average a small set of late checkpoints rather than using exponential decay. This can broaden the solution but risks averaging points along a nonstationary cosine trajectory and complicates BN statistics.
- **Early mixup with a clean tail** — use alpha-0.2 mixup during the high-LR majority of training and switch to hard labels for the last 25-30% of counted time. This directly targets the near-zero training-loss gap while retaining late class-confidence refinement.
- **Early label smoothing with hard-label finish** — use mild smoothing such as 0.05-0.1 before the final cosine tail, then return to ordinary cross entropy. This is simpler than mixup and adds no image transform overhead, but its upside may be smaller.
- **On-device Cutout** — mask a modest 8x8 patch per image during the early critical period and disable it late. GPU-side masking avoids the CPU/wall-time failure risk identified for RandAugment.
- **Width-factor 3 WRN** — increase stage widths to 48/96/192 while keeping depth and schedule structure. The H20 can easily hold it, but fewer passes and stronger overfitting may offset extra representation capacity.
- **Schedule/decay refinement** — keep the architecture fixed and adjust peak LR, warmup fraction, minimum LR, or weight decay. This is highly controlled but risks spending a full loop on a small local hyperparameter move without a strong directional signal.
- **Shake-style multi-branch moonshot** — add a second residual branch with stochastic affine mixing during training. Literature suggests strong CIFAR regularization, but doubled branch compute and implementation complexity make it high risk under 300 seconds.

## Combinations

- **Early mixup + late EMA**: mixup regularizes representation learning while EMA stabilizes the late hard-label trajectory. The mechanisms are complementary, but the combination weakens attribution and may over-smooth a small-margin experiment.
- **Mild label smoothing + late hard-label EMA**: low-cost target regularization addresses overconfidence, then EMA reduces late iterate noise. This is lighter than mixup plus EMA but still combines two unvalidated changes.
- **Width factor 3 + early Cutout**: extra capacity raises the ceiling while Cutout counters overfit. The combination could beat either alone, but reduced data passes make it less attractive than regularizing the proven WRN first.

## Candidate Ideas

### Throughput-Balanced WRN-16-3
**Summary**: Increase the successful model to width factor 3, batch size to 384, and peak LR to 0.3 while preserving depth, schedule shape, decay, augmentation, persistent workers, and sparse evaluation. This tests the smallest additional integer width step with a paired utilization adjustment.

**What it targets**: Remaining representation capacity and the H20's 96+ GiB memory headroom.

**Reasoning**: WRN width is already validated by EXP-001, and batch 384 can partially amortize the wider convolutions. The time-based schedule remains comparable even when realized steps change.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/wide-residual-networks.md`; EXP-001 report and project insights.

**Estimated Effort**: low

**Risk Assessment**: The current model already reaches near-zero training loss, so more width may worsen rather than close the generalization gap. LR 0.3 and reduced update count add optimization risk, making this less directly matched to the diagnosed limiter.

### Late-Phase Exponential Weight Averaging
**Summary**: Preserve the complete EXP-001 training path and start an FP32 parameter EMA at 70% counted time. Update every ten optimizer steps with decay 0.995, evaluate only the EMA parameter view after initialization, retain live BN buffers, and restore live parameters after each sparse evaluation.

**What it targets**: Residual late-iterate variance around EXP-001's stable 93.3% plateau, without changing gradients, examples, architecture, optimizer, or LR.

**Reasoning**: EXP-001's final and best scores differed by only 0.04 points, indicating a converged basin where low-cost averaging may yield a small but sufficient improvement. A late start avoids averaging under-trained high-LR weights.

**Sources**: `proposals/idea-01.md`; `papers/weight-averaging.md`; EXP-001 report and results index.

**Estimated Effort**: medium

**Risk Assessment**: EMA can lag a still-improving cosine trajectory, and averaged affine parameters may not align perfectly with live BN statistics. Safe parameter swapping/restoration adds implementation risk, though compute and memory overhead are tiny.

### Early-Only Mild Mixup With a Hard-Label Cosine Tail
**Summary**: Keep WRN-16-2 and all EXP-001 optimizer/schedule settings, but apply batchwise mixup with alpha 0.2 during the first 65% of counted training. Use ordinary hard-label cross entropy for the final 35%, leaving about 105 seconds for clean low-LR margin refinement. Do not stack Cutout or label smoothing.

**What it targets**: The near-zero training loss and remaining generalization gap of the proven WRN baseline.

**Reasoning**: Mixup has direct CIFAR evidence, adds only one interpolation and one forward pass, and critical-period evidence supports removing it before late convergence. The long clean tail mitigates the main short-budget risk of soft-target under-convergence.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; `papers/label-smoothing.md`; EXP-001 report.

**Estimated Effort**: low

**Risk Assessment**: Even mild mixup may slow hard-label fitting or reduce throughput enough to miss 93.48%. The 65% cutoff is evidence-informed but uncalibrated on this exact WRN.

## Review

The reviewer found that mixup is the only finalist whose mechanism directly targets EXP-001's near-zero-loss generalization gap and whose expected effect has comfortable margin over the 0.1-point threshold. Significant feedback adopted: keep mixup isolated from Cutout/label smoothing, preserve the full EXP-001 architecture and schedule, log realized steps/epochs to rule out throughput loss, and pre-register a 50% cutoff follow-up if a near-flat result occurs with normal throughput. Full scores and concerns are in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`: **Early-Only Mild Mixup With a Hard-Label Cosine Tail** scored 8/10 for both evidence/reasoning and impact. EMA's observed headroom is smaller than the acceptance margin, while WRN-16-3 contradicts the generalization-limited diagnosis.

## Chosen Idea
**Selected**: Early-Only Mild Mixup With a Hard-Label Cosine Tail

**Why this idea**:
EXP-001 is well optimized but overconfident, making sample/target interpolation a direct intervention on the remaining gap. Alpha-0.2 mixup adds negligible overhead and the final 35% hard-label cosine tail preserves roughly 50 epochs for margin recovery, while all proven WRN and runtime settings remain fixed.

**Hypothesis**:
Applying batchwise mixup with alpha 0.2 during the first 65% of counted training, then returning to ordinary hard-label cross entropy for the final 35%, will raise `best_test_acc` from 93.38% to at least 93.48% without materially reducing steps, epochs, or wall-time compliance.

# Brainstorm EXP-004
**Created**: 2026-08-05

## Web Search & Literature Review

- **RandAugment** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/randaugment.md`): a small augmentation search space improves CIFAR generalization without dependencies, but magnitude and host-transform throughput must match the operating regime.
- **SGDR** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`): horizon-aligned cosine refinement is now both literature-backed and locally validated in EXP-002.
- **When Does Label Smoothing Help?** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/label-smoothing.md`): smoothing often helps NLL/calibration; EXP-003 locally confirmed lower test loss without higher top-1.
- **When, Where and Why to Average Weights?** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`): late averaging may complement annealing, but BatchNorm state remains a correctness cost.
- **Identity Mappings in Deep Residual Networks** (<https://arxiv.org/abs/1603.05027>): full preactivation improves clean identity propagation and CIFAR results, with an explicit caveat that benefits are weaker for shallow networks.
- **Accurate, Large Minibatch SGD** (<https://arxiv.org/abs/1706.02677>): linear LR scaling can preserve optimization when batch size increases, but evidence comes from much larger ImageNet regimes.

## Experimental History Review

- Current best remains `91.83%` at commit `5016cc4`; EXP-004 must reach at least `91.93%`.
- EXP-001 showed a 15% high-LR hold plus cosine/Nesterov was too short (`91.57%`). EXP-002 validated standard momentum, an 80% `lr=0.1` plateau, and a low-LR cosine tail (`91.83%`). Preserve that optimizer horizon unless schedule timing itself is the isolated intervention.
- EXP-003's built-in `label_smoothing=0.05` matched best accuracy at `91.83%` while lowering final test loss `0.2843 -> 0.2740`; it also reduced fixed-budget steps `38,629 -> 36,039` (`-6.7%`). This exact operating point is rejected, and claimed cheap interventions now require measured throughput.
- Remaining plausible limiters are weak image-level invariance, untapped H20 examples/second, and representation quality. The fixed evaluator is not a limiter: dense tail evaluation has repeatedly captured terminal or near-terminal peaks.

## Collected Ideas

- **Conservative RandAugment** — Add a single randomly selected torchvision augmentation at modest magnitude before tensor conversion while keeping crop/flip and EXP-002 optimization fixed. It targets image invariance rather than target confidence and runs in persistent worker processes, though prefetch starvation must be measured.
- **Cutout-style random erasing** — Erase one modest square region after tensor conversion to encourage distributed features and occlusion robustness. It is simpler than RandAugment and has low policy complexity, but a poorly chosen region size can discard too much of a 32x32 image.
- **Batch 256 throughput recipe** — Double batch size to process more examples per synchronized step and use the H20's large memory headroom, with an explicitly justified LR choice. It attacks examples/second directly but reduces gradient noise and changes the number of optimizer updates, so accuracy may decline despite higher sample exposure.
- **Same-width preactivation ResNet-20** — Replace post-activation blocks with full preactivation at the existing `16/32/64` widths. It targets residual optimization and representation without the large throughput loss of widening, but published advantages are strongest for much deeper networks.
- **Single-log-softmax label smoothing** — Reimplement `epsilon=0.05` mathematically from one log-softmax and gather to recover EXP-003's 6.7% step loss. It directly tests whether throughput masked a statistical gain, but exact semantics must be verified and top-1 was flat even with lower NLL.
- **Narrow plateau-boundary adjustment** — Move the accepted 80% hold to 75% or 85% while retaining the `0.1 -> 0.01` step and cosine tail. It refines a proven lever with zero overhead, but a single setting has modest upside and risks overfitting to the prior trajectory.
- **Late raw-parameter EMA** — Average parameters only through the low-LR tail and handle BatchNorm buffers explicitly. This can stabilize the endpoint, but repeated dense evaluations already show a small best/final gap and BN refresh adds wall time.
- **Tail-only sharpness-aware moonshot** — Pay a second forward/backward pass only during the final refinement phase to seek a flatter solution. It attacks generalization directly but could halve updates precisely when refinement matters most.

## Combinations

- **RandAugment + validated schedule**: worker-side image diversity targets a different failure mode than the accepted optimizer and avoids soft-target NLL/top-1 divergence. The schedule supplies late refinement after stronger invariance learning.
- **Batch 256 + schedule retiming**: increased sample throughput could be paired with an update-aware boundary, but this bundles two changes and is less interpretable than a first isolated large-batch test.
- **Preactivation + conservative augmentation**: improved signal propagation plus stronger invariance has higher upside than either alone, but should be decomposed because both can independently change convergence.
- **Efficient smoothing + hard-label tail**: recover step throughput and align final optimization to top-1, plausibly addressing both EXP-003 failure mechanisms. It is promising but materially changes two aspects of the rejected operating point.

## Candidate Ideas

### Noise-Scale-Matched Batch 256
**Summary**: Set batch size to 256 and scale the complete LR schedule 2x (`0.2 -> 0.02 -> 2e-4`) while preserving its 80/20 time fractions and all other controls. This treats batch and LR scaling as one coupled intervention that holds approximate `lr/batch` noise scale fixed. Full proposal: `proposals/idea-02.md`.

**What it targets**: Untapped H20 examples/second. EXP-002 presents 4.94M images in 300 seconds while using only 330 MB; batch 256 could amortize fixed overhead and present 5.9-8.5M augmented views.

**Reasoning**: Linear scaling preserves first-order update magnitude/noise at equal examples and is more coherent than a batch-only change. The model is small enough that batch 256 may improve accelerator utilization substantially.

**Sources**: Goyal et al. large-minibatch SGD; SGD noise-scale literature cited in `proposals/idea-02.md`; EXP-002/003 throughput measurements.

**Estimated Effort**: low.

**Risk Assessment**: `lr=0.2` may be unstable without warmup, fewer parameter updates may outweigh extra examples, reduced BatchNorm/gradient noise may hurt generalization, and more tail epochs increase evaluator wall time.

### One-Operation Magnitude-7 RandAugment
**Summary**: Insert `transforms.RandAugment(num_ops=1, magnitude=7)` after crop/flip and before `ToTensor`, keeping hard labels and all EXP-002 choices fixed. Gate the full run on a fresh-process loader benchmark to ensure eight persistent workers can supply at least 80 batches/s with stable timed epochs. Full proposal: `proposals/idea-01.md`.

**What it targets**: Weak input-level invariance after target smoothing improved NLL but not top-1. Worker-side transforms may strengthen hard-label generalization without adding synchronized GPU loss work.

**Reasoning**: RandAugment has direct CIFAR evidence and torchvision support with no dependency. One operation below the default magnitude bounds 32x32 distortion, while the mandatory loader gate responds to EXP-003's 6.7% fixed-budget step loss.

**Sources**: `knowledge/papers/randaugment.md`; EXP-003 analysis; `proposals/idea-01.md`.

**Estimated Effort**: medium.

**Risk Assessment**: PIL transforms may starve the H20 or push total runtime toward 600 seconds; default operations/interpolation may be destructive at CIFAR resolution; stronger augmentation may underfit in the fixed horizon.

### Same-Width Full-Preactivation ResNet-20
**Summary**: Convert blocks to BN-ReLU-Conv full preactivation, remove post-add ReLU, move stem normalization/activation into the blocks, and add final BN-ReLU before pooling. Retain Option-A shortcuts, widths, convolution tensors, and exactly 269,722 parameters. Full proposal: `proposals/idea-03.md`.

**What it targets**: Representation and residual signal flow, an untested lever after loss regularization failed top-1. It avoids widening's predicted update loss while preserving architecture capacity and accepted training policy.

**Reasoning**: Full preactivation creates clean additive identity paths and has primary CIFAR evidence. Same convolution shapes and parameter parity make it an unusually clean architecture test with expected throughput within about 3% of EXP-002.

**Sources**: Identity Mappings primary paper; EXP-002/003 analyses; `proposals/idea-03.md`.

**Estimated Effort**: medium.

**Risk Assessment**: ResNet-20 may be too shallow for the published mechanism to matter, Option-A transitions may limit benefits, and changed kernel ordering or activation distributions can reduce throughput or require a different LR.

## Review

Mandatory external Claude review completed successfully with no fallback (`01-idea-review.md`). It selected RandAugment and identified short-horizon underfit as the dominant risk that the loader throughput gate cannot address. Adopted its refinement: apply one-operation magnitude-7 RandAugment only during the high-LR plateau, then rebuild the persistent training loader at the first epoch boundary after 80% with the original crop/flip transform for clean hard-label refinement. The preflight is explicitly only a feasibility gate. Batch 256 was rejected because its expected 14-35% update reduction repeats EXP-003's harmful direction at larger magnitude; preactivation remains a clean but lower-impact fallback because its source discounts the mechanism at ResNet-20 depth.

## Idea Evaluation

Adopt the external verdict and refinement. Claude scored RandAugment `7/10` for both evidence/reasoning and impact, preactivation `6/10` evidence and `5/10` impact, and batch 256 `3/10` on both. Plateau-only augmentation retains the winning input-invariance mechanism while the original final 20% objective and BatchNorm statistics can settle on crop/flip images. Planning must handle worker shutdown/reconstruction explicitly and verify that the added loader restart leaves total runtime below 600 seconds.

## Chosen Idea
**Selected**: Plateau-Only One-Operation Magnitude-7 RandAugment

**Why this idea**:
It targets the strongest untried limiter, input-level invariance, with direct CIFAR evidence and avoids synchronized loss-path overhead. Claude judged it the highest-upside finalist, while its plateau-only refinement addresses the main finite-horizon and BatchNorm mismatch risks without changing the accepted optimizer schedule.

**Hypothesis**:
Applying `RandAugment(num_ops=1, magnitude=7)` during the first 80% of counted training, then returning to the original crop/flip pipeline for the low-LR tail, will preserve near-baseline optimizer exposure and raise `best_test_acc` from `91.83%` to at least `91.93%` on one fixed-seed H20 run.

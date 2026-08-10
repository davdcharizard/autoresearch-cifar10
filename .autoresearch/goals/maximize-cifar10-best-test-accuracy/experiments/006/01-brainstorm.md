# Brainstorm EXP-006
**Created**: 2026-08-05

## Web Search & Literature Review

- **RandAugment** (`knowledge/papers/randaugment.md`): EXP-004 validates N1/M7 locally; magnitude remains the cleanest exposed policy parameter.
- **Identity Mappings in Deep Residual Networks** (<https://arxiv.org/abs/1603.05027>): full preactivation improves residual information flow, though its benefit is weaker at shallow depth.
- **SGDR** (`knowledge/papers/sgdr.md`): preserve the accepted time-aligned optimizer horizon.

## Experimental History Review

- Current best is 92.30% at `11f8469`; threshold is 92.40%.
- EXP-004's N1/M7 augmentation through 80% plus weak tail gained 0.47 points while preserving 99.3% of steps. EXP-005 moved the switch to 75%, preserved throughput, and regressed to 92.12%; keep the 80% alignment.
- Target smoothing and early annealing failed. Remaining credible levers are augmentation strength/type and same-cost representation flow; exact effect sizes remain entangled with augmentation RNG draws under the fixed single-seed protocol.

## Collected Ideas

- **Magnitude 8** — exploit the accepted policy with the smallest nontrivial strength increase.
- **Same-width preactivation** — change residual signal flow without width, depth, or nominal convolution cost.
- **Plateau-only Cutout** — replace photometric/geometric RandAugment with localized occlusion regularization, then retain the weak tail.
- **Magnitude 6** — test whether the accepted policy is slightly too strong; lower upside after EXP-004's clear gain.
- **Two operations with low magnitude** — broaden invariances but risks loader starvation and compounded distortion.
- **Late weight averaging** — stabilize the tail, but BatchNorm handling and the small best/final gap weaken the case.
- **Wider model moonshot** — exploit VRAM headroom, but expected step loss remains a major fixed-time risk.

## Combinations

- **Preactivation + accepted augmentation** preserves the successful input recipe while testing an orthogonal representation lever; any gain is not dependent on stronger distortion.
- **Cutout + weak tail** mirrors the successful strong-exploration/clean-refinement composition with a different input error mode.
- **Magnitude 8 + accepted tail** is the narrowest exploitation step and leaves throughput/lifecycle unchanged.

## Candidate Ideas

### Candidate B: RandAugment Magnitude 8

**Summary**: Change exactly `RandAugment(num_ops=1, magnitude=7)` to magnitude 8 in the accepted EXP-004 recipe. Preserve the model, operation count, transform order, 80% augmentation/LR boundary, weak tail, optimizer, workers, seed, and evaluation cadence byte-for-byte.

**What it targets**: Incremental local exploitation of the only augmentation policy already demonstrated to improve the moving baseline. It tests whether one additional torchvision magnitude bin supplies useful invariance during the high-LR plateau without adding any operation or changing lifecycle overhead.

**Reasoning**: EXP-004 gained 0.47 points with N1/M7 and EXP-005 ruled against shortening its exposure. Magnitude is therefore the cleanest remaining coordinate within the accepted family, and M8 is the smallest nontrivial upward step. The one-literal diff gives exceptionally strong attribution and should preserve throughput. A disposable loader preflight must show the slowest M8 epoch at least 140 batches/s. Expected best accuracy is approximately 92.40-92.55%, but a ten-image threshold and a changed augmentation RNG stream make marginal effects uncertain.

**Sources**: Cubuk et al., [RandAugment: Practical Automated Data Augmentation with a Reduced Search Space](https://arxiv.org/abs/1909.13719); local EXP-004 and EXP-005 results.

**Estimated Effort**: Low. One training-code literal plus loader throughput/lifecycle verification and the standard full experiment.

**Risk**: Medium. M7 may already be locally optimal; stronger translations, solarization, or photometric changes may erase useful small-object detail; the weak tail may not fully repair additional distribution shift; a marginal outcome cannot distinguish true magnitude response from fixed-seed augmentation variance.

### Candidate A: Same-Width Full-Preactivation ResNet-20

**Summary**: Preserve the accepted EXP-004 augmentation, optimizer, loader, timing, and evaluation recipe, but convert the parameter-matched ResNet-20 from post-activation residual blocks to full preactivation. Move BN/ReLU before each residual convolution, remove the post-addition ReLU, add the required final BN/ReLU before pooling, retain raw Option-A shortcuts, and keep the exact 269,722 trainable parameters and all convolution shapes.

**What it targets**: Representation and optimization quality under the successful hard-view plateau. Unimpeded identity paths may help the shallow network optimize strongly augmented examples and carry a cleaner accumulated representation into the weak refinement tail without sacrificing fixed-time update count.

**Reasoning**: EXP-004 established the data/schedule composition and EXP-005 showed its 80% boundary should not be retuned. Preactivation is an orthogonal lever with nominally unchanged convolutional work and exact parameter parity. The mechanism is credible, but published benefits are strongest for much deeper residual networks; ResNet-20 has only nine residual units, so this is a higher-novelty, weaker-local-evidence candidate. Expected best accuracy is 92.42-92.62% if the mechanism transfers, with steps within 3% of EXP-004's 38,358.

**Sources**: He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), including the authors' shallow-depth caveat; local EXP-004 and EXP-005 trajectories.

**Estimated Effort**: Medium. The diff is confined to model classes, but requires exact topology, raw-shortcut, output-shape, parameter-count, and throughput verification.

**Risk**: Medium-high. Nine residual units may not be deep enough for identity-path improvements to move top-1; changed BN placement may over-regularize with RandAugment or alter kernel scheduling; an implementation that uses the preactivated shortcut or retains a post-add ReLU would test the wrong architecture.

### Candidate C: Plateau-Only Fixed-Square Cutout

**Summary**: Replace EXP-004's plateau RandAugment with one mean-valued 16x16 square erased from every normalized training view using `RandomErasing(p=1, scale=(0.25, 0.25), ratio=(1, 1), value=0, inplace=True)`. Retain crop/flip, switch to the accepted weak loader at exactly 80%, and preserve every non-augmentation setting.

**What it targets**: Occlusion robustness and part-distributed evidence, an input-space error mode orthogonal to RandAugment's broad color and geometric transformations. The weak final 20% would still adapt BatchNorm and the classifier to unoccluded evaluation images.

**Reasoning**: Cutout is evidence-backed on CIFAR and cheaper than PIL RandAugment, so it should retain optimizer exposure while testing whether localized occlusion is a better strong-view family for this small model. Replacement rather than stacking keeps attribution readable and avoids compounded distortion. A disposable preflight must sustain at least 160 batches/s and verify exact masks plus clean worker teardown. Expected best accuracy is 92.40-92.70%, but the proposal replaces an already successful regularizer and therefore has material downside.

**Sources**: DeVries and Taylor, [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552); Zhong et al., [Random Erasing Data Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/7000); local EXP-004 and EXP-005 results.

**Estimated Effort**: Low-medium. The transform/state rename is small, but exact mask semantics and worker throughput/lifecycle require a dedicated preflight.

**Risk**: Medium-high. Removing RandAugment also removes its broader invariances; masking 25% of every 32x32 image may be excessive under a 300-second budget; mean-filled hard edges can become artifacts; the papers support the mechanism but not superiority over the accepted N1/M7 composition.

## Review

The mandatory external Claude review completed successfully with exit code 0 and is preserved in `01-idea-review.md`. No fallback reviewer was used.

The reviewer found no fatal scope, feasibility, reward-hacking, seed-hacking, or already-failed-approach violations. It selected plateau-only fixed-square Cutout because its occlusion mechanism is source-backed, orthogonal to the accepted broad RandAugment invariances, and offers the highest plausible ceiling. It rejected preactivation as the lead because the cited evidence predicts a near-null at ResNet-20 depth, and rejected M8 because its direction is unsupported and the likely effect barely clears the ten-image threshold despite excellent attribution.

The selection is conditional on five refinements adopted for planning:

1. Treat the comparison honestly as Cutout versus a proven RandAugment recipe, not as a literature-backed expectation that Cutout must win.
2. Pre-register plateau train-loss EMA and the final strong-phase checkpoint as underfitting diagnostics relative to EXP-004.
3. Preserve 16x16 and `p=1.0` as one canonical, predeclared strength while explicitly accepting that full-strength occlusion during a shortened plateau may starve fitting; do not tune after seeing the run.
4. Verify exact post-normalization mean-fill semantics, both state predicates, the switch log, and one clean forkserver teardown.
5. Treat a marginal 92.40-92.45 pass as formally successful under the protocol but not a reliable effect-size claim under the fixed single-seed constraint.

## Idea Evaluation

| Candidate | Evidence / reasoning | Potential impact | Decision |
|---|---:|---:|---|
| Plateau-only fixed-square Cutout | 6/10 | 8/10 | Select. Highest ceiling and a distinct CIFAR-relevant invariance, with the review's underfitting and interpretation controls. |
| Same-width full-preactivation ResNet-20 | 5/10 | 5/10 | Defer. Exact parameter and compute parity are strong, but shallow-depth evidence makes a metric-moving result unlikely. |
| RandAugment magnitude 8 | 6/10 | 3/10 | Reject as standalone. Clean one-literal attribution cannot compensate for an unargued direction and near-zero expected effect. |

Cutout carries more regression risk than M8 because it discards a proven regularizer. That downside is accepted deliberately: this goal optimizes the moving best accuracy, and the orthogonal mechanism plus plausible ceiling outweigh the safer candidates' predicted-null outcomes. A failure will be interpreted narrowly as fixed 16x16 Cutout not beating N1/M7 under this model, seed, and time budget.

## Chosen Idea
**Selected**: Plateau-only fixed-square Cutout with the accepted weak refinement tail

**Why this idea**:
It preserves EXP-004's validated 80% strong-exploration / 20% weak-refinement structure while replacing the broad RandAugment operation family with a genuinely different regularization mechanism. Cutout has credible CIFAR evidence and ample loader-throughput headroom, and it offers more plausible upside than a one-bin magnitude adjustment or preactivation in a nine-block network. The choice does not assume external Cutout results transfer head-to-head against RandAugment; that comparison is precisely the experiment.

**Hypothesis**:
Applying one post-normalization, mean-valued 16x16 Cutout patch to every crop/flip view during the first 80% of counted training, followed by the unchanged weak low-LR tail, will force the model to distribute evidence across object parts while retaining at least 98.5% of EXP-004's 38,358 optimizer steps and will raise `best_test_acc` from 92.30% to at least 92.40%. Plateau train-loss EMA and the final strong-phase checkpoint will diagnose whether the canonical 25%-every-image mask instead causes underfitting; they will not be used to tune this run after launch.

# Brainstorm EXP-007
**Created**: 2026-08-05

## Web Search & Literature Review

- **Wide Residual Networks** (<https://arxiv.org/abs/1605.07146>): widening relatively shallow residual networks improved CIFAR accuracy and computational efficiency versus extreme depth. This is directly relevant because the current ResNet-20 has only 269,722 parameters and uses 330.1 MB on a 97,871 MiB H20.
- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (<https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html>): replacing erased pixels with a labeled patch from another training image retains information while providing regional-dropout regularization. It supplies a mechanism-specific response to EXP-006's lossy Cutout failure.
- **Train longer, generalize better** (<https://arxiv.org/abs/1705.08741>): large-batch generalization depends strongly on the number of parameter updates, and Ghost BatchNorm can decouple normalization from physical batch size. This cautions against assuming that more images/s from batch 256 automatically improves fixed-time accuracy.
- **Mixup** (`knowledge/papers/mixup.md`): conservative input/target interpolation improves CIFAR generalization, but must be implemented outside or benchmarked inside the synchronized step to avoid EXP-003's fixed-budget loss-overhead failure.
- **RandAugment** (`knowledge/papers/randaugment.md`): the locally accepted N1/M7 plateau policy should remain the default data recipe unless a replacement has a concrete reason to retain more useful information.

## Experimental History Review

- Current best is 92.30% at `11f8469`; success requires at least 92.40%.
- EXP-002 established an 80% `lr=0.1` exploration plateau plus a low-LR cosine tail. EXP-004 composed N1/M7 RandAugment through that boundary with a weak hard-label tail and gained 0.47 points while retaining 38,358 steps.
- Shortening the strong phase to 75% (EXP-005) regressed 0.18 points. Replacing RandAugment with fixed 16x16 Cutout (EXP-006) retained 99.14% of steps but regressed 0.67 points; preserve the 80% boundary and do not discard broad invariances without a stronger information-retention argument.
- Label smoothing reduced loss but not accuracy and cut step exposure 6.7% (EXP-003). Any batch-mixing loss or optimizer change requires a throughput preflight.
- The accepted model is extremely small relative to the H20: 269,722 parameters, 330.1 MB peak allocation, and approximately 8 ms synchronized steps. Capacity scaling, physical batch scaling, batch-level mixing, weight decay, and deeper blocks remain untested.
- A disposable synthetic H20 diagnostic including pinned host-to-device transfer plus SGD forward/backward/step measured 7.515 ms at width 16/batch 128, 9.251 ms at width 24/batch 128, 10.852 ms at width 32/batch 128, and 13.040 ms at width 16/batch 256. Projected 300-second exposure was respectively 39.9k, 32.4k, 27.6k, and 23.0k updates; batch 256 projects 15.3% more examples but 42.4% fewer updates, while width 2 projects 30.8% fewer updates and examples. These are feasibility diagnostics, not accuracy evidence, and full loader/evaluator runs can differ.
- Inline diagnosis: the best data/schedule recipe now appears better established than the representation consuming it. With best accuracy still far below published wider CIFAR residual networks and enormous compute/memory headroom, model capacity is the leading untested limiter. The main operational risk is losing too many of the roughly 38k parameter updates under the fixed 300-second budget.

## Collected Ideas

- **Same-depth width multiplier 2** — double stage widths from 16/32/64 to 32/64/128 while preserving ResNet-20 depth, Option-A shortcuts, and the complete EXP-004 input/optimizer recipe. This attacks representational capacity with about four times the convolutional parameters; the Wide ResNet evidence favors width over depth, and H20 memory headroom is overwhelming. A GPU microbenchmark must predeclare a minimum update-exposure band.
- **Same-width ResNet-32** — increase blocks per stage from three to five while retaining widths and accepted augmentation. This adds representation depth with less parameter growth than width 2, but the Wide ResNet paper and shallow preactivation review suggest additional thin depth may be less compute-efficient and harder to optimize in a fixed horizon.
- **Plateau CutMix with weak hard tail** — replace lossy mean Cutout with batch-level labeled patches, then return to ordinary hard targets at 80%. CutMix retains every pixel as training signal and directly addresses EXP-006's hypothesized information-loss failure; the hard part is implementing mixing in worker collation or otherwise accounting for GPU-side loss/mask overhead without altering evaluation.
- **Conservative plateau Mixup** — apply alpha 0.2 input interpolation and lambda-weighted hard-target cross entropy during the high-LR phase, then keep the accepted hard-label weak tail. Literature supports CIFAR generalization, but EXP-003 shows that even built-in target smoothing can reduce counted steps, so a worker-side or carefully benchmarked implementation is essential.
- **Batch 256 with learning-rate scaling** — double the physical batch and test a predeclared LR rule while keeping the accepted model and data phases. H20 kernel utilization may improve and expose more images in 300 seconds, but fewer updates and a possible large-batch generalization gap make this a throughput/optimization trade rather than a free win.
- **Weight decay 5e-4** — change only SGD weight decay from 1e-4 to the stronger value commonly used in CIFAR residual training while preserving all accepted mechanics. This is exceptionally controlled and compute-neutral, but the current RandAugment plateau already regularizes strongly and the likely effect may be below the 0.10-point threshold.
- **Plateau-only stochastic depth** — randomly bypass residual branches during the high-LR phase and disable dropping for the weak tail/evaluation. It could regularize an expanded model and reduce average compute, but nine blocks are likely too shallow for enough path diversity, and training/evaluation semantics add implementation risk.
- **Width 2 plus batch 256 moonshot** — jointly use the H20's capacity and batch throughput headroom, with LR and update exposure chosen from a microbenchmark. The combination could train a substantially stronger model on more examples per second, but capacity and optimizer-noise changes would be bundled and difficult to attribute.

## Combinations

- **Width 2 + accepted RandAugment tail composition**: wider representations may absorb the strong N1/M7 plateau without the underfitting seen in the 0.27M model, while the proven weak tail performs clean-objective refinement. This is stronger than width alone on weak inputs or another regularizer on the capacity-limited model because it composes the most credible new representation lever with the validated data recipe.
- **CutMix + accepted weak tail**: CutMix retains informative pixels and proportional labels during exploration, while the hard weak tail removes soft-target and patch-distribution mismatch. This directly improves on Cutout alone by replacing missing regions with class-bearing content and improves on always-on CutMix by preserving final clean calibration.
- **Width 2 + stronger weight decay**: added capacity increases overfitting headroom, and a compute-neutral regularization increase may control it. The combination may outperform either alone, but it bundles two untested operating points and should follow an isolated width result rather than lead.
- **Batch 256 + Ghost BatchNorm**: physical throughput can rise while BatchNorm retains batch-128 statistics. This may beat plain batch scaling by avoiding one source of generalization shift, but update-count loss and extra reshape/normalization complexity remain.

## Candidate Ideas

### RandAugment Plus Worker-Side CutMix with a Weak Hard Tail

**Summary**: Keep crop/flip plus accepted N1/M7 RandAugment per image, then apply installed `torchvision.transforms.v2.CutMix(alpha=1.0, num_classes=10)` to each strong batch inside a picklable worker `collate_fn`. At the existing 80% boundary, explicitly terminate those workers and rebuild the unchanged weak default-collate loader so both CutMix and soft targets end with RandAugment. Full proposal: `proposals/idea-02.md`.

**What it targets**: Partial-view robustness without EXP-006's information deletion. Donor rectangles contain real class-bearing pixels and area-adjusted targets, while retaining N1/M7 avoids discarding the only locally validated broad-invariance method.

**Reasoning**: The CutMix paper's central mechanism directly addresses Cutout's uninformative hole and full original target. The installed API provides exact clipped-area target adjustment. Worker collation keeps image clone/patch work outside synchronized timing, and a weak hard-label tail removes the composite/soft-target distribution late. Before execution, the strong loader must sustain at least 140 batches/s and interleaved H20 timing must project at least 96.5% of EXP-004 steps because dense-target cross-entropy remains timed.

**Sources**: `knowledge/papers/cutmix.md`; `knowledge/papers/mixup.md`; EXP-004 and EXP-006 reports; installed torchvision 0.24.1 API; `proposals/idea-02.md`.

**Estimated Effort**: Medium-high. It requires module-level forkserver-safe collation, RNG-state preservation, dual target semantics, worker/loss preflights, and transition leakage checks.

**Risk Assessment**: High. Always-on alpha-1 CutMix stacked on RandAugment may over-regularize the small model, dense-target loss may repeat EXP-003's step-cost failure, area targets are imperfect for 32x32 objects, and a 60-second weak tail may be insufficient to recover hard-label confidence. RNG isolation must be proven rather than assumed.

### Isolated Weight Decay 5e-4

**Summary**: Change only `WEIGHT_DECAY = 1e-4` to `5e-4` on the complete accepted EXP-004 recipe. Model, N1/M7 phase, weak tail, loss, LR schedule, momentum, loader lifecycle, seed, and evaluator remain byte-for-byte unchanged. Full proposal: `proposals/idea-04.md`.

**What it targets**: Parameter-norm control and clean-test margins under a strong augmentation plateau. RandAugment constrains input invariance but not parameter scale; stronger coupled L2 pressure may reduce brittle filters and augmentation-artifact co-adaptation without changing compute exposure.

**Reasoning**: `5e-4` is a canonical CIFAR Wide ResNet setting with the same batch-128, LR-0.1, momentum-0.9 scale, and the literal change is maximally attributable. The existing SGD kernel already computes coupled decay, so expected exposure remains within 1% of EXP-004's 38,358 steps. The evidence is directional rather than transferable: the accepted model is much smaller and trains for about half as many epochs, so the fivefold decay may instead intensify underfitting.

**Sources**: original CIFAR ResNet and Wide Residual Networks recipes; EXP-003 compute-cost warning; EXP-004 accepted recipe; `knowledge/papers/wide-residual-networks.md`; `proposals/idea-04.md`.

**Estimated Effort**: Low. One literal, an optimizer configuration assertion, ordinary static checks, and the standard single full run.

**Risk Assessment**: Medium. Coupled decay applies to convolution/linear weights, BatchNorm affine terms, and bias through momentum; fivefold pressure during the long high-LR phase can substantially underfit a 0.27M model already challenged by N1/M7. Likely upside is only a few tenths and may lie within the ten-image threshold noise.

### Width-2 ResNet-20 on the Accepted RandAugment Recipe

**Summary**: Double all stage widths from 16/32/64 to 32/64/128 while preserving ResNet-20 depth, post-activation blocks, parameter-free Option-A shortcuts, and every accepted EXP-004 training setting. The model grows from 269,722 to exactly 1,073,962 parameters; only width-derived model channel arguments change. Full proposal: `proposals/idea-01.md`.

**What it targets**: The leading untested quality limiter is representation capacity. The current final feature width is 64 and uses 330.1 MB on a 97,871 MiB H20, while augmentation and schedule choices have already produced the strongest gains. Width 2 tests whether a richer feature basis can absorb the hard N1/M7 plateau and generalize better during the weak tail.

**Reasoning**: Wide Residual Networks provides directional CIFAR evidence that width can be more effective than additional thin depth. Local timing makes the cost explicit: width 32 measured 10.852 ms versus 7.515 ms for width 16, with 599.2 MB peak allocation and a calibrated projection near 26,563 steps. Nearly fourfold capacity therefore costs about 31% of accepted updates/examples, leaving roughly 68 epochs and 13-14 weak-tail epochs. The candidate deliberately tests whether capacity gain outweighs that measured exposure loss; width 1.5 is deferred as a lower-risk follow-up.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-004 accepted recipe and trajectory; EXP-007 synthetic H20 diagnostic; `proposals/idea-01.md`.

**Estimated Effort**: Medium. Model edits are small, but exact parameter/shortcut checks, a repeated GPU timing gate, and careful exposure analysis are required.

**Risk Assessment**: High. A fourfold-parameter model may be under-optimized in only 26-28k steps, fewer evaluation epochs can reduce best-metric sampling, unchanged `1e-4` decay may under-regularize it, and wider initialization necessarily changes the fixed augmentation stream. Published WRN gains use different blocks, regularization, and longer schedules.

## Review

The mandatory external Claude idea review completed successfully with exit code 0 and is preserved in `01-idea-review.md`; no fallback reviewer was used. It verified the width-2 parameter count, accepted the batch-256 exclusion, and selected width-2 ResNet-20 as the highest-ceiling measured-feasibility bet. Weight decay was judged the cleanest but lowest-ceiling candidate, while CutMix was judged poor risk/reward because always-on alpha-1 mixing stacked on N1/M7 could repeat EXP-006's every-view underfitting mechanism despite retaining pixels.

The review's load-bearing challenge was that raw capacity is not established as the limiter: EXP-001 reached near-zero train loss on the weaker data recipe. It requested EXP-004's train-loss-under-N1/M7 trajectory before planning. That value was not persisted in EXP-004's execution/report artifacts, and its raw log was correctly removed. Re-running the accepted baseline only to recover a missing diagnostic would add an unplanned duplicate stochastic run and violate the loop's one-run discipline. Therefore the chosen hypothesis is narrowed: width 2 tests whether capacity specifically under the strong-view plateau is worth its measured update cost; it does not assert that generic clean-data capacity is already proven.

The following actionable refinements are adopted:

1. Re-run the width-16/width-32 fresh-process timing diagnostic at plan time. Concrete plan review subsequently replaced the point-estimate-pinned 26,500 gate with margin-bearing operational feasibility gates so benchmark noise cannot suppress a valid fixed-time experiment.
2. Make tail trajectory shape a primary diagnostic because EXP-004 peaked at epoch 98/99 and width 2 projects only 13-14 weak-tail epochs. Record whether accuracy is still rising at termination, not only scalar best.
3. Pre-register failure routing: an expected-throughput result still rising at termination or showing optimization lag routes to width 1.5; a converged overfit signature routes to separately reviewed width 2 plus `5e-4`; neither modification is added inside EXP-007.
4. Keep `1e-4` weight decay, post-activation blocks, and every data/optimizer setting unchanged so capacity and update exposure are the only net trade.

## Idea Evaluation

| Candidate | Evidence / reasoning | Potential impact | Decision |
|---|---:|---:|---|
| Width-2 ResNet-20 | 4/5 | 4.5/5 | Select conditionally with the tightened timing gate and capacity-hypothesis caveat. Highest ceiling and measured H20 feasibility. |
| Isolated weight decay `5e-4` | 4/5 | 2/5 | Defer. Maximally attributable and compute-neutral, but canonical transfer is weak and likely upside sits near the threshold. |
| RandAugment plus CutMix | 3/5 | 2.5/5 | Reject for this slot. Excellent engineering controls do not offset compounded every-view regularization on the small model. |

Batch 256 is not a finalist: its measured 23,006-step projection lost 42.4% of updates for only 15.3% more examples and failed its own feasibility gates before adversarial review.

## Chosen Idea
**Selected**: Width-2 ResNet-20 on the accepted RandAugment plateau recipe

**Why this idea**:
It offers the largest plausible metric gain while preserving the locally validated N1/M7-through-80% and weak-tail composition. The architecture remains a post-activation ResNet-20 with identical block count and Option-A shortcuts; only feature widths change. Local H20 timing shows that 3.98x parameters cost 1.44x synchronized step time and about 599 MB, making the capacity/update trade feasible but genuinely risky. Unlike CutMix it does not compound input regularization, and unlike weight decay it has a credible ceiling well above the ten-image threshold. The missing EXP-004 train-loss evidence prevents calling capacity a confirmed bottleneck, so the experiment itself is the controlled test of that premise under strong views.

**Hypothesis**:
If a fresh planning-time benchmark confirms operational feasibility, doubling widths to 32/64/128 while retaining every accepted EXP-004 training setting will provide enough additional augmentation-stable representation capacity to outweigh its expected roughly 31% update loss and raise `best_test_acc` from 92.30% to at least 92.40%. A valid no-improvement with accuracy still rising at the final weak-tail epoch will be advisory evidence for width 1.5; a converged train/test gap will be advisory evidence for a separately reviewed width-2-plus-decay follow-up. A null cannot cleanly falsify the unresolved capacity premise.

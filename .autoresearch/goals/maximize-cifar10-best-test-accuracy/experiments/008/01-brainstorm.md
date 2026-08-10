# Brainstorm EXP-008
**Created**: 2026-08-05

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): the CIFAR recipe pairs wider residual models with `5e-4` weight decay and optional block dropout. These are the most directly relevant regularization precedents after width 2 became the local baseline.
- **Original CIFAR ResNet** (<https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf>): the narrow baseline's `1e-4` decay is historically grounded, but it was tuned for a much smaller model and different augmentation/schedule.
- **RandAugment** (`knowledge/papers/randaugment.md`): preserve the accepted N1/M7 strong phase; EXP-004 and EXP-007 show broad invariance is complementary to the weak hard-label tail.

## Experimental History Review

- Current best is 93.55% at `8faf0f3`; success requires at least 93.65%.
- EXP-007 widened stage channels to 32/64/128 and gained 1.25 points despite retaining only 27,143 steps. Its final strong checkpoint improved from EXP-004's 84.60% to 90.08%, establishing capacity under N1/M7 as valuable.
- During EXP-007's weak tail, train-loss EMA fell from 0.2283 near the switch toward roughly 0.04 while clean accuracy plateaued around 93.5%; test loss bottomed near 0.214 and ended at 0.2196. This is advisory evidence that regularization, rather than more narrow-model capacity, is now the next limiter.
- Boundary shortening, label smoothing, and Cutout replacement have failed. Keep the 80% boundary, hard-label weak tail, N1/M7, and width 2 fixed while testing regularization.
- A disposable width-3 H20 probe measured 2,412,730 parameters, 18.983 ms/step, 667.2 MB, and only 15,595 calibrated steps versus width 2. Width 3 is operationally feasible but sacrifices about 42.5% of width-2 updates and leaves only about eight weak-tail epochs.
- Inline diagnosis: EXP-007's large strong-view gain removed evidence for immediate under-capacity at width 2, while declining train loss plus flat/rising test loss suggests excess fit late. The cleanest next question is whether stronger norm control improves the accepted wide model without consuming updates.

## Collected Ideas

<!-- Quick pass: intentionally empty. -->

## Combinations

<!-- Quick pass: intentionally empty. -->

## Candidate Ideas

### Width-3 ResNet-20

**Summary**: Change only `WIDTH_MULTIPLIER = 2` to 3, producing stage widths 48/96/192 and exactly 2,412,730 parameters while preserving every accepted training setting.

**What it targets**: Additional representation capacity after width 2's large gain, testing whether the capacity frontier continues upward.

**Reasoning**: Width 2 gained 1.25 points, so another width step has a real local directional signal. However, measured cost is steep: 18.983 ms/step versus 10.906 ms for width 2, with only about 15,595 calibrated updates and roughly eight weak-tail epochs. This candidate is a quantified high-upside moonshot rather than an assumed free scaling win.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-007 improvement; EXP-008 disposable H20 probe.

**Estimated Effort**: Low implementation effort, medium verification effort. One literal plus repeated timing and exact parameter/padding checks.

**Risk Assessment**: Very high. Width 3 loses about 42.5% of width-2 updates, may be severely under-optimized, gets fewer best-metric samples, and can end before the weak tail settles. The 2.25x parameter increase costs 1.74x step time, a substantially worse trade than width 2 over width 1.

### Width-2 Weight Decay 5e-4

**Summary**: Change exactly `WEIGHT_DECAY = 1e-4` to `5e-4` on the complete accepted width-2 recipe. Keep 1,073,962 parameters, post-activation blocks, N1/M7 through 80%, the weak hard-label tail, LR schedule, momentum, workers, seed, and evaluator unchanged.

**What it targets**: Late excess fit and parameter-norm control in the newly widened model. Width 2 added 804,240 parameters; its tail train-loss EMA continued falling while test accuracy and loss plateaued, suggesting that stronger coupled L2 pressure may improve clean margins without reducing fixed-time work.

**Reasoning**: `5e-4` is the canonical Wide ResNet CIFAR operating point at batch 128, LR 0.1, and momentum 0.9. The exact value is not guaranteed to transfer to this shorter post-activation run, but width 2 now matches the capacity class better than the narrow model considered in EXP-007. The one-literal change uses the same SGD kernel path and should preserve approximately 27,143 steps, giving unusually strong attribution.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-007 `04-analysis.md` and trajectory; EXP-007 `proposals/idea-04.md` for coupled-decay mechanics; original CIFAR ResNet recipe.

**Estimated Effort**: Low. One literal, optimizer-value assertion, ordinary static checks, and one fixed-seed run.

**Risk Assessment**: Medium. Coupled PyTorch SGD decays convolution/linear weights, BatchNorm affine parameters, and classifier bias through momentum. Fivefold pressure during the long high-LR N1/M7 phase may suppress useful wide-model capacity; the WRN precedent used different blocks and about 200 epochs. A regression of several tenths is plausible.

### Width-2 Residual-Block Dropout 0.3

**Summary**: Add `F.dropout(..., p=0.3, training=self.training)` after the first BN/ReLU and before the second convolution in every width-2 residual branch, preserving all accepted data, optimizer, schedule, and shortcut settings. Evaluation automatically disables dropout.

**What it targets**: Feature co-adaptation inside the wider residual branches. Width 2 has fourfold parameters and low late train loss; block-level dropout can regularize internal representations without removing input pixels or changing labels.

**Reasoning**: Wide Residual Networks reports that dropout between residual convolutions can improve wider CIFAR networks. It is more architecture-aware than global target smoothing and avoids EXP-006's information deletion. A disposable synchronized-step preflight must require at least 96% of width-2's 27,143 projected steps because dropout adds an RNG/mask kernel inside the counted region.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-007 results; current `BasicBlock` topology.

**Estimated Effort**: Medium. Small code diff, but exact placement, train/eval behavior, RNG implications, and GPU step-cost gates require care.

**Risk Assessment**: High. N1/M7 already regularizes strongly, nine blocks may not benefit from branch dropout, the added timed kernel can reduce exposure, and p=0.3 may be too strong for a 300-second horizon. It also changes the RNG stream and makes a marginal result less attributable than weight decay.

## Review

The mandatory external Claude review completed successfully with exit code 0 and is preserved in `01-idea-review.md`; no fallback reviewer was used. It selected width-2 weight decay `5e-4` as the only finalist that attacks the diagnosed norm/generalization limiter while preserving all scarce fixed-time updates. Dropout was rejected for stacking another every-view regularizer and timed RNG kernel on N1/M7, while width 3 was rejected for contradicting the regularization diagnosis and projecting only about eight weak-tail epochs.

The reviewer challenged the strength and persistence of the late-overfit evidence and the transfer of a canonical 200-epoch WRN decay value into a 71-epoch post-activation run. The selected experiment is therefore framed as a point test, not a proof that `5e-4` is optimal. The plan must:

1. Treat coupled decay as plateau norm control whose lower-norm state carries into the weak tail, not as a direct late-only regularizer.
2. Pre-register the strong-switch checkpoint versus 90.08%, train-loss EMA near the switch and through the tail, test-loss minimum/final gap, and best/final trajectory as primary mechanism diagnostics.
3. Require at least 26,872 steps (99% of EXP-007) plus unchanged parameter/loader/seed integrity; the literal should be compute-neutral.
4. Interpret a depressed strong checkpoint and elevated tail train loss as `5e-4` underfitting, routing only a future experiment to lower decay. Do not adapt or rerun EXP-008.
5. Keep width 2, N1/M7, 80% boundary, hard-label tail, momentum, LR schedule, and every other line unchanged.

## Idea Evaluation

| Candidate | Evidence / reasoning | Potential impact | Decision |
|---|---:|---:|---|
| Width-2 weight decay `5e-4` | 4/5 | 3/5 | Select. One-literal, compute-neutral norm-control test on a model with genuine new regularization headroom. |
| Width-2 residual-block dropout `0.3` | 3/5 | 3/5 | Defer. Similar target but greater every-view over-regularization, exposure, and attribution risk. |
| Width-3 ResNet-20 | 2.5/5 | 3/5 | Reject for this slot. Poor marginal compute economics and only about eight projected weak-tail epochs. |

## Chosen Idea
**Selected**: Width-2 weight decay `5e-4`

**Why this idea**:
Width 2 raised the baseline to 93.55% but entered a flat tail while train-loss EMA continued falling, creating a credible norm/generalization question. Increasing the already-computed coupled decay scalar is the cleanest way to test it: one literal, unchanged kernels/shapes/RNG consumption, and expected preservation of all 27k updates. The 5e-4 value is grounded in CIFAR Wide ResNet practice but is not assumed optimal for this shorter run; it is a predeclared operating point whose underfit risk will be diagnosed from the full trajectory.

**Hypothesis**:
Increasing coupled SGD weight decay from `1e-4` to `5e-4` on the accepted width-2 recipe will constrain parameter norms during the long high-LR N1/M7 plateau, carry a better-regularized representation into the unchanged weak tail, retain at least 26,872 steps, and raise `best_test_acc` from 93.55% to at least 93.65%. A lower strong checkpoint plus elevated tail train loss will indicate the point is too aggressive and justify only a future lower-decay experiment, not a rerun.

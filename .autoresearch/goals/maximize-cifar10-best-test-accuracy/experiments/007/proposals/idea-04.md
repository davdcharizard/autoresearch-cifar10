# EXP-007 Proposal: Isolated CIFAR Weight Decay 5e-4

## Proposal

Change exactly one literal in accepted EXP-004 `train.py`:

```diff
-WEIGHT_DECAY = 1e-4
+WEIGHT_DECAY = 5e-4
```

Preserve the complete accepted recipe: width-16 ResNet-20, batch 128, hard-label cross-entropy, ordinary SGD momentum 0.9, `lr=0.1` through the first 80% of counted time, N1/M7 RandAugment through the same boundary, explicit strong-worker shutdown, weak crop/flip tail, the `0.01 -> 1e-4` cosine refinement, synchronized timing, evaluation cadence, seed 42, fixed evaluator, and all loader settings.

This is an isolated, compute-neutral test of whether the accepted 0.27M-parameter model is insufficiently norm-regularized under strong input augmentation. It does not claim that `5e-4` is universally better than `1e-4`; it tests a canonical CIFAR residual-network operating point under this repository's specific 98-epoch fixed-time horizon.

## Local Context

The moving baseline is EXP-004 at 92.30% best accuracy, so success requires at least 92.40%. EXP-004 completed 38,358 optimizer steps, 99 reported epochs, 300.0 counted seconds, 340.7 total seconds, 330.1 MB peak VRAM, and 269,722 parameters. Its successful composition was N1/M7 RandAugment through the full 80% high-LR phase followed by a weak hard-label tail.

The local history constrains this experiment:

- Preserve the long high-LR phase: shortening strong augmentation to 75% regressed 0.18 points in EXP-005.
- Preserve N1/M7: replacing it with fixed 16x16 Cutout retained steps but regressed 0.67 points in EXP-006.
- Preserve optimizer exposure: label smoothing's 6.7% step loss coincided with no top-1 gain in EXP-003.
- Avoid capacity/throughput bundling: measured batch 256 loses 42.4% of updates for only 15.3% more synthetic sample exposure and is a preflight no-go.

Weight decay is therefore unusually clean: the optimizer already performs the same coupled decay multiply/add at `1e-4`, so changing its scalar should leave step throughput, sample exposure, worker behavior, memory, and phase timing effectively identical.

## Directional Case for 5e-4

The original CIFAR ResNet recipe used weight decay `1e-4`, momentum 0.9, batch 128, initial LR 0.1, simple crop/flip, and 64,000 updates. The current code inherits that decay but executes only about 38,000 updates and adds substantially stronger RandAugment. This is evidence that `1e-4` is historically grounded, not evidence that it remains optimal after the schedule and data recipe changed.

The Wide Residual Networks CIFAR recipe used initial LR 0.1, momentum 0.9, weight decay `5e-4`, batch 128, and 200 epochs. It established `5e-4` as a serious CIFAR residual-network setting rather than an arbitrary fivefold jump. Its models were wider and trained about twice as long as this ResNet-20, so the value cannot be transferred uncritically.

Primary sources:

- He et al., *Deep Residual Learning for Image Recognition*, specify `1e-4` weight decay for the original CIFAR residual experiments: <https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf>.
- Zagoruyko and Komodakis, *Wide Residual Networks*, use a CIFAR recipe with initial LR 0.1, momentum 0.9, `5e-4` weight decay, and 200 epochs: <https://arxiv.org/abs/1605.07146>.

The positive mechanism is norm control. N1/M7 forces the network to fit many transformed views, but it does not directly constrain parameter scale. Stronger L2 pressure can reduce reliance on large, brittle filters and BatchNorm scales, discourage co-adaptation to augmentation artifacts, and improve clean-test margins during the weak tail. It may also keep the small model in a smoother region while the long `lr=0.1` plateau explores.

There is also a real negative mechanism. EXP-004's model is only 269,722 parameters, strong augmentation makes plateau training difficult, and the run has roughly 98 full epochs rather than the 164 epochs of the original 64k-step ResNet schedule or the 200 epochs of WRN. Fivefold stronger decay may reduce useful capacity before the short 20% tail can recover. The proposal is valuable because it isolates which regime applies without consuming updates or introducing a second regularizer.

## PyTorch SGD Shrinkage Reasoning

This code uses coupled L2 decay in `torch.optim.SGD`, not decoupled SGD-W. Conceptually, for parameter `p`, gradient `g`, momentum buffer `b`, momentum `mu=0.9`, LR `eta`, and decay `lambda`:

```text
g <- g + lambda * p
b <- mu * b + g
p <- p - eta * b
```

The candidate makes the decay force exactly five times larger on every optimizer step. It applies to every parameter passed to the single optimizer group, including convolution/linear weights, BatchNorm affine parameters, and the classifier bias. No modern no-decay exception for norm or bias parameters is added, because that would violate the one-literal scope.

Two cumulative views show why this is a substantive change:

1. Ignoring momentum amplification and task gradients, parameter-only shrink is approximately `exp(-lambda * sum(eta_t))`. EXP-004 has about 30,686 plateau steps at LR 0.1 and about 7,672 cosine-tail steps whose mean LR is approximately 0.00505. Thus `sum(eta_t) ~= 3,107`. The direct-decay surrogate retains `exp(-0.311) ~= 73%` at `1e-4`, versus `exp(-1.554) ~= 21%` at `5e-4`.
2. With a slowly varying parameter, the decay component of the momentum buffer approaches `lambda * p / (1-mu)`, a 10x amplification at momentum 0.9. The plateau's steady-state per-step decay component is therefore about `eta * lambda / (1-mu)`: `1e-4` at the current setting and `5e-4` for the candidate. A parameter-only exponential extrapolation would be much stronger (`exp(-3.1)` versus `exp(-15.5)` over the whole schedule).

Neither extrapolation predicts literal parameter collapse: supervised gradients continually drive weights, parameters and BatchNorm scales are not stationary, residual/BatchNorm networks have scale interactions, and momentum mixes task and decay gradients. They do establish that `5e-4` is not a cosmetic scalar change. It materially shifts the equilibrium between data fit and norm pressure, especially during the 80% high-LR plateau. The low-LR tail automatically weakens the instantaneous shrink because the update remains proportional to LR; this is useful because clean refinement should dominate late.

## Exact Scope and Implementation

The only behavioral edit is:

```python
WEIGHT_DECAY = 5e-4
```

Do not:

- Create optimizer parameter groups or exempt BatchNorm/bias.
- Change LR, momentum, batch size, model, loss, augmentation magnitude, switch fraction, or tail.
- Add dropout, smoothing, Mixup, CutMix, EMA, AMP, compilation, or architecture changes.
- Add per-step norm logging inside the counted region.
- Change evaluation frequency, timing, seed, or worker lifecycle.

The existing optimizer construction already consumes `WEIGHT_DECAY`, so no other code edit is required.

## Throughput Equivalence

The candidate executes the same optimizer kernel path and number of tensor operations as EXP-004. Multiplying the already-computed decay term by a different scalar has no meaningful compute or memory cost. Expected full-run exposure is:

- 37,975-38,742 steps, a +/-1% band around EXP-004's 38,358.
- Approximately 97-100 reported epochs and 4.86-4.96M presented samples.
- Approximately 330 MB peak VRAM.
- 300 counted seconds and roughly 335-350 total seconds.
- One N1/M7-to-weak switch near 80.0%, with eight old workers stopped.

Require at least 37,975 steps (99% retention) for throughput equivalence. A larger step loss cannot plausibly be caused by the scalar and indicates environment drift, an unintended diff, or a measurement problem. No GPU throughput microbenchmark is necessary because the operation graph and shapes are identical, but a short disposable optimizer API check may confirm the literal reaches `optimizer.param_groups[0]["weight_decay"] == 5e-4`.

## Hypothesis and Expected Impact

**Hypothesis:** increasing coupled SGD weight decay from `1e-4` to `5e-4` will improve norm control and clean-test margins under the accepted strong-augmentation plateau without reducing optimizer exposure, raising `best_test_acc` from 92.30% to at least 92.40%.

Expected best accuracy is 92.35-92.60%, with 92.40-92.50% the most plausible successful band. This is a low-cost, medium-risk candidate: its upside is likely a few tenths rather than a full point, while a 0.2-0.5 point regression is plausible if the small model becomes underfit. The acceptance threshold is close enough that one valid fixed-seed run is informative under the repository protocol, but no reroll is allowed.

Useful trajectory signals:

- If the strong-phase checkpoint is near EXP-004's 84.60% and the weak-tail peak improves, stronger decay likely improved late clean margins without harming representation learning.
- If the strong checkpoint and early weak-tail recovery are both depressed, the model is underfit under N1/M7.
- If final loss rises while accuracy is flat, the stronger norm constraint changed confidence/fit but did not move top-1 boundaries; reject it for this goal.
- If train-loss EMA is much higher throughout the tail and test accuracy falls, the fivefold value is too strong for this capacity/horizon.

## Risks

- **Small-model underfitting.** The model has only 0.27M parameters and already sees strong augmentation. This is the primary risk.
- **Short horizon versus the WRN precedent.** WRN's `5e-4` setting accompanies larger models and 200 epochs; this run has about 98. More regularization with less optimization time may be mismatched.
- **Decay of BatchNorm affine terms and bias.** The single optimizer group applies 5e-4 everywhere. Suppressing BN scales may change effective feature amplitudes more than weight-only decay would.
- **Interaction with the long high-LR plateau.** Coupled decay is multiplied by LR and accumulated through momentum, so most shrinkage pressure occurs during the same 80% phase where N1/M7 makes fitting hardest.
- **Strong augmentation may already provide enough regularization.** EXP-004 gained 0.47 points from RandAugment; stacking stronger norm pressure can cross from complementary to redundant/aggressive.
- **Canonical-value transfer.** `5e-4` is well established for CIFAR WRNs but not validated for this exact post-activation ResNet-20, schedule, or evaluator.
- **Single-run stochasticity.** Changing the scalar does not intentionally change RNG consumption, so fixed seed and data order should be much more comparable than augmentation changes. CUDA nondeterminism can remain; do not reroll.

## Confound Controls

- Diff against accepted commit `11f8469` and require exactly one behavioral line changed.
- Keep seed 42 and all augmentation/loader code byte-for-byte identical, preserving the same intended random stream.
- Keep optimizer type, momentum, LR schedule, and all parameter membership unchanged.
- Keep exactly 269,722 parameters, batch 128, 390 batches per full epoch, and hard labels.
- Keep the same elapsed-time phase boundary and accepted evaluation cadence.
- Run one full experiment only if this proposal is selected; do not sweep 2e-4/3e-4/4e-4 after observing its result.

## Verification Plan

No full training run is part of proposal development. If selected:

1. Confirm the moving baseline is 92.30% and improvement requires at least 92.40%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM.
3. Verify the tracked diff modifies only `train.py` and contains exactly the `1e-4 -> 5e-4` literal change.
4. Instantiate the optimizer in a disposable check and assert one parameter group with `weight_decay == 5e-4`, momentum 0.9, and initial LR 0.1.
5. Run syntax compilation, Ruff, pre-commit, and assert 269,722 model parameters and 390 loader batches.
6. Remove stale logs and run once as `uv run train.py > run.log 2>&1` under the 600-second supervisor.
7. Require exit zero, one complete finite ten-field summary, approximately 300 counted seconds, total below 600 seconds, and unchanged parameter count.
8. Require exactly one `randaugment->base` switch near 80.0%, eight workers stopped, unique evaluation epochs, and a terminal evaluation matching the summary epoch.
9. Require at least 37,975 steps for throughput equivalence; compare actual steps, epochs, samples, VRAM, and total time with EXP-004.
10. Require `best_test_acc >= 92.40%` for an improvement verdict. Compare strong switch accuracy, early weak recovery, best/final gap, final loss, and train-loss EMA with EXP-004.
11. Remove `run.log` after analysis.

## Decision Rules

- **Accept:** accuracy at least 92.40%, all integrity checks pass, and at least 37,975 steps complete. Promote `5e-4` as the new accepted decay.
- **No improvement with equivalent throughput:** reject `5e-4` for the accepted small-model/strong-augmentation recipe. Do not infer that intermediate decay values would succeed without a new brainstorm.
- **Underfitting signature:** lower switch accuracy, weaker weak-tail recovery, elevated final loss, and lower top-1 indicate excessive combined regularization. Revert to `1e-4`.
- **Step shortfall or unexpected runtime change:** treat as a confounded or invalid comparison; inspect environment/diff because the literal should be compute-neutral.
- **Crash, timeout, lifecycle, or scope failure:** invalid. Revert to accepted EXP-004 and diagnose without changing decay or seed in place.

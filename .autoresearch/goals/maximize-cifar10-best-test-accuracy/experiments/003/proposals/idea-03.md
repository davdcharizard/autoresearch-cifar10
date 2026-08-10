# EXP-003 Proposal: 1.5x-Width ResNet-20 as an Isolated Capacity Test

## Summary

Increase only the channel width of the current CIFAR ResNet-20 from `16/32/64` to `24/48/96`, while preserving the successful EXP-002 optimizer, elapsed-time schedule, data pipeline, augmentation, batch size, evaluation policy, residual-block ordering, shortcut implementation, and seed exactly. This produces a 20-layer network with exactly 605,026 parameters, 2.24x the current 269,722, while remaining very small relative to the available NVIDIA H20.

The proposal deliberately does **not** combine widening with preactivation, learned projection shortcuts, AMP, channels-last, compilation, larger batches, or stronger regularization. Those may be useful later, but combining them would prevent EXP-003 from answering the immediate question: is representation capacity an orthogonal limiter once the long-plateau schedule is fixed?

## Diagnosis and Rationale

EXP-002 established the current moving baseline at 91.83% using 38,629 steps, 100 epochs, 300.0 counted training seconds, 336.0 total seconds, 330.1 MB peak VRAM, and 269,722 parameters. Its key result is that the 80%-of-budget `lr=0.1` plateau followed by an explicit step to `0.01` and cosine refinement to `1e-4` improved the original baseline by 0.16 percentage points. That schedule should now be treated as a controlled constant.

The current network uses less than 0.4% of the H20's 97,871 MiB and has only about 0.27M trainable parameters. VRAM is plainly not constraining model size. The plausible constraint is instead a three-way tradeoff:

1. More channels can learn a richer set of CIFAR-10 features and may reduce representation bias.
2. Convolutional work grows approximately with width squared, so a wider model completes fewer updates and epochs in the fixed 300 seconds.
3. The current model is so small that launch, synchronization, and input overhead are a meaningful part of its 7.8 ms baseline step; on an H20, a 2.25x-FLOP model may cost materially less than 2.25x wall time per step.

A 1.5x width multiplier is the conservative point on this curve. It gives a meaningful 2.24x parameter/compute increase while avoiding the roughly 4x cost of a 2x-width `32/64/128` network. Widths 24, 48, and 96 are also multiples of eight, a favorable shape family for accelerator convolution kernels. The experiment therefore tests whether unused accelerator capacity can buy better representations without giving up too much of EXP-002's productive training horizon.

## Exact Architecture Candidate

Retain the current post-activation `BasicBlock` unchanged:

```text
3x3 conv -> BatchNorm -> ReLU -> 3x3 conv -> BatchNorm
        + identity / stride-and-zero-pad shortcut
        -> ReLU
```

Change only the widths supplied by `ResNet`:

```text
Stem:    3x3 conv, 3 -> 24, stride 1; BatchNorm; ReLU
Stage 1: 3 BasicBlocks, 24 -> 24, spatial size 32x32
Stage 2: 3 BasicBlocks, 24 -> 48 in first block at stride 2, then 48 -> 48
Stage 3: 3 BasicBlocks, 48 -> 96 in first block at stride 2, then 96 -> 96
Head:    adaptive average pool -> Linear(96, 10)
```

Keep the existing parameter-free stride-and-zero-pad shortcuts at stage transitions. Do not add learned 1x1 projections. Keep Kaiming-normal initialization, convolution bias settings, BatchNorm defaults, final pooling, and classifier behavior unchanged.

For base width `w`, this exact implementation has `1044*w^2 + 153*w + 10` parameters. At `w=24`, the expected summary is therefore `num_params: 605,026`. This deterministic count is an implementation-integrity check.

## Mechanism

Widening increases the number of feature channels available at every spatial scale. Compared with the current model, each residual transformation can encode more local patterns and preserve more class-relevant alternatives before downsampling. The final representation grows from 64 to 96 dimensions. Because depth and residual topology remain unchanged, optimization path length does not increase; the intervention spends compute on parallel representation capacity rather than additional sequential blocks.

This mechanism is supported by two relevant primary sources:

- Zagoruyko and Komodakis, *Wide Residual Networks* (BMVC 2016), report that widening residual blocks consistently improved CIFAR performance across several depths and argue that width can be a more efficient capacity lever than adding many layers. Their tested networks and training recipes are much larger than this experiment, so this supports the direction, not a numerical guarantee: <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>.
- He et al., *Identity Mappings in Deep Residual Networks* (ECCV 2016), show that preactivation can ease optimization and improve generalization in very deep residual networks: <https://arxiv.org/abs/1603.05027>. EXP-003 intentionally defers preactivation because changing activation order together with width would confound the capacity test, and its evidence is strongest at depths far beyond ResNet-20.

The most direct local evidence is the remaining hardware headroom: EXP-002 used 330.1 MB peak VRAM on a 97,871 MiB H20. Capacity can increase substantially before memory becomes relevant. The uncertain resource is the number of examples and updates that fit in 300 seconds, which EXP-003 will measure.

## Preserved EXP-002 Training Recipe

The following must remain unchanged from the current `train.py`:

- Batch size 128 and `drop_last=True`.
- Standard random crop with four-pixel padding and random horizontal flip.
- CIFAR-10 mean and unit standard deviation.
- SGD at `lr=0.1`, momentum 0.9, no Nesterov, and weight decay `1e-4`.
- The elapsed-time schedule: hold `lr=0.1` through 80% of counted training time, step to approximately `0.01`, then cosine-decay to `1e-4` by the end.
- The 300-second counted training budget, `MAX_STEPS`, per-step timing/synchronization, and summary fields.
- Persistent DataLoader workers.
- The existing early checkpoints and dense once-per-epoch evaluation in the final 20%, including terminal evaluation.
- Seed 42 for CPU and CUDA.

In particular, do not scale the learning rate simply because model width changed; batch size is unchanged. Do not raise weight decay preemptively for the larger model. Either adjustment could be useful in a follow-up, but would make the first capacity result ambiguous.

## Expected Throughput and Step Tradeoff

Convolution parameters and multiply-accumulate work scale approximately as width squared. Moving from base width 16 to 24 therefore raises the dominant convolutional work by about `(24/16)^2 = 2.25x`. A naive compute-bound projection would increase step time from about 7.8 ms to 17.5 ms, yielding only about 17,200 steps and 44 full epochs. That is a pessimistic bound, not the expected result: the tiny current network pays fixed Python, kernel-launch, synchronization, transfer, and input costs, and likely underutilizes the H20.

The practical expectation is 10-13 ms per step, 23,000-30,000 steps, and approximately 60-77 epochs in 300 seconds. An optimistic outcome near 9 ms would retain roughly 33,000 steps and 85 epochs. Peak VRAM should remain well below 1 GB because parameter memory is negligible and activation widths increase only 1.5x at the same batch size.

The successful schedule is based on elapsed time, so its 80/20 exploration/refinement allocation remains exact even when the step count changes. The scientific tradeoff is therefore explicit: the wider network receives fewer SGD updates, but every phase gets the same fraction of the fixed budget. Report completed steps and epochs alongside accuracy to determine whether a negative result reflects insufficient horizon rather than evidence that width itself is useless.

## Hypothesis and Expected Benefit

**Hypothesis:** a 1.5x-wide ResNet-20 will retain enough H20 throughput to complete at least about 65 epochs while its 2.24x parameter budget improves learned CIFAR-10 representations, raising `best_test_acc` from the moving baseline of 91.83% to at least 91.93% under the unchanged EXP-002 schedule.

The expected first-run range is 91.95-92.30%, or +0.12 to +0.47 percentage points. A gain in this range would establish model capacity as a productive orthogonal lever. The estimate is intentionally conservative: published wide-network gains come from different, longer recipes, while this run trades away optimizer updates under a strict time budget.

The most informative non-improvement outcomes are:

- Similar or lower accuracy with at least 75 epochs: capacity alone is probably not the current limiter, or the wider model needs stronger regularization.
- Lower accuracy with fewer than 60 epochs: the test is compute-horizon limited; a base width of 20 or a throughput optimization should be tested before rejecting width.
- Better training loss but unchanged test accuracy: the extra capacity is fitting the training distribution and should next be paired with an isolated regularizer, not widened further.

## Risks

- **Too few updates in 300 seconds.** This is the main risk. Width 24 is chosen instead of width 32 to bound it. The run remains valid even if slower; steps and epochs explain the result.
- **Overfitting.** A 605k-parameter network is still small, but added capacity may lower train loss without improving the test metric. Keep weight decay fixed for attribution, then consider one isolated regularizer only after observing this failure mode.
- **The schedule's absolute learning rate may not be ideal for the wider model.** Batch size, not width, usually drives first-order LR scaling, so keeping `0.1` is the clean control. A follow-up may tune LR if loss curves show under-optimization.
- **BatchNorm behavior can change with channel count.** Batch size stays 128, so per-channel statistics retain the same sample count; no special handling is warranted.
- **Published WRN results may not transfer to this shallow, fixed-time setting.** The proposal uses the paper as directional evidence and explicitly measures the opposing throughput cost.
- **Combining preactivation would obscure attribution.** It is deferred. If width succeeds, preactivation can be tested on the accepted wider baseline; if width fails with adequate epochs, preactivation at width 16 is an orthogonal representation experiment.
- **A 0.1-point threshold is close to ordinary run variance.** The seed must remain fixed and there must be no retry or reroll. Improvement is judged using the repository's specified single-run protocol.

## Implementation Sketch

Only `train.py` changes:

```python
BASE_WIDTH = 24

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, width=BASE_WIDTH):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, width, num_blocks, stride=1)
        self.layer2 = self._make_layer(width, 2 * width, num_blocks, stride=2)
        self.layer3 = self._make_layer(2 * width, 4 * width, num_blocks, stride=2)
        self.fc = nn.Linear(4 * width, num_classes)
        self.apply(self._weights_init)
```

No changes are needed in `BasicBlock`: its existing channel-difference calculation automatically pads 24 channels at each stage transition. Update the model-identification printout to include `width=24`, but preserve all required summary keys and formatting.

Before execution, statically assert the parameter count is 605,026. A mismatch means the architecture is not the isolated candidate described here.

## Confound Controls

- Diff against the accepted EXP-002 commit and verify that only channel construction and descriptive logging changed.
- Keep `NUM_BLOCKS=3`, so depth remains ResNet-20.
- Keep all optimizer and schedule constants byte-for-byte identical.
- Keep batch size, augmentation, loader, timing, evaluation cadence, and seed identical.
- Do not introduce projection convolutions, dropout, label smoothing, Mixup, RandAugment, EMA, AMP, compilation, or channels-last in this experiment.
- Run once on the required single H20. Do not retry a valid run for a more favorable seed outcome.

## Verification Plan

1. Confirm the moving baseline is 91.83% from `04-results.tsv`; the minimum valid improvement is therefore 91.93%.
2. Confirm exactly one NVIDIA H20 with approximately 98 GB VRAM is visible.
3. Verify `git diff` changes only `train.py` and matches the width-only implementation sketch.
4. Run syntax compilation, Ruff, and the repository's pre-commit checks without altering `prepare.py` or dependencies.
5. Confirm no stale `run.log` exists, then execute exactly `uv run train.py > run.log 2>&1` under the 600-second supervisor.
6. Require a zero exit code and one complete numeric summary with `training_seconds` approximately 300, `total_seconds < 600`, and `num_params: 605,026`.
7. Require `best_test_acc >= 91.93%` for an improvement verdict.
8. Record step time, `num_steps`, and `num_epochs`. Compare them with EXP-002's 38,629 steps and 100 epochs to quantify the cost of width.
9. Check peak VRAM against EXP-002's 330.1 MB. An increase is expected and acceptable; an unexpectedly multi-gigabyte result should trigger an implementation review.
10. Inspect the late evaluation trajectory and final train-loss EMA. Classify a failure as likely horizon-limited, optimization-limited, or overfit using the outcomes above rather than treating every miss as evidence against capacity.

## Decision Rule After the Run

- **Accept** if accuracy reaches at least 91.93% with all integrity checks passing. The 1.5x-width model becomes the moving baseline, and later regularization experiments should preserve it unless their runtime interaction is prohibitive.
- **Reject width 24 as currently configured** if it completes at least 75 epochs but misses 91.93%. This is enough horizon to make the capacity-only result informative; revert the architecture before testing another orthogonal idea.
- **Treat as throughput-inconclusive** if it completes fewer than 60 epochs. Revert for the official verdict, but preserve the learning that FP32 batch-128 widening is too expensive; the next capacity test should use width 20 or first isolate a sanctioned throughput optimization.
- **Do not tune within EXP-003.** Any width, weight-decay, LR, or preactivation follow-up is a new experiment, preventing adaptive retries from contaminating this single-run result.

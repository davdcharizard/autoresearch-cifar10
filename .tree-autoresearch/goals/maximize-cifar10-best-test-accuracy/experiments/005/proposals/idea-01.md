# Proposal: Identity-Initialized ECA in Every WRN Residual Branch

## Summary

Add Efficient Channel Attention (ECA) to the residual output of each of the six
PreAct WRN-16-4 blocks in the accepted EXP-004 model. Each gate uses a fixed
three-tap 1D channel kernel after spatial global averaging. Place the gate after
the block's second 3x3 convolution and before stochastic depth and residual
addition.

Use a zero-initialized kernel and the neutral gate parameterization
`2 * sigmoid(logits)`. The initial gate is therefore exactly one, so the model
starts as the parent function rather than halving every residual branch as a
raw sigmoid gate would. Implement the kernel directly as a three-element
`nn.Parameter` and call `F.conv1d`; this adds exactly 18 parameters across six
blocks and consumes no RNG during construction. The existing seed-42 parent
initialization can remain bit-identical for all shared parameters.

Everything else remains unchanged: front-loaded CutMix, depth-dependent drop
path and its late annealing, clean-tail period-two SAM, SGD/Nesterov, the
time-indexed LR schedule, batch 256, BF16 channels-last execution, one
evaluation per epoch, and the fixed 300-second charged budget.

## Motivation and Evidence

The current parent is already a strong, validated stack. EXP-004 reaches
95.40%, improves its EXP-002 parent by 0.17 points, finishes at the same 95.40%
accuracy with 0.1654 loss, and retains 25,560 optimizer steps despite periodic
late SAM. Its 2,748,890 parameters and 1,190.5 MiB peak allocation leave ample
memory headroom on the 98 GB H20. Narrow tuning of CutMix and drop-path strength
failed confirmation in EXP-003, so the next mechanism should change learned
representation rather than revisit a nearby regularization scalar.

`experiments/005/papers/eca-net.md` describes a low-cost channel-attention
mechanism that preserves direct channel information: global-average-pool each
feature channel, use a short 1D convolution to communicate between neighboring
channels, apply a sigmoid, and multiply the original feature map by the
resulting channel gates. The CVPR 2020 paper reports only 80 parameters and
`4.7e-4` GFLOPs of overhead for a ResNet-50 integration while improving
ImageNet top-1 by more than two points over its plain backbone. Those numbers
do not establish the effect in this CIFAR-10 WRN or short wall-clock regime,
but they support ECA as an orthogonal, plausible, low-arithmetic-cost
representation intervention.

The parent's residual branches currently treat all channels uniformly after
their second convolution. ECA lets each block modulate channels based on the
whole image while avoiding the dimensionality-reduction bottleneck of
squeeze-and-excitation. This may be useful after CutMix has encouraged diverse
part-level features and before SAM biases the clean tail toward flatter
solutions.

## Exact Architecture Change

### ECA module

Add one minimal module:

```python
class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ECA kernel size must be odd")
        self.kernel = nn.Parameter(torch.zeros(1, 1, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x):
        descriptor = x.mean(dim=(2, 3)).unsqueeze(1)  # [N, 1, C]
        logits = F.conv1d(descriptor, self.kernel, padding=self.padding)
        gates = 2.0 * torch.sigmoid(logits)
        gates = gates.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * gates
```

Use `kernel_size=3` for every width. The parent's residual outputs have 64,
128, or 256 channels; the experiment paper identifies a fixed odd kernel of
three as the conservative setting at these small widths. Do not add a bias,
channel bottleneck, MLP, reduction ratio, dropout, or separate attention
hyperparameter schedule.

The raw parameter tensor is intentional. Constructing an `nn.Conv1d` invokes a
random default initializer before any later zeroing and would advance the
global RNG, changing shared parent weights under seed 42. `torch.zeros` does not
consume RNG. The current model's `_weights_init` ignores a bare parameter, so
the ECA kernels remain exactly zero after model construction while all shared
Conv2d, BatchNorm, and Linear initialization retains the parent's draw order.

### Block integration

In `PreActWideBlock.__init__`, add
`self.eca = EfficientChannelAttention(kernel_size=3)`. In `forward`, change
only the residual branch tail:

```python
out = self.conv2(F.relu(self.bn2(out)))
out = self.eca(out)

drop_prob = self.drop_prob * drop_scale
# existing stochastic-depth logic follows unchanged
return shortcut + out
```

Apply this to all six blocks, including the three shape-changing first blocks
and the three same-shape second blocks. Attention always sees the output width
of the residual branch, and the shortcut is never gated. The gate precedes
stochastic depth so the existing mask still decides whether the complete
learned residual transformation is present for each example.

At initialization, every ECA logit is zero and every gate is exactly one,
making each block's forward output identical to EXP-004 for the same input and
drop-path mask. The derivative of `2 * sigmoid(z)` at zero is 0.5, so the
zero-initialized kernel is trainable rather than frozen. Its output range
`(0, 2)` allows both attenuation and modest amplification while preserving the
parent scale at initialization.

### Parameter and logging changes

Six three-tap kernels add 18 trainable parameters, increasing `num_params` from
2,748,890 to exactly 2,748,908. Keep the existing Kaiming/BatchNorm
initialization code unchanged. Add a concise setup field such as
`eca_kernel=3 eca_blocks=6 eca_gate=2sigmoid` so the run artifact proves which
variant executed. No per-step gate-statistic logging is needed because `.item()`
synchronization would contaminate charged throughput.

## Interaction with the Parent Stack

### CutMix

CutMix remains active with probability 0.5 only before progress 0.75 and uses
its dedicated seed-42 CPU/CUDA generators. ECA is deterministic and consumes no
random draws, so it does not disturb data-loader, CutMix, or stochastic-depth
RNG streams. During the early phase, attention learns from both clean and mixed
images using the existing area-corrected loss. No CutMix logic, exposure,
geometry, or cutoff changes.

CutMix can make the image-level channel descriptor reflect two source images;
that is a real interaction, but the gate is learned jointly under the validated
mixed supervision. The clean final quarter then refines the attention kernels
on unmixed examples.

### Drop path

The maximum depth-dependent drop probability stays 0.08 and its scale remains
one through progress 0.75 before annealing to zero. Because ECA is inside the
residual branch and before the mask, a dropped residual contributes neither
features nor ECA gradient for that example, exactly as other branch parameters
behave. The shortcut path and stochastic-depth survival probabilities are
unchanged. With batch 256 and at most 8% dropping, every ECA kernel should still
receive a gradient on each ordinary training step.

### Period-two SAM

The ECA kernels are ordinary trainable parameters, so the current
`sam_parameters` and `sam_snapshots` comprehensions include all six
automatically. On scheduled clean-tail steps, the first backward computes ECA
gradients, the global FP32 norm includes them, `sam_perturb` moves them with the
rest of the model, and `restore_sam_parameters` restores them exactly before
the sole Nesterov update. The second pass reuses the existing CUDA RNG replay
and BatchNorm tracking suppression; ECA itself has no buffers or randomness.

Do not special-case the kernels out of SAM. Excluding them would optimize the
backbone and attention under different objectives and weaken the claimed
parent-plus-ECA comparison. The 18 added scalars are too few to materially
change snapshot memory or the foreach perturbation cost.

## Timing and Budget Compatibility

All ECA work occurs inside `model(inputs, ...)`, hence inside the current `t0`
to CUDA-synchronize charged interval. It is also charged on SAM's second
forward for every second eligible clean-tail step. There is no extra data pass,
backward pass, evaluation, compilation step, or uncharged recalibration.

Per block, ECA adds one spatial reduction, a length-C three-tap 1D convolution,
a sigmoid, and a broadcast multiply. Across the six blocks it adds only 18
weights and tiny arithmetic relative to the WRN convolutions. The principal
risk is kernel-launch and feature-map memory traffic, not FLOPs. Based on the
parent's 25,560 steps, budget for a 1-4% throughput reduction and expect roughly
24,500-25,300 steps in 300 charged seconds. Peak VRAM should remain within a few
MiB of 1,190.5 MiB; the gates and descriptors are small relative to existing
activations.

The parent completed in 457.3 total seconds with 132 once-per-epoch
evaluations. ECA does not change evaluation cadence, so the total run should
remain below the mandatory 600-second timeout. Use physical GPU 0 only via
`CUDA_VISIBLE_DEVICES=0`.

## Expected Effect and Hypothesis

The accepted parent metric is 95.40%, so a valid improvement requires at least
95.50%. The testable hypothesis is:

> Identity-initialized ECA in all six residual branches will improve
> `best_test_acc` to 95.50-95.75% while retaining at least 24,500 optimizer
> steps, because adaptive channel interactions add representational selectivity
> without disrupting the validated parent at initialization or consuming a
> material fraction of the wall-clock budget.

The expected gain is deliberately much smaller than the ECA paper's ImageNet
result. This parent is already at 95.40%, CIFAR-10 has only ten classes, and the
exact PreAct WRN/CutMix/SAM regime is not covered by the paper. A result below
95.50% is a no-improvement even if loss or qualitative gate behavior improves.
Do not retry with a different kernel or gate scale selected from the same test
metric.

## Failure Modes

- **Effect below the 0.10-point gate:** ECA may add little on CIFAR-10 because
  global channel selection is already representable by the WRN and final global
  pooling. This is the primary scientific risk.
- **Launch-bound slowdown:** six tiny reductions/convolutions/sigmoids per
  forward can cost more than their FLOP count suggests at 32x32 resolution. SAM
  repeats this overhead on half of the last-quarter steps. A step count below
  roughly 24,000 would indicate that all-block ECA is too expensive for this
  budget even if VRAM is negligible.
- **Neutral parameterization differs from the paper:** `2 * sigmoid` allows
  amplification up to two and is not the paper's raw `(0, 1)` gate. Raw sigmoid
  would scale every residual branch by 0.5 initially, confounding attention
  with a large residual-amplitude change. The identity-preserving variant is
  chosen to make this a safer parent extension, but its effect size is less
  directly supported by the cited result.
- **Slow symmetry breaking:** all three kernel taps start at zero. Gradients are
  nonzero at the identity point, but the gate may remain near one if descriptor
  correlations are weak or weight decay/SAM suppresses the tiny kernel.
- **Local channel adjacency may be arbitrary:** convolution assumes neighboring
  learned channels contain useful local relationships. The network can adapt
  channel ordering during training, but a short budget may limit that
  co-adaptation.
- **Residual/drop-path interaction:** gating before stochastic depth means
  attention gradients are absent for dropped examples. The low maximum drop
  rate bounds this effect; moving ECA after residual addition would instead
  alter the identity path and is not proposed.
- **SAM sensitivity:** the new parameters join the global perturbation norm.
  Their small count should be immaterial, but a missing/zero gradient would
  trigger the parent's deliberate SAM assertion. The full-model smoke test must
  cover a scheduled SAM step before the run.
- **Measurement variance:** EXP-003 showed 0.14-0.29-point reversals after
  selection. This proposal fixes one architecture in advance and uses one
  seed-42 run; no ECA variant search or metric-driven retry is permitted.

## Minimal Verification

### Pre-run deterministic checks

1. Instantiate parent and ECA blocks with shared Conv2d/BatchNorm state. In eval
   mode, verify the zero-kernel ECA block is bit-identical to the ungated block
   for the same input; separately assert all initial gates equal exactly one.
2. Backpropagate a nonconstant scalar loss through a tiny ECA block and verify
   each three-tap kernel has a finite, nonzero gradient.
3. With seed 42, compare all shared state tensors between parent and ECA model
   construction and verify they are identical. Confirm the ECA construction
   adds no RNG consumption and `num_params == 2_748_908`.
4. Run the existing full-WRN BF16/channels-last SAM smoke with ECA present.
   Verify every trainable parameter has a gradient, the SAM perturbation norm is
   0.05, ECA parameters are perturbed and restored exactly, CUDA RNG parity is
   preserved, BatchNorm buffers update once, and the optimizer updates once.

### Full run

1. Confirm physical GPU 0 is the approximately 98 GB NVIDIA H20. Launch exactly
   once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
2. Confirm the config line reports six kernel-3 identity-initialized ECA gates,
   parent CutMix/drop-path/SAM constants are unchanged, and parameter count is
   exactly 2,748,908.
3. Confirm CutMix exposure remains near 0.5 before progress 0.75, SAM begins at
   progress 0.75 and applies on every second eligible one-based step, and no RNG
   or BatchNorm invariant fails.
4. Confirm all attention computation is included in charged step timing,
   `training_seconds` is approximately 300, total runtime is below 600 seconds,
   the run retains approximately 24,500 or more steps, and validation occurs no
   more than once per epoch.
5. Parse the complete summary and verify `best_test_acc >= 95.50%`, finite final
   loss, no traceback/CUDA error, and reasonable peak VRAM. Accuracy is the
   verdict criterion; the step expectation is diagnostic rather than a hard
   goal constraint.
6. Verify only `train.py` changed during implementation, `prepare.py` and the
   evaluator remain untouched, global and CutMix seeds remain 42, and no
   dependencies were added. Remove the transient run log after analysis.


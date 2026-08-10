# Proposal: Identity-Preserving ECA in Every WRN Residual Branch

## Summary

Add one Efficient Channel Attention (ECA) gate to the residual output of each
of EXP-004's six PreAct WRN-16-4 blocks. Place attention after the second 3x3
convolution and before stochastic depth and residual addition. Use independent
channel kernels per block, with kernel size 3 for the two 64-channel blocks and
size 5 for the 128- and 256-channel blocks.

Standard ECA uses `sigmoid(logits)`, which produces a gate near 0.5 at a
zero-centered initialization and would halve all six residual branches at the
start. That is a large optimization and residual-scale intervention in this
short schedule. Instead, initialize every ECA kernel to zero and use
`2 * sigmoid(logits)`. The initial gate is exactly one, preserving the parent
function, while its derivative at zero is 0.5, so the kernel receives a
first-step gradient and is not inert.

This is a fixed, additive architecture experiment. Preserve EXP-004's complete
independent-image stream, CutMix gate, drop-path schedule, period-two clean-tail
SAM, optimizer, LR, BF16/channels-last execution, seed 42, and evaluation
cadence. The six gates add exactly 26 parameters, taking the model from
2,748,890 to 2,748,916 parameters.

## Why It Could Produce a Detectable Gain

The current bottleneck is representation/generalization, not memory. EXP-004
reaches 95.40% with only 1,190.5 MiB on the 97,871 MiB H20. Three subsequent
children changed data history, augmentation allocation, or SAM geometry and
did not improve the accepted metric. None added per-example representational
capacity while preserving the full validated parent path.

ECA globally summarizes each residual feature map, then learns local
cross-channel interactions without squeeze-and-excitation's dimensionality
bottleneck. Every image can therefore amplify useful channels and suppress
irrelevant ones at all three spatial scales. The mechanism is active in every
ordinary forward and both SAM forwards, but does not consume a second model
pass or replace any training example.

`experiments/008/papers/eca-net.md` reports a 2.28-point ImageNet top-1 gain for
ECA-Net50 with only 80 parameters and negligible published FLOPs. That result
does not directly transfer to a shallow CIFAR WRN, but it establishes an effect
ceiling far above the roughly 0.3-point signal required by this noisy protocol.
The all-block placement gives the compact model six opportunities to adapt
channels, and identity initialization avoids spending the early critical
period recovering from six 0.5-scale residual gates.

A gain above 0.3 points is plausible because this changes class boundaries,
not only confidence or checkpoint selection: spatially averaged context
modulates the residual basis before every addition. CutMix can teach the gates
to retain channels useful across composite images; the final clean/SAM phase
then refines those gates toward a flatter clean predictor. This is a hypothesis,
not an extrapolation of the ImageNet number.

## Exact Module

Implement the kernel directly as a zero tensor parameter rather than
constructing `nn.Conv1d` and zeroing it later. `nn.Conv1d` runs a random default
initializer before zeroing and would advance the global RNG, changing all
subsequent parent initialization draws. A raw parameter consumes no RNG.

```python
class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be a positive odd integer")
        self.kernel = nn.Parameter(torch.zeros(1, 1, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x):
        descriptor = x.mean(dim=(2, 3)).unsqueeze(1)  # [N, 1, C]
        logits = F.conv1d(descriptor, self.kernel, padding=self.padding)
        gate = 2.0 * torch.sigmoid(logits)
        gate = gate.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * gate
```

The gate range is `(0, 2)`: channels can be attenuated or amplified, and one is
the neutral midpoint. Algebraically, `2*sigmoid(z)` is a scale-corrected form
of the paper's sigmoid rather than a new attention network. At zero,
`gate=1` and `d(gate)/dz=0.5`; with nonconstant channel descriptors and upstream
residual gradients, every kernel can move on the first optimizer step.

Keep the descriptor and convolution inside the existing BF16 autocast context.
Parameters remain FP32 and accumulate FP32 gradients under normal PyTorch
autocast semantics. Do not add an MLP, bias, reduction ratio, attention dropout,
temperature, learned global strength, or attention schedule.

## Exact Placement and Configuration

In each `PreActWideBlock`, construct one ECA module based on `out_channels`:

```python
kernel_size = 3 if out_channels == 64 else 5
self.eca = EfficientChannelAttention(kernel_size)
```

The six kernels are therefore `[3, 3, 5, 5, 5, 5]`, totaling 26 scalars. This
follows the ECA paper's principle of increasing odd channel-kernel size with
channel width while fixing the exact choice in advance.

Change the residual tail only:

```python
out = self.conv2(F.relu(self.bn2(out)))
out = self.eca(out)

drop_prob = self.drop_prob * drop_scale
# existing mask and inverse-survival scaling unchanged
return shortcut + out
```

Attention must not gate the shortcut or the post-addition tensor. Gating the
shortcut would destroy the pre-activation identity path; post-addition gating
would make the module responsible for both residual and identity channels and
would not start as the exact parent unless additional constraints were added.
Placing it before drop path treats ECA as part of the residual transformation,
so a dropped branch remains wholly dropped.

Do not modify the model stem, stage widths, number of blocks, final BN/pooling,
classifier, or existing initialization helper. Because `torch.zeros` consumes
no RNG and `_weights_init` ignores the raw ECA parameter, all shared parent
weights can initialize bit-identically under seed 42.

## Compute and Step Risk

For one image, the six gates reduce and multiply feature maps containing:

```text
2 * (64*32*32 + 128*16*16 + 256*8*8) = 229,376 elements
```

The descriptor convolutions add only
`2*(64*3 + 128*5 + 256*5) = 4,224` MACs. Counting reduction and broadcast
multiplication, added arithmetic is below roughly 0.12% of the parent's
approximately 392.6M convolution MACs. VRAM should rise by only a few MiB.

The real risk is 24 small GPU operations per forward (six each of reduction,
1D convolution, sigmoid, and multiplication), not FLOPs. EXP-004 ordinary
steps are about 10 ms and scheduled SAM steps about 20 ms; SAM repeats ECA on
its second forward. Kernel-launch latency or nonideal BF16 Conv1d dispatch
could therefore reduce the parent's 25,560-step horizon by several percent.

Require a separate GPU-0 microbenchmark before the metric run. Compare actual
parent and ECA models at batch 256, channels-last, BF16 autocast, with 50 warmup
iterations followed by at least 200 synchronized ordinary training iterations
and 100 production-faithful SAM iterations. Weight the latency ratio using the
parent's observed mix (about 90.4% ordinary, 9.6% SAM). Proceed only if:

- weighted latency is at most 1.06x parent;
- projected 300-second exposure is at least 24,100 steps;
- all ECA kernels use native finite BF16/FP32 operations with no error/OOM;
- projected evaluation time keeps total runtime below 600 seconds.

This gate selects feasibility, not a hyperparameter. If it fails, reject the
fixed all-block configuration rather than trying gate subsets or kernel sizes
against the test metric.

## Interaction with the Parent Recipe

### CutMix

Keep the parent's 0.5 gate, `Beta(1,1)` geometry, cutoff 0.75, area-corrected
loss, and dedicated seed-42 CPU/CUDA generators exactly. ECA uses no random
draws, so it does not alter CutMix decisions, permutations, data-loader order,
or crop/flip randomness. Mixed images influence global descriptors, but their
paired labels supervise the resulting gates through the unchanged objective.

### Drop path

Keep all six depth-dependent probabilities and the scale decay after progress
0.75. ECA runs before the existing per-example mask. Surviving branches receive
the same inverse-survival scaling as the parent; dropped examples contribute
no residual or ECA gradient, just as they contribute no Conv2d branch gradient.
At maximum 8% drop and batch 256, every kernel should still receive a gradient.
No extra random draw is introduced, so mask-stream structure is preserved.

### Period-two SAM

The raw ECA parameters are automatically included in `sam_parameters`,
snapshots, the global gradient norm, perturbation, exact restoration, and the
sole Nesterov update. Do not exclude them. They have no BatchNorm buffers or
randomness, so existing CUDA RNG replay and second-pass BN suppression require
no special case. ECA receives the same first- and perturbed-point objectives as
the backbone, making its channel policy part of the flatness-aware solution.

Keep `SAM_RHO=0.05`, start 0.75, period 2, CutMix/drop-path transitions, LR,
weight decay, batch size, and all timing boundaries unchanged. All ECA work is
inside the charged forward paths.

## Fixed Hypothesis and Falsification

The accepted parent is 95.40%; formal tree improvement begins at 95.50%. The
strong preregistered hypothesis is:

> Six identity-initialized ECA gates with kernels `[3,3,5,5,5,5]` will reach
> `best_test_acc` of 95.70-96.00% (+0.30 to +0.60 points), retain at least
> 24,100 optimizer steps, and add fewer than 10 MiB peak VRAM.

Use one seed-42 metric run only. `best_test_acc < 95.50%` is a tree
no-improvement. A result from 95.50% through 95.69% formally improves the tree
but falsifies the claimed detectable >=0.30-point mechanism effect. Do not
choose raw sigmoid, gate strength, placement subset, or alternate kernel sizes
after observing the test result.

Also classify the mechanism as functionally weak if final kernels remain
effectively zero (for example all `max_abs < 1e-5`), even if a small nominal
metric delta appears. That diagnostic does not override the metric verdict but
prevents attributing noise to learned attention.

## Failure Modes

- CIFAR-10 may not need image-conditioned channel recalibration; the WRN and
  global classifier can already learn static channel importance.
- Local adjacency in learned channel order may lack useful semantics in a
  shallow six-block model, unlike the deeper ImageNet backbone.
- Zero initialization is trainable but can learn slowly under weight decay and
  SAM. Identity preservation trades a random initial attention signal for a
  cleaner parent comparison.
- The `(0,2)` scale-corrected gate departs from the paper's `(0,1)` range and
  can amplify noisy channels. Sigmoid boundedness and SAM limit, but do not
  eliminate, this risk.
- CutMix descriptors summarize two source images and may teach diffuse gates;
  the final clean quarter may be too short to specialize them.
- Added tiny kernels can be launch-bound and reduce sample exposure enough to
  offset representational gains despite negligible MACs.
- The ECA parameters alter the SAM global norm slightly. Their count is tiny,
  but the full integration test must confirm exact restore and finite gradients.
- The protocol has 0.14-0.29-point selection/tail variability. One fixed run
  cannot establish a small effect, hence the separate 95.70 effect threshold.

## Tests and Verification

1. Unit-test ECA shape preservation for 64/128/256 channels; zero kernels must
   produce gates exactly one and output bit-identical to input in FP32.
2. Backpropagate a nonconstant loss through each kernel size and assert finite,
   nonzero kernel gradients at zero initialization.
3. Construct parent and ECA models from the same seed; verify global CPU/CUDA
   RNG states and every shared initialized tensor are identical, ECA kernels
   are zero, and `num_params == 2_748_916`.
4. Run a full-WRN BF16/channels-last smoke. Assert finite forward/loss/backward,
   non-None gradients for every parameter, preserved output layout, and no
   Conv1d/autocast fallback error.
5. On a scheduled SAM smoke, verify rho-0.05 perturbation, identical replayed
   drop masks, exactly one BN-buffer update, ECA perturbation, exact restoration
   of all parameters, and one optimizer/momentum update.
6. Pass the parent-vs-ECA GPU-0 latency gate above before the metric run.
7. Launch once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
   Confirm the 97,871 MiB H20, approximately 300 charged seconds, under 600
   total seconds, one evaluation per epoch, parent CutMix/SAM ratios and 0.75
   transition, at least 24,100 steps, complete summary, and metric thresholds.
8. Log final per-block kernel L2/max-absolute values outside charged training,
   without another data forward. Verify only `train.py` changed, all seeds and
   evaluator stayed fixed, no dependency was added, and transient logs are
   removed after analysis.


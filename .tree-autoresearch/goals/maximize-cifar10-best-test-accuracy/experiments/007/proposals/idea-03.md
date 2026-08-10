# Proposal: Identity-Centered Efficient Channel Recalibration

## Summary

Add a three-tap Efficient Channel Attention (ECA) gate to the residual branch of every EXP-004 PreAct WRN block, after the second convolution and before stochastic depth and shortcut addition. Keep the complete parent data stream, front-loaded CutMix, optimizer, schedules, and clean-tail period-two SAM unchanged.

Use one fixed design, not a scalar or placement search:

- six ECA gates, one per residual block;
- fixed channel-kernel size 3, padding 1, no bias;
- gate `2 * sigmoid(channel_logits)`;
- zero-initialized ECA kernels represented as standalone `nn.Parameter` tensors;
- no new normalization, activation, RNG, dependency, or auxiliary loss.

The identity-centered gate is a deliberate adaptation of standard ECA. Standard zero-logit sigmoid gates scale each residual branch by 0.5 at initialization, changing the already validated architecture before the attention weights learn. Multiplying the sigmoid by two makes every initial scale exactly 1 while retaining bounded positive per-channel recalibration in `(0, 2)` and a nonzero derivative at the origin.

## Why This Mechanism

The system is limited by detectable generalization gain, not memory. EXP-004 reaches 95.40% with only 1,190.5 MiB on the 97,871 MiB H20 and 25,560 steps. EXP-005 preserved steps but halved new-image introduction and regressed 0.12 points. EXP-006 preserved throughput while replacing one quarter of CutMix with manifold mixup and gained only 0.01 while worsening loss. The next candidate should therefore be additive, preserve independent-image and validated regularization exposure, and have a plausible effect above the observed 0.14-0.29-point selection variability.

ECA uses global average pooling followed by a short 1D convolution across adjacent channels, avoiding the reduction bottleneck and parameter cost of squeeze-and-excitation. The CVPR paper reports more than two ImageNet top-1 points over a plain ResNet-50 with only 80 parameters and `4.7e-4` GFLOPs. There is no matched result for this CIFAR-10 WRN/SAM regime, so the effect estimate is deliberately discounted.

The fixed hypothesis is a 0.30-0.50-point gain, yielding 95.70-95.90% from the 95.40% parent while retaining at least 24,000 steps. This is a mechanism-sized expectation rather than a promise; a gain below 0.10 still fails the formal 95.50% threshold.

Sources:

- `02-system-understanding.md`
- `experiments/004/04-analysis.md`
- `experiments/005/04-analysis.md`
- `experiments/006/04-analysis.md`
- `experiments/006/proposals/idea-03.md`
- `experiments/005/papers/eca-net.md`
- https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html

## Exact Module and Placement

Implement a small `IdentityECA` module in `train.py`:

```python
class IdentityECA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 3))

    def forward(self, residual):
        descriptor = F.adaptive_avg_pool2d(residual, 1)
        descriptor = descriptor.squeeze(-1).transpose(1, 2)  # N, 1, C
        logits = F.conv1d(descriptor, self.weight, padding=1)
        scale = 2.0 * torch.sigmoid(logits)
        scale = scale.transpose(1, 2).unsqueeze(-1)  # N, C, 1, 1
        return residual * scale
```

Each `PreActWideBlock` owns one gate. Its forward path becomes:

```text
preactivate -> conv1 -> preactivate -> conv2 -> IdentityECA
            -> existing drop path -> add existing shortcut
```

Gate the residual tensor before the existing per-example stochastic-depth mask. Do not gate the shortcut, post-addition state, stem, or final pooled representation. The six gate widths are 64, 64, 128, 128, 256, and 256; one shared three-tap kernel within each block operates locally along that block's channel axis.

Using a standalone zero tensor rather than constructing and then zeroing `nn.Conv1d` is important: it consumes no random initialization draws. The common parent Conv2d/Linear/BatchNorm parameters therefore initialize identically under seed 42. The existing model-wide initializer does not match a bare parameter, so the ECA tensors remain exactly zero.

At initialization, `F.conv1d(..., 0) = 0` and `2*sigmoid(0) = 1`, making every gated residual exactly equal to its parent value. The derivative of the gate with respect to its kernel is nonzero, so attention learns on the first backward. Because the zero kernel also blocks the descriptor-to-residual derivative through the gate at initialization, gradients of all pre-existing parameters initially equal the parent's gradients; only the new ECA parameters add a gradient path.

## Parameter and Arithmetic Cost

Each block adds exactly three trainable scalars, for 18 total parameters:

```text
parent parameters     2,748,890
ECA parameters               18
candidate parameters  2,748,908
```

For one image, the six residual outputs contain:

```text
2 * (64*32*32) + 2 * (128*16*16) + 2 * (256*8*8)
= 229,376 elements
```

ECA adds approximately:

- 229,376 pooling accumulations;
- 2,688 learned 1D-convolution MACs (`3 * sum(block_channels)`);
- 229,376 residual scale multiplications;
- sigmoid and small descriptor overhead.

Even counting pooling and scaling as scalar operations, about 461,000 operations/image is under 0.12% of the parent's approximately 392.6M convolution/classifier MACs. The practical cost can be higher because six small pool/conv1d/sigmoid/multiply sequences add kernel launches and memory traffic. Expect a 1-4% weighted training-latency increase and approximately 24,500-25,300 steps versus EXP-004's 25,560. Peak memory should remain close to 1.2 GiB; the gates add tiny descriptors, 18 parameters, gradients, momentum, and SAM snapshots.

## Parent Recipe Preservation

- Keep batch size 256, the independent shuffled DataLoader, seed 42, crop/flip transforms, BF16 autocast, and channels-last inputs/model.
- Keep `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, `CUTMIX_END=0.75`, and both dedicated seed-42 CutMix generators. ECA consumes no RNG, so the parent gate/lambda/center/permutation streams and global drop-path stream remain aligned over their shared step prefix.
- Keep `MAX_DROP_PATH=0.08`, its depth scaling, and final-quarter annealing. The gate precedes the mask but does not change its shape or draw count.
- Keep Nesterov SGD, weight decay `1e-4`, LR warmup/cosine schedule, and wall-clock progress boundaries unchanged. ECA weights receive ordinary gradients, momentum, and weight decay through the existing optimizer.
- Keep `SAM_RHO=0.05`, `SAM_START=0.75`, and `SAM_PERIOD=2`. All six ECA kernels join the first-pass global gradient norm, exact snapshots, perturbation, second backward, restoration, and sole optimizer update. Do not exempt attention weights from SAM.
- Preserve CUDA RNG replay, second-pass BatchNorm tracking suppression, and one BatchNorm update per batch. ECA has no stateful buffers or stochastic behavior.
- Keep evaluator calls, once-per-epoch cadence, best-metric accumulation, charged timer, and final summary semantics untouched.

All ECA forward/backward and SAM work occurs inside the existing `t0` through CUDA synchronization interval, so its cost is charged to the fixed 300 seconds.

## Implementation Scope and Audit Output

Modify only `train.py`:

1. Add `ECA_KERNEL_SIZE = 3`, `IdentityECA`, and one gate per `PreActWideBlock`.
2. Insert the gate at the single declared residual-branch location.
3. Extend the startup configuration with `channel_attention=identity_eca`, `eca_kernel=3`, and `eca_blocks=6`.
4. Before the unchanged final summary, print a mechanism-audit line with ECA parameter count, finite kernel L2 norm, maximum absolute kernel value, and number of nonzero kernel elements. Compute this after charged training, without another data/model forward.

Do not alter block widths/depth, parent initialization rules, forward signature, CutMix/SAM code, RNG seeds, transforms, optimizer, schedules, evaluator, or summary keys. The architecture name may remain `PreActWideResNet`; the config must identify the added ECA mechanism explicitly.

## Discriminating Smokes

1. **Initialization and RNG parity:** Reset seed 42 separately, instantiate parent and candidate, and require every common state tensor to be bitwise equal, all 18 ECA values to be zero, and the global CPU RNG state after construction to match. This catches hidden RNG consumption by module construction.
2. **Exact initial function:** Load identical common weights, run deterministic eval forwards, and require candidate logits to equal parent logits exactly. In training mode with matched CUDA RNG state, require identical logits, identical common BatchNorm updates, identical drop-path RNG end state, and identical gradients for every common parameter; require finite, nonzero ECA gradients.
3. **Gate math and locality:** On a source-coded descriptor and hand-set three-tap kernel, compare `F.conv1d` output and `2*sigmoid` scales with a manual padded channel-neighborhood calculation. Verify first/last-channel padding, output shape, scale range `(0,2)`, and exact identity for a zero kernel.
4. **Placement:** Use hooks or a toy block to prove gating occurs once after `conv2`, before the drop mask and residual addition, and never affects the shortcut directly. Require six gate invocations per model forward.
5. **Shape/layout/complexity:** Check all block output shapes, `num_params == 2_748_908`, exactly six three-element ECA parameters, finite BF16/channels-last forward/backward, and channels-last residual outputs after broadcast multiplication.
6. **Learning smoke:** Run one ordinary optimizer step and require at least one ECA kernel element to leave zero, while every common parameter gradient remains finite. Verify the ECA update receives existing momentum/weight decay exactly once.
7. **CutMix regression:** Re-run patch orientation, clipped-area lambda, target pairing, zero-area, and dedicated-generator tests. Candidate and parent helper RNG states/counters must agree over an identical shared prefix.
8. **SAM integration:** On a scheduled candidate step, require plain-SAM perturbation norm 0.05 across all parameters including ECA, replayed drop masks, one BatchNorm-buffer update, exact restoration of common and ECA parameters, and one optimizer/momentum update.
9. **Evaluator/default path:** Confirm `Eval.evaluate(model, device)` needs no signature change and attention is active identically in training/evaluation aside from the model's existing BatchNorm/drop-path modes.

## H20 Cost Gate

Before the metric run, benchmark parent and candidate in separate fixed-seed GPU-0 processes with batch 256, BF16, and channels-last. After warmup, measure at least 200 ordinary train steps and 100 production SAM steps, with synchronization, reporting median/p90 latency and peak allocation. Weight the ratio using EXP-004's approximately 90.4% ordinary and 9.6% SAM steps.

Proceed only if weighted latency is at most 1.06x parent, projected exposure is at least 24,000 steps, total runtime remains below 600 seconds, and all finite/layout/SAM invariants pass. Reject this fixed design if the gate fails; do not remove blocks or change kernel size based on the benchmark.

## Full Verification

Run exactly once after confirming physical GPU 0 is the 97,871 MiB NVIDIA H20:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 24,000 steps, `num_params=2,748,908`, all required summary keys exactly once, one evaluation per completed epoch, CutMix exposure near 0.5 before progress 0.75, no late CutMix, exact period-two SAM arithmetic after 0.75, first SAM progress near 0.75, finite/nonzero learned ECA audit values, and no NaN/Inf, traceback, CUDA error, or timeout. Success requires `best_test_acc >= 95.50%`; the preregistered expected range is 95.70-95.90%.

Do not rerun, select a different ECA placement/kernel, change initialization, or adjust any parent scalar after seeing test accuracy.

## Risks

- **Evidence transfer:** ECA's strongest evidence is ImageNet and deeper residual models, not a six-block CIFAR WRN already at 95.40%. The expected 0.30-0.50-point effect may not transfer.
- **Identity-centered departure:** `2*sigmoid` preserves the parent initially but differs from standard ECA's `(0,1)` gate and permits amplification. It may be less regularizing or amplify noisy channels.
- **Slow learning from exact identity:** The kernel starts at zero and must learn all channel preferences during the fixed budget. Its gradient is live, but the clean-tail SAM phase may arrive before a useful gate forms.
- **Kernel-size rigidity:** Fixed `k=3` is conservative for 64/128/256 channels but may under-model wider-stage dependencies. No scalar search is allowed in this experiment.
- **Kernel-launch overhead:** Arithmetic is negligible, yet six tiny operator chains can be launch- or memory-bound on CIFAR feature maps. The weighted H20 cost gate prevents a hidden throughput collapse.
- **SAM interaction:** New ECA gradients alter the global SAM perturbation direction and the representation entering late flatness optimization. This is intended composition, but there is no direct ECA+SAM evidence.
- **Run variability:** Prior selected results moved 0.14-0.29 points. A result below 95.50% is no improvement regardless of final loss or learned gate magnitude.

## Testable Hypothesis

Six identity-centered three-tap ECA gates will preserve EXP-004 exactly at initialization, learn useful local channel dependencies without changing sample or regularization exposure, retain at least 24,000 optimizer steps, and improve `best_test_acc` by 0.30-0.50 points to 95.70-95.90%. Any score below 95.50%, cost-gate failure, initial-parity failure, or CutMix/SAM/timing violation falsifies the proposal without a retry or scalar search.

# System Understanding: Maximize CIFAR-10 Best Test Accuracy
**Last verified**: 2026-08-06 @ `7c1e7d8` (baseline: 94.15%)

## Problem Decomposition

- **Strong data production** — cost: 0.145 ms median iterator wait, 0.171 ms p95 with eight persistent workers; bound by host transforms but hidden by prefetch; evidence: EXP-013 500-step real N1/M7+CutMix probe.
- **Host-to-device transfer** — cost: 0.067 ms median, 0.61% of measured GPU-stage time; bound by PCIe/DMA and pinned memory; evidence: EXP-013 CUDA-event probe.
- **Forward model** — cost: 2.408 ms median, 22.11% of GPU-stage time; bound by 19 convolution/BN/ReLU blocks; evidence: EXP-013 CUDA-event probe.
- **Cross-entropy** — cost: 0.016 ms median, 0.15%; not limiting; evidence: EXP-013 CUDA-event probe alternating hard/probability targets.
- **Backward model** — cost: 8.220 ms median, 75.46%; dominant fixed-time systems cost; bound by convolution/BN backward kernels and saved activations; evidence: EXP-013 CUDA-event probe.
- **Gradient reset plus SGD update** — cost: 0.182 ms median, 1.67%; too small for fused SGD alone to materially increase exposure; evidence: EXP-013 CUDA-event probe.
- **Launch/synchronization gap** — cost: 0.034 ms beyond 10.893 ms summed CUDA-event stages in a 10.927 ms wall step; host dispatch is not the main limiter; evidence: EXP-013 synchronized wall/CUDA-event probe.
- **Evaluation** — excluded from the 300-second training counter but included in the 600-second wall limit; dense weak-tail passes keep accepted total at 330.7 seconds; evidence: EXP-010 report.

## Current Bottleneck

The systems bottleneck is model backward (75.46% of counted step time), while loader wait, loss, transfer, optimizer, and visible host overhead are collectively small. The accuracy bottleneck is generalization under a short strong-view phase: width and conservative CutMix improved the frontier, whereas decay changes, stronger CutMix, and full preactivation either suppress strong fit or fail to clear the +0.10 gate. A performance candidate must therefore accelerate convolution/BN backward or process more useful examples without worsening large-batch generalization; a representation candidate must preserve EXP-010's healthy 89.73% switch fit.

## Headroom Assessment

- Accepted exposure is 26,898 updates / 3.44M images (68.9 dataset passes) in 300 seconds; faster steps or sublinear batch scaling can increase image exposure, but additional exposure has not yet been causally tested at the accepted recipe.
- Peak allocation is 598.7 MB, only about 0.61% of the 97,871 MiB H20 capacity. Memory capacity does not constrain larger batches, wider models, shadow averaged weights, or compiler workspaces.
- Pure Python-overhead reduction has little measured ceiling because the launch/sync gap is 0.034 ms. TorchInductor is unavailable: installed PyTorch 2.9.1 rejects `torch.compile` on Python 3.14 before capture, and dependency changes are forbidden.
- Fused SGD can remove at most roughly the measured 0.084 ms optimizer stage (under 0.8%) before secondary effects, below the accuracy gate's likely exposure needs.
- The probe did not measure achieved FLOPs, memory bandwidth, occupancy, or batch-size scaling, so no claim is made that H20 compute or bandwidth is saturated.

## Open Questions

- Does batch 256 or 512 scale step time sublinearly enough to increase images processed while preserving useful update noise at LR 0.1?
- Does late checkpoint/weight averaging improve the noisy weak-tail trajectory without BN-stat recalibration or meaningful counted overhead?
- Would zero-gamma initialization deepen the same strong-phase underfit seen in EXP-012, or improve postactivation generalization without changing compute?
- Which final spatial aggregation preserves localized CutMix features better than global average pooling at acceptable kernel cost?

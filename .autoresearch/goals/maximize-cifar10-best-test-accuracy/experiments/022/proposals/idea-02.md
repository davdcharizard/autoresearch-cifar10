# Proposal: FP32 Channels-Last for the Accepted Width-2 Model

## Claim and mechanism

Test a pure memory-layout optimization on the accepted EXP-010 recipe. Keep tensors logically NCHW and all arithmetic FP32, but store every convolution weight and 4D activation in `torch.channels_last` physical order. The system profile assigns 97.57% of measured GPU-stage time to model forward/backward, with backward alone at 75.46%; channels-last can therefore matter only by selecting faster cuDNN convolution/BN kernels and avoiding layout transposes. It does not change the learned function or add a generalization mechanism.

The registered hypothesis is deliberately two-stage:

1. Full-model FP32 channels-last reduces the synchronized 80/20 strong/weak weighted step time by at least 3.0%, raising fixed-budget exposure from 26,898 to at least 27,705 optimizer steps (at least 807 extra updates, about 103,000 images or 2.1 dataset passes).
2. The extra exploration and refinement raises `best_test_acc` from 94.15% to at least the formal 94.25% threshold; point prediction: **94.27%**.

Confidence in stage 1 is moderate-low and confidence in stage 2 is low. PyTorch and NVIDIA document channels-last propagation and NHWC-friendly cuDNN kernels, but their strongest performance evidence is coupled to reduced precision rather than this tiny FP32 CIFAR model. The goal diagnosis also says generalization, not exposure, is the accuracy limiter. EXP-021 does not strengthen the exposure claim; it only leaves this systems candidate distinct after pool-first shortcuts failed their safety gate.

## Exact production implementation

Make only these layout changes after ordinary seed-42 construction and before optimizer construction:

```python
class ResNet(nn.Module):
    # construction unchanged
    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        out = F.relu(self.bn1(self.conv1(x)))
        # every remaining operation is unchanged

model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(
    device=device, memory_format=torch.channels_last
)

# Inside the existing t0-counted region:
inputs = inputs.to(
    device, non_blocking=True, memory_format=torch.channels_last
)
targets = targets.to(device, non_blocking=True)
```

Do not use `permute`: logical shape remains `[N,C,H,W]`; only strides change. Keep the model-boundary conversion because `prepare.py` supplies contiguous CUDA evaluation tensors and cannot be edited. It must be a pointer-identical no-op for already channels-last training inputs. The combined training H2D/restride remains inside counted work, so the candidate pays its real conversion cost. Construct SGD after model conversion so parameter references and lazy momentum buffers inherit the final layouts.

Preserve the complete accepted recipe: width-2 postactivation ResNet-20, Option-A slice/pad shortcuts, global average pooling, 1,073,962 parameters, batch 128, seed 42, N1/M7 plus alpha-1 CutMix on 50% of strong batches through 80%, hard weak crop/flip tail, SGD momentum 0.9, all-parameter decay `1e-4`, and the existing LR/timer/worker logic. Do not change autocast, TF32/cuDNN flags, batch size, compilation, optimizer, or data transforms.

A speedup can add an epoch and therefore an extra max-selection opportunity. Add `evaluation_count = 0`; preserve all four elapsed checkpoints; permit ordinary dense-tail evaluation only while `evaluation_count < 18`; and always allow the single terminal evaluation. Increment after each evaluation. Preflight simulated 69- and 70-epoch schedules: each must produce exactly 19 unique looks, include the terminal epoch, evaluate at most once per epoch, and leave the accepted 69-epoch schedule unchanged. This cap is metric parity, not part of the accuracy mechanism.

## Semantic and numerical gates

Use disposable diagnostics, not production hooks.

- Reconstruct control and candidate from reset seed 42. Compared in logical contiguous order, require bitwise-identical initial parameters/buffers, identical parameter ordering/count, and identical CPU/CUDA RNG states after construction/conversion.
- Require all 19 Conv2d weights, their gradients, and initialized momentum buffers to be channels-last; BN/Linear values remain unchanged. For `[128,3,32,32]`, require candidate input stride `(3072,1,96,3)` and pointer identity across the forward-boundary call.
- Hook every Conv, BN, and block output and inspect both transition shortcut slice/pad paths. Every non-ambiguous 4D activation must remain channels-last. Profile a warmed full step and veto repeated `aten::contiguous`, `_to_copy`, or clone operations between convolutions; document the declared input conversion and unavoidable transition operation separately.
- Verify adaptive pooling followed by the unchanged `.view` yields finite `[128,10]` logits. Feed contiguous evaluator-style CUDA inputs through the boundary and verify correct conversion, stable BN buffers in evaluation mode, and finite outputs.
- On identical persisted hard and probability-target batches, require initial control/candidate FP32 logits and loss within `rtol=atol=1e-5`; for every nonzero parameter gradient require relative L2 error at most `1e-4` and cosine at least `0.99999`. Run four identical optimizer steps and require finite parameters/state, unchanged RNG consumption, and no candidate-only prediction concentration above 95%. Different valid cuDNN reductions may prevent bitwise equality after arithmetic, but larger divergence vetoes the experiment.

The main semantic risks are silent NCHW fallback after stride slicing, `F.pad`, or adaptive pooling; a copy hidden by a later Conv restoring NHWC; and ambiguous 1x1 pooled strides breaking `.view`. The main numerical risk is a different cuDNN algorithm/reduction order changing the seed-42 trajectory despite mathematically equivalent FP32 operations. Input conversion could erase all kernel savings. These are implementation effects of the tested layout and must not be rescued with per-layer permutations, autocast, or backend-flag changes.

## Paired timing design

On one idle 97,871-MiB H20, persist post-transform CPU batches once so both arms receive byte-identical strong N1/M7 hard/soft CutMix inputs and weak hard inputs. Run seven fresh-process pairs with order alternating AB/BA. Each arm restores identical logical model/optimizer state and unchanged backend flags, performs 100 untimed warmup steps, then measures at least 2,000 complete synchronized steps. Timing includes pinned H2D, candidate restride, zero-grad, forward, loss, backward, SGD, and final synchronization. Measure strong and weak paths separately and combine their means 80/20.

Record each arm's mean, median, p95, CV, peak allocation, strides, and profiler conversion counts. Production is authorized only if:

- weighted candidate/control mean is at most `0.9700`, every paired ratio is below `1.0`, and per-arm trial-mean CV is at most 2%;
- candidate weighted p95 is no worse than control and projected steps `floor(26_898 * control_mean / candidate_mean)` are at least 27,705;
- peak allocation is below 650 MiB and no more than 32 MiB above control;
- the layout, numerical, and hidden-copy gates all pass.

Separately time evaluator-like batch-256 forwards with contiguous CUDA input entering the model boundary (100 warmups, 500 measured forwards per trial). Candidate/control mean must be at most 1.10, CV at most 2%, and the conservative 19-evaluation wall projection below 540 seconds.

## Feasibility, production test, and falsification

Implementation is small and dependency-free, memory headroom is enormous (accepted peak 598.7 MiB on a 97,871-MiB H20), and the graph is mostly channels-last-supported Conv/BN/ReLU/add. Feasibility risk is performance, especially the RGB stem, tiny 32/16/8-pixel kernels, Option-A stride views, and FP32 rather than AMP.

If every gate passes, run seed 42 once with output redirected to `run.log`, no retry. Require 300 counted seconds, total below 600 seconds, exactly 19 unique evaluations including terminal, one 80% loader switch with eight workers stopped, expected target formats/CutMix share, and at least 27,705 steps.

The proposal is falsified early if the paired ratio exceeds 0.9700, any pair regresses, layout repairs appear, numerical gates fail, or wall projection fails; retire exact FP32 full-model channels-last without a production run. It is falsified mechanistically if production exposure is below 27,705 even if accuracy happens to rise. If exposure clears but `best_test_acc <94.25%`, record valid no-improvement: additional updates did not improve this fixed-seed trajectory. Only `best_test_acc >=94.25%` with all integrity and exposure gates supports acceptance, and even then the single-seed 0.10-point margin is weak causal evidence rather than proof that exposure generally improves accuracy.

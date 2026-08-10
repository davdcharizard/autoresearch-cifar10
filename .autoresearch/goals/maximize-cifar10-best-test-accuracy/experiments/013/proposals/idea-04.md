# Proposal: Isolated Zero-Gamma Initialization on Accepted Postactivation Blocks

## Decision

Define one exact candidate on the accepted EXP-010 postactivation width-2 ResNet-20: initialize the final BatchNorm scale in every residual branch to zero and change nothing else. The implementation is a deterministic post-construction overwrite after the existing Conv/Linear Kaiming pass:

```python
self.apply(self._weights_init)
for module in self.modules():
    if isinstance(module, BasicBlock):
        init.zeros_(module.bn2.weight)
```

All nine `BasicBlock.bn2.weight` tensors are zeroed: three 32-channel, three 64-channel, and three 128-channel vectors, or 672 existing scalar parameters. Their biases remain at PyTorch's unchanged default zero. Stem BN and all nine `bn1.weight` tensors remain one; every BN bias remains zero. Do not zero convolution weights, add a scalar multiplier, alter Option-A shortcuts, or combine this with preactivation.

The nominal one-run hypothesis is that making each residual contribution initially zero will ease early optimization and improve the accepted p=0.5 CutMix recipe from 94.15% to at least the 94.25% acceptance threshold, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's 26,898 updates. However, the accepted parameter-free Option-A transitions create a specific channelwise gradient deadlock under exact zero-gamma. The structural gradient gate below is therefore expected to retire this exact candidate before a GPU run. That result must not be rescued by epsilon gamma, projection shortcuts, nonzero beta, leaky ReLU, partial zeroing, or any other tuning.

## Intended Mechanism

For one accepted block, write the residual branch and output as:

```text
h = Conv2(ReLU(BN1(Conv1(x))))
r = gamma * normalize(h) + beta
y = ReLU(shortcut(x) + r)
```

With `gamma=0` and `beta=0`, the residual contribution is exactly zero. For ordinary blocks whose input came from the preceding post-add ReLU, `x` is nonnegative and the initialized output is exactly `x`. At the two transition blocks, the initialized output is the stride-2 Option-A slice followed by high-channel zero padding. More precisely, this is **zero-residual initialization**, not a mathematical identity on arbitrary signed input because the existing post-add ReLU is retained.

The literature motivation is Goyal et al.'s isolated ImageNet ResNet-50 result: zero-initializing the last postactivation residual BN scale modestly improved top-1. The intended mechanism is gradual branch recruitment around a skip-dominated initial network, not added regularization, capacity, or compute.

## First-Step Gradient Flow

This candidate differs critically from zeroing the internal BN in canonical full preactivation. In the accepted block, `bn2` is after the last convolution and its affine output feeds the residual addition directly. If `g` is the upstream derivative through the post-add ReLU, then generically:

```text
dL/dbeta  = sum(g)
dL/dgamma = sum(g * normalize(h))
dL/dh     is proportional to gamma and is zero on the first backward
```

Thus channels with a positive shortcut activation can give `bn2.weight` and `bn2.bias` nonzero first-step gradients even though `conv2`, `bn1`, and `conv1` receive zero data gradient on that backward. After SGD updates gamma, data gradients can enter the internal branch on the next step. Coupled `1e-4` decay still updates zero-data-gradient nonzero Conv/BN1 weights on step one because their gradient tensors exist; this is preserved accepted optimizer behavior, not branch learning from the loss.

Canonical preactivation has the problematic order `BN-ReLU-Conv-BN-ReLU-Conv-add`. Zeroing its second BN puts exactly zero into the following ReLU. PyTorch's derivative at ReLU zero is zero, so even that BN's gamma and beta receive no loss gradient; the whole branch is permanently dead. The accepted postactivation placement avoids that **whole-branch** deadlock.

It does not avoid a **channelwise** deadlock at Option-A transitions. When a transition changes 32 to 64 channels, the upper 32 shortcut channels are exact padded zeros. With residual gamma and beta also zero, their pre-ReLU values are exactly zero; PyTorch sends zero derivative backward there. Those upper `bn2` gamma/beta entries cannot move. The following same-stage blocks also receive exact zero shortcut activations in those channels, so their corresponding final BN affine entries remain dead. At the 64-to-128 transition, the same mechanism leaves the newly padded upper 64 channels dead, while the 32 channels already dead from the previous stage remain dead. Under exact arithmetic, the final stage can expose only the original lower 32 shortcut channels to pooling; 96 of 128 output channels never activate.

This is not a speculative accuracy risk. It follows from the current Option-A zero padding plus PyTorch's zero ReLU derivative. The ResNet-50 literature result used projection shortcuts, which do not impose this all-zero newly introduced channel boundary. Exact all-nine zero-gamma is therefore structurally incompatible with the local accepted shortcut design unless a correctness test disproves this derivation.

## Parameter, RNG, and Compute Identity

The candidate retains exactly **1,073,962 trainable parameters** and all existing buffers. It changes the initial values of 672 BN scale scalars but no tensor shape, module, parameter object, parameter group, operation, or state allocation.

Place the zeroing loop after the existing `self.apply(self._weights_init)` call. `init.zeros_` consumes no random numbers. When control and candidate construction each begin from seed 42:

- all Conv and Linear parameters must be bitwise identical;
- every BN bias, every `bn1.weight`, stem `bn1.weight`, running mean, running variance, and `num_batches_tracked` must be bitwise identical;
- only the nine `bn2.weight` tensors may differ, exactly one in control versus exactly zero in candidate;
- post-construction CPU and CUDA RNG states must be bitwise identical.

The forward graph still contains the same 19 convolutions, 19 BNs, 19 ReLUs, two Option-A slice/pad shortcuts, pooling, and classifier. Backward launches the same dense kernels even when numerical gradients are zero; PyTorch does not skip them based on gamma values. The optimizer traverses the same tensors. Parameter count, theoretical operations, persistent allocation, evaluation cost, and loader behavior are therefore identical. Any material timing difference is environmental or an implementation error, not a claimed mechanism.

## Preserved Accepted Recipe

Keep every EXP-010 behavior except the declared initialization values:

- width-2 postactivation ResNet-20, raw identity shortcuts within stages, and parameter-free Option-A transition shortcuts;
- seed 42, Conv/Linear Kaiming initialization, BN defaults except the nine final scales;
- N1/M7 RandAugment and alpha-1 CutMix probability 0.5 through the 80% high-LR plateau;
- hard-label weak crop/flip tail after deterministic shutdown of all eight strong-loader workers;
- batch 128, hard/probability-target cross entropy, ordinary SGD momentum 0.9, and coupled all-parameter decay `1e-4`;
- `lr=0.1` through 80%, then `0.01` cosine refinement to `1e-4`;
- current 300-second counted timer, synchronization, checkpoint/evaluation cadence, fixed evaluator, seed, logging, and summary.

Do not add warmup, modify LR, exempt zeroed BN scales from decay, alter beta, delay CutMix, lengthen the weak tail, change shortcut type, or selectively initialize only some blocks. Any such response to the feasibility or underfit concern is a new reviewed experiment.

## Structural and Gradient Gates

Run these gates in a disposable CPU process before any H20 timing or full run:

1. Require exactly nine `BasicBlock` instances, 1,073,962 parameters, and exactly 672 zero `bn2.weight` scalars partitioned as `3*32 + 3*64 + 3*128`. Require all nine `bn2.bias` tensors zero and unchanged, all ten other BN scale tensors one, and all BN buffers at defaults.
2. Construct paired control/candidate models from reset seed 42. Require bitwise equality of every state item except the named nine final scales and exact equality of post-construction CPU/CUDA RNG states.
3. Hook all block outputs on a nonnegative input. Require every ordinary block to equal its shortcut at initialization and each transition to equal the exact `[:, :, ::2, ::2]` slice plus high-channel zero pad after the existing ReLU.
4. On one fixed ordinary hard-label batch, run a first backward before optimizer update. Require finite nonzero tensor-level gradient norms for each block's `bn2.weight` and `bn2.bias`, and require zero data gradients into each branch's `conv1`, `bn1`, and `conv2`, consistent with gamma zero.
5. Inspect gradients per output channel, not only by tensor norm. At both Option-A transitions and all downstream blocks, require every newly introduced padded-channel `bn2.weight` and bias entry to receive a finite nonzero data gradient on step one and after eight ordinary SGD steps.
6. Hook post-add activations after eight steps and require nonzero activation variance in all 64 stage-2 and all 128 stage-3 channels. Require every final classifier input channel to receive a nonzero data-dependent activation and gradient.
7. Require finite loss and logits, unchanged parameter identities and optimizer membership, and no unplanned state or RNG changes.

Gate 5 or 6 is expected to fail on the current implementation: the padded channels are analytically dead. This is a mandatory no-go, not an invitation to weaken the check. Record the exact dead-channel masks and retire the candidate from EXP-013. Only if all gates unexpectedly pass under the exact accepted PyTorch operations may paired H20 timing proceed.

## Conditional H20 Timing Gate

If and only if every CPU structural/gradient gate passes, use one idle 97,871 MiB H20 for five alternating fresh-process control/candidate pairs. Each process must reset seed and build the accepted soft-target training path, warm up identically, then time at least 500 synchronized forward/backward/SGD steps.

Require:

- candidate/control median step ratio `<=1.01`;
- both timing distributions finite with CV `<=2%`;
- projected exposure `floor(26,898 * control_time / candidate_time) >= 26,629`, or 99% of EXP-010;
- peak allocated memory `<=610 MB`; and
- conservative projected end-to-end runtime below 540 seconds.

Initialization occurs before `t_start_training` in both control and candidate and requires no candidate-specific timed work. Failure of any timing gate is no-go; do not compensate through batch size, fused SGD, compilation, or timer changes.

## Strong-Phase Underfit Risk

Even without the Option-A dead channels, zero-gamma is not automatically beneficial in this short, heavily regularized regime. The accepted N1/M7 plus p=0.5 CutMix plateau already trades fit for invariance. EXP-010's 89.73% switch checkpoint was healthy, whereas EXP-011's stronger CutMix and EXP-012's full preactivation both lowered it by roughly 2.9 points and missed the frontier. Zero-gamma suppresses all residual-path data gradients on step one and keeps them scaled by small learned gamma early in training, potentially delaying useful capacity recruitment under a fixed 240-second high-LR phase.

The delay is only one exact step for channels with active shortcuts, but gradual scale growth can matter beyond that step. For padded channels it is permanent, making the effective terminal representation drastically narrower. The expected failure signature is therefore a strong checkpoint below the predeclared 87.08 underfit marker, low or zero activity in padded stage channels, and inadequate weak-tail recovery. These are diagnostics only: no early termination, adaptive initialization, or rescue schedule is allowed if a full run is reached.

## Conditional One-Run Hypothesis and Decision Rule

If all feasibility and timing gates pass despite the predicted Option-A issue, run exactly once at seed 42 with output redirected to `run.log`.

**Hypothesis:** isolated zero-gamma on the accepted postactivation blocks will preserve at least 99% of EXP-010 update exposure and improve basin conditioning enough to reach **94.30% point-estimate best test accuracy**, satisfying the formal **94.25%** threshold without pushing the 80% checkpoint below 87.08%.

Require exit zero, ten finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, at least 26,629 steps, peak VRAM no greater than 610 MB, one 80% augmentation switch, eight stopped workers, about 50% strong-phase CutMix, hard targets in the weak tail, and unique evaluation epochs. Compare switch accuracy, first weak accuracy, final NLL, endpoint slope, best/final gap, exposure, and channel-activity diagnostics with EXP-010. Only `best_test_acc >=94.25%` with every integrity gate is an improvement.

A valid result below 94.25% is no-improvement and cannot be rerun. A checkpoint below 87.08 is underfit evidence only and cannot trigger rescue tuning. Crash or protocol faults may be fixed only while preserving exact all-nine zero gamma. Failure of the structural padded-channel gate is a feasibility rejection, not an experiment crash and not permission to launch knowingly capacity-disabled training.

## Failure Mechanisms

- **Option-A padded-channel deadlock:** exact zero shortcut, zero residual affine output, and zero ReLU derivative permanently prevent new stage channels from activating.
- **Delayed branch recruitment:** active-channel residual internals receive no first-step data gradient and initially learn behind small gamma scales.
- **Strong composite underfit:** identity-biased initialization can compound N1/M7 plus CutMix regularization during the short high-LR horizon.
- **Depth mismatch:** the published gain was on deeper ResNet-50/ImageNet; a nine-block CIFAR network may not need optimization help.
- **Postactivation limitation:** ordinary blocks are identity only for nonnegative inputs, and transition blocks remain slice/pad mappings rather than information-preserving projections.
- **Coupled decay interaction:** zero gamma has no initial decay term, while branch weights still decay during their zero-data-gradient first step, slightly changing early balance.
- **Single-seed resolution:** a bare 0.10-point pass is only ten CIFAR-10 examples and must be reported as protocol-valid but weak causal evidence, never confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py` scope, seed-42 one-H20 protocol, primary metric, and 94.25% moving gate.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: backward-dominated fixed-time cost, accepted exposure, and zero-gamma as an open representation question.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: accepted postactivation/CutMix recipe and repeated strong-phase underfit warning.
- `goals/maximize-cifar10-best-test-accuracy/experiments/012/proposals/idea-03.md`: primary zero-gamma evidence and proof of canonical preactivation deadlock.
- `goals/maximize-cifar10-best-test-accuracy/experiments/012/04-analysis.md`: compute-neutral 94.22% preactivation near miss and 2.85-point switch underfit.
- `goals/maximize-cifar10-best-test-accuracy/experiments/013/01-brainstorm.md`: exact candidate seed and requirement to preserve accepted postactivation semantics.
- `train.py`: current final-BN placement, post-add ReLU, Option-A zero padding, parameter count, optimizer, and fixed training/evaluation protocol.
- P. Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, Section 5.1 and Table 2b: postactivation final-BN zero initialization on projection-shortcut ResNet-50.

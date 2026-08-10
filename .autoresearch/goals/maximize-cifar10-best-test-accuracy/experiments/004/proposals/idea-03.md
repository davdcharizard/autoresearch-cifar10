# Idea: Same-Width Full-Preactivation ResNet-20

## Summary

Replace the current post-activation CIFAR ResNet-20 with a full-preactivation ResNet-20 while preserving the accepted EXP-002 optimizer schedule and every other training control. Keep the same 3x3 convolutions, stage widths `16/32/64`, three two-convolution blocks per stage, Option-A strided/zero-padded shortcuts, classifier, augmentation, hard-label loss, batch size, momentum, weight decay, seed, and evaluation cadence.

The intervention moves each block's BatchNorm and ReLU before its convolution and removes the ReLU after residual addition. A final BatchNorm/ReLU is placed before global pooling. This gives the within-stage shortcut path no activation or learned transformation, allowing both activations and gradients to pass through additions directly. Because the existing shortcut and convolution shapes are retained, the proposed model has exactly the same expected trainable parameter count as EXP-002: 269,722.

## Diagnosis

The experimental record has isolated two facts:

- EXP-002 established a productive fixed-budget optimization horizon: hold `lr=0.1` for 80% of counted training, step to `0.01`, then cosine-decay to `1e-4`. It reached 91.83% best accuracy in 38,629 steps and 300.0 counted seconds.
- EXP-003 preserved that schedule and added `label_smoothing=0.05`. Test loss improved from 0.2843 to 0.2740, but top-1 remained 91.83% and the slower loss path reduced steps by 6.7%. This operating point should not be composed into the next experiment.

The accepted model is therefore still a 2016-style post-activation ResNet-20. Every residual sum is immediately passed through ReLU, so negative components of both the shortcut-carried representation and residual correction are truncated before the next unit. Architecture and representation flow have not yet been tested independently. Full preactivation directly targets this behavior without increasing width, depth, data cost, or parameter count.

This is a measured-risk proposal rather than a guaranteed transfer. The strongest published results are on ResNet-110 and deeper networks, where direct gradient paths matter more. The current network has only nine residual blocks and runs for about 100 epochs, so its optimization problem may already be easy enough that preactivation yields little benefit. That shallow-depth caveat is the main reason to keep the experiment otherwise exact.

## Mechanism

The current block computes approximately:

```text
y = ReLU(BN(conv1(x)))
r = BN(conv2(y))
output = ReLU(shortcut(x) + r)
```

The proposed block computes:

```text
h = ReLU(BN(x))
r = conv1(h)
r = conv2(ReLU(BN(r)))
output = shortcut(x) + r
```

For same-shape units, `shortcut(x) = x`. There is no operation after addition, so a sequence of residual units has an additive identity path. This supports direct forward information flow and a gradient term that bypasses the residual branch. BatchNorm still regularizes every convolutional input, while the final BatchNorm/ReLU prepares the last additive representation for pooling and classification.

He et al.'s ECCV 2016 paper reports that full preactivation improved CIFAR-10 ResNet-110 error from 6.61% to 6.37% and attributes the effect to easier optimization plus BN-related regularization. The same paper explicitly notes that activation-induced optimization difficulty is less severe for shallower networks, so those gains are evidence for the mechanism, not a quantitative promise for ResNet-20. Source: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).

## Exact Architecture and Shortcut Design

### Preactivation Basic Block

Each block owns:

```python
self.bn1 = nn.BatchNorm2d(in_channels)
self.conv1 = nn.Conv2d(
    in_channels,
    out_channels,
    3,
    stride=stride,
    padding=1,
    bias=False,
)
self.bn2 = nn.BatchNorm2d(out_channels)
self.conv2 = nn.Conv2d(
    out_channels,
    out_channels,
    3,
    stride=1,
    padding=1,
    bias=False,
)
```

Its forward pass is:

```python
def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))

    shortcut = x
    if self.need_pad:
        shortcut = shortcut[:, :, :: self.stride, :: self.stride]
        shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))

    return out + shortcut
```

Important details:

- Do not apply ReLU or BatchNorm after the addition.
- Compute the shortcut from raw `x`, not the preactivated tensor. This preserves the cleanest available information path.
- Retain the current Option-A transition shortcut: spatial striding by slicing and zero-padding the added high channels. Do not add a learned 1x1 projection. This keeps both parameter count and shortcut policy fixed; only two of nine blocks change shape.
- Retain channel padding exactly as the baseline implements it rather than introducing symmetric padding as an unrelated variation.

### Network Stem and Head

Use the same bias-free stem convolution but remove its immediate BatchNorm/ReLU:

```python
out = self.conv1(x)
out = self.layer1(out)
out = self.layer2(out)
out = self.layer3(out)
out = F.relu(self.bn_final(out))
out = F.adaptive_avg_pool2d(out, 1)
return self.fc(out.view(out.size(0), -1))
```

Add `self.bn_final = nn.BatchNorm2d(64)`. Keep `conv1: 3 -> 16`, the three stages `16 -> 16`, `16 -> 32`, `32 -> 64`, block counts `(3, 3, 3)`, transition strides `(1, 2, 2)`, and `fc: 64 -> 10` unchanged. Keep the existing Kaiming-normal initialization for convolutional and linear weights and the default BatchNorm affine initialization. Do not add zero-gamma initialization, stochastic depth, dropout, projection shortcuts, or any other architectural refinement in this run.

## Parameter Count Expectation

The convolution and classifier tensors are unchanged. Their total remains 268,346 trainable parameters. BatchNorm affine parameters are redistributed rather than added:

| Component | Trainable BN parameters |
|---|---:|
| Stage 1 preactivation BNs | 192 |
| Stage 2 preactivation BNs | 352 |
| Stage 3 preactivation BNs | 704 |
| Final `BatchNorm2d(64)` | 128 |
| Total BN affine parameters | 1,376 |

The total is therefore `268,346 + 1,376 = 269,722`, exactly matching EXP-002. BatchNorm running statistics are buffers and are not included in `num_params`. A static instantiation check should assert `sum(p.numel() for p in model.parameters()) == 269_722` before the full run.

A projection-shortcut implementation would add `16*32 + 32*64 = 2,560` weights and would change both shortcut learning and parameter count. It is deliberately excluded to make this a clean activation-order experiment.

## Preserved EXP-002 Training Policy

Keep the current accepted hard-label configuration byte-for-byte outside the model classes:

- hard-label `F.cross_entropy(outputs, targets)`; no label smoothing;
- batch size 128 and the same shuffled, drop-last, persistent-worker loader;
- random crop, random horizontal flip, tensor conversion, and existing normalization;
- SGD with `lr=0.1`, ordinary momentum `0.9`, weight decay `1e-4`, and no Nesterov;
- `LR_HOLD_FRACTION=0.8`, then the existing step to `ANNEAL_START_LR=0.01` and elapsed-time cosine to `MIN_LR=1e-4`;
- early evaluation checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense once-per-epoch tail evaluation, and unconditional terminal evaluation;
- seed 42, `MAX_STEPS=64000`, fixed evaluator, and all summary/reporting behavior.

## Hypothesis and Expected Impact

**Primary hypothesis:** same-width full preactivation will raise `best_test_acc` from the moving baseline of 91.83% to at least 91.93%, with a plausible outcome around 91.95-92.15% (+0.12 to +0.32 percentage points), because the clean within-stage identity path preserves signed representations and improves optimization/regularization without reducing model exposure.

**Secondary hypotheses:**

- Parameter count will remain exactly 269,722 and peak VRAM will remain close to EXP-002's 330.1 MB.
- Counted throughput will remain within roughly 3% of EXP-002's 38,629 steps, because the number and shapes of all convolutions are unchanged and the model still executes 19 BatchNorm and 19 ReLU operations per forward pass.
- If the mechanism helps, checkpoint accuracy should be similar or slightly better during the high-LR plateau and separate most clearly during final refinement; it should not require a different learning-rate schedule to train.

The expected gain is conservative relative to deeper published models. A result below threshold is entirely plausible because ResNet-20 is shallow and the paper's strongest optimization argument scales with the number of residual units.

## Throughput and Fixed-Budget Risk

Convolution FLOPs, convolution count, feature widths, residual-block count, batch size, loss kernel, and parameter count are unchanged. BN/ReLU modules are moved to different tensor shapes, and the stem BN/ReLU is exchanged for a final BN/ReLU. Aggregate elementwise work is similar and no extra synchronized GPU operation is introduced, so a large throughput regression is not expected.

Nevertheless, EXP-003 demonstrated that even a simple implementation change can cost 6.7% of fixed-budget steps. Preactivation changes kernel ordering and may affect CUDA launch/fusion behavior for this very small, kernel-bound model. Treat throughput as a first-class measured outcome:

- Expected band: approximately 37,470-39,790 steps (within +/-3% of 38,629).
- A drop greater than 5% is a material mechanism confound even if the run is valid; distinguish representation quality from reduced optimization exposure in analysis.
- The architecture adds no test passes, loader work, model copies, or persistent large tensors, so total runtime should remain near EXP-002's 336.0 seconds and comfortably under 600 seconds.

## Failure Modes

- **Shallow-network non-transfer:** direct gradient propagation may not constrain a nine-block ResNet enough to affect top-1. The paper itself says post-addition truncation is less severe at lower depth.
- **Finite-horizon mismatch:** published preactivation benefits were established on deeper models and longer conventional training. About 100 epochs in 300 seconds may not reproduce the same representation effect.
- **Option-A transition mismatch:** zero-padded shortcuts retain isolation and parameter parity but differ from the learned projections used in some preactivation implementations. If within-stage preactivation looks promising but transition capacity limits accuracy, projection shortcuts should be a separate follow-up, not silently bundled here.
- **Early optimization shift:** moving normalization before convolutions changes activation distributions. The accepted `lr=0.1` policy may not be optimal for the new block even though it is intentionally fixed for causal attribution.
- **Throughput loss:** unfavorable kernel scheduling could reduce optimizer steps under the wall budget. Report step delta explicitly rather than attributing all accuracy change to representation.
- **BatchNorm implementation error:** using `BatchNorm2d(out_channels)` for `bn1` in a transition block, reusing the preactivated tensor for the shortcut, omitting final BN/ReLU, or retaining post-add ReLU would implement a different architecture.
- **Metric variance near threshold:** the required improvement is 0.10 points. Use the single fixed seed and predeclared run; do not reroll or repeat solely to obtain a favorable result.

## Why This Is Isolated and De-bundled

EXP-002 remains the accepted optimizer baseline. EXP-003 showed that adding even one loss feature can alter both generalization and throughput, so architecture should now be tested without simultaneous regularization or schedule tuning.

This proposal changes only the ordering of existing normalization/activation operations plus their required final placement. It does not widen or deepen the model, learn the shortcuts, change augmentation or loss, tune the LR hold fraction, add Nesterov, alter weight decay, add EMA/SWA, use mixed precision/compilation, or change validation. Exact parameter parity and the retained zero-pad shortcut make the result substantially easier to interpret than a wider ResNet or a projection-based preactivation variant.

## Implementation and Static Verification

1. Rename `BasicBlock` to `PreActBasicBlock` or replace its internals exactly as specified.
2. Change `bn1` to normalize `in_channels`; retain both convolution shapes.
3. Return the residual sum directly with no post-add ReLU.
4. Remove the stem `BatchNorm2d(16)` and stem ReLU from the network forward path.
5. Add `BatchNorm2d(64)` plus ReLU after `layer3` and before adaptive average pooling.
6. Leave the entire training/evaluation loop unchanged.
7. Before GPU execution, verify:

   ```python
   model = ResNet(NUM_BLOCKS, NUM_CLASSES)
   assert sum(p.numel() for p in model.parameters()) == 269_722
   assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
   ```

8. Run Python compilation, Ruff/pre-commit, and a diff/scope check. Confirm `prepare.py`, dependencies, seed, optimizer, scheduler, transforms, loss, and evaluation code are untouched.

## Full-Run Verification

Before launch:

- Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is visible.
- Confirm the moving baseline is 91.83%, making 91.93% the minimum improvement threshold.
- Confirm no stale `run.log` or renamed log remains.
- Confirm the model reports 269,722 parameters and the static shape test passes.

Execute exactly once with redirected output:

```bash
uv run train.py > run.log 2>&1
```

Terminate and mark failure if wall time exceeds 10 minutes. After completion:

- Require a clean exit and one complete, finite summary.
- Require counted training near 300 seconds and total runtime below 600 seconds.
- Require `best_test_acc >= 91.93%` for improvement over EXP-002.
- Confirm validation occurred at most once per epoch and terminal evaluation matches the summary epoch.
- Compare `num_steps` against 38,629, `peak_vram_mb` against 330.1 MB, and best/final accuracy and test loss against EXP-002.
- Inspect early checkpoint accuracy for optimization lag and late-tail accuracy for the hypothesized representation/generalization gain.
- Verify `num_params == 269722`; any other value means the intended isolated architecture was not run.
- Do not retry with another seed. Record the fixed-seed result and remove `run.log` before any next experiment.

## Decision Rule

- **Improvement:** accept only if `best_test_acc >= 91.93%` and all protocol/integrity checks pass.
- **No improvement, throughput preserved:** reject same-width preactivation at this shallow fixed-horizon operating point; do not infer that deeper preactivation is ineffective.
- **No improvement with material throughput loss:** reject this implementation under the wall-clock objective, while recording that representation quality was confounded by fewer updates.
- **Promising but transition-limited behavior:** any follow-up with learned 1x1 transition shortcuts must be predeclared as a separate parameter-increasing architecture experiment.
- **Invalid/crash:** correct only the implementation or protocol fault and rerun the same predeclared design; do not tune architecture or seed opportunistically.

## Evidence

- `TASK.md`: only `train.py` is mutable; training is limited to one H20 and a fixed 300-second counted budget, with a 600-second total timeout.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: goal metric, moving-baseline improvement rule, and integrity constraints.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`: preserve persistent workers, bounded evaluation, and the long high-LR plateau; treat throughput as an explicit cost after EXP-003.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`: current accepted baseline is EXP-002 at 91.83%; label smoothing did not move top-1.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/002/04-analysis.md`: validated schedule, 38,629-step throughput, runtime, VRAM, and parameter reference.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/003/04-analysis.md`: built-in smoothing's unchanged top-1 and 6.7% step penalty motivate an architecture-only next test.
- [He et al., "Identity Mappings in Deep Residual Networks," ECCV 2016](https://arxiv.org/abs/1603.05027): full-preactivation mechanism, CIFAR evidence, and shallow-depth caveat.

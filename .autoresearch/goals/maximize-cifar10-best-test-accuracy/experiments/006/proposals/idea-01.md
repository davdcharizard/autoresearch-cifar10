# Idea: Full-Preactivation ResNet-20 on the Accepted Plateau-Only RandAugment Recipe

## Summary

Compose the accepted EXP-004 data/optimization recipe with one orthogonal architecture change: replace the current post-activation ResNet-20 block ordering with same-width full preactivation. Preserve `RandAugment(num_ops=1, magnitude=7)` through exactly 80% of counted training, the deterministic worker shutdown and crop/flip loader switch at the LR boundary, the 80%-hold optimizer schedule, hard-label loss, ordinary momentum, seed, evaluation cadence, and every loader control.

The proposed network keeps all 18 residual convolutions, `16/32/64` stage widths, three blocks per stage, Option-A strided/zero-padded shortcuts, and the classifier unchanged. BatchNorm and ReLU move before each convolution, the post-addition ReLU is removed, and a final BatchNorm/ReLU is added before pooling. The expected trainable parameter count remains exactly 269,722, matching EXP-004.

Relative to the moving baseline, this is an isolated composition experiment: EXP-004's accepted model/data recipe is reproduced, and only representation flow through the residual network changes.

## Experimental Diagnosis

The local history supports three constraints on the next experiment:

- EXP-002 established the optimizer base: `lr=0.1` for 80% of counted time, then a step to `0.01` and cosine refinement to `1e-4`.
- EXP-004 added strong worker-side augmentation only during that high-LR plateau and improved best accuracy from 91.83% to 92.30%. It preserved 38,358 steps, 99 epochs, 300.0 counted seconds, 340.7 total seconds, 330.1 MB VRAM, and 269,722 parameters.
- EXP-005 moved the strong-to-weak switch to 75% while keeping LR high until 80%. Throughput stayed intact but best accuracy fell to 92.12%, so the exact EXP-004 alignment of augmentation and LR boundaries should be treated as accepted, not retuned during architecture testing.

Label smoothing has also been rejected at its tested operating point: it improved test NLL but not top-1 and cost 6.7% of optimizer steps. The remaining untested lever is representation/optimization inside the model. The post-activation baseline truncates each residual sum with ReLU; full preactivation instead keeps within-stage shortcut paths free of learned transforms and post-addition nonlinearities.

Preactivation may be especially relevant under the accepted strong-view plateau. RandAugment deliberately makes the fitting problem harder, while clean identity paths can ease signal and gradient propagation through the nine residual units. The weak final 20% already present in EXP-004 also gives all BatchNorm buffers, including the proposed final BatchNorm, time to adapt back to the crop/flip distribution before the best checkpoint is selected.

This is not guaranteed to transfer. Published preactivation gains are strongest for ResNet-110 and deeper models; ResNet-20 is shallow enough that post-add ReLU may not be an important optimizer bottleneck. Preactivation's BN regularization can also combine with RandAugment into excessive regularization. Those risks motivate strict preservation of every non-architecture setting.

## Mechanism

The accepted post-activation block computes:

```text
y = ReLU(BN(conv1(x)))
r = BN(conv2(y))
output = ReLU(shortcut(x) + r)
```

The proposed full-preactivation block computes:

```text
h = ReLU(BN(x))
r = conv1(h)
r = conv2(ReLU(BN(r)))
output = shortcut(x) + r
```

For same-shape blocks, `shortcut(x) = x`. With no activation after addition, forward features and an additive gradient term can pass from one block to another without traversing BN, ReLU, or a convolution. BN still normalizes every residual-branch convolutional input, and a final BN/ReLU prepares the accumulated representation for global pooling.

He et al. report that full preactivation improves optimization and generalization in deep CIFAR residual networks; on ResNet-110, error improved from 6.61% to 6.37%. They also state that post-addition truncation is less severe at shallower depth, which limits the strength of inference for this nine-block model. Source: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).

## Exact Architecture

### Block Definition

Replace `BasicBlock` with the following topology:

```python
class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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

        self.stride = stride
        self.need_pad = stride != 1 or in_channels != out_channels
        self.pad_channels = out_channels - in_channels if self.need_pad else 0

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        shortcut = x
        if self.need_pad:
            shortcut = shortcut[:, :, :: self.stride, :: self.stride]
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))

        return out + shortcut
```

Required details:

- `bn1` normalizes `in_channels`, particularly in transition blocks.
- The shortcut uses raw `x`, not the preactivated tensor.
- There is no BatchNorm or ReLU after addition.
- Dimension changes retain the current parameter-free Option-A shortcut: stride by spatial slicing, then append zero channels with the exact existing `F.pad` call.
- Do not introduce 1x1 projections. They would add 2,560 weights and confound activation order with shortcut capacity.

### Stem, Stages, and Head

Retain the bias-free `3 -> 16` stem convolution, but remove its immediate BatchNorm/ReLU. Build the same stages with `PreActBasicBlock`:

- stage 1: three `16 -> 16` blocks, first stride 1;
- stage 2: one `16 -> 32` stride-2 block plus two `32 -> 32` blocks;
- stage 3: one `32 -> 64` stride-2 block plus two `64 -> 64` blocks.

Add `self.bn_final = nn.BatchNorm2d(64)` and use:

```python
def forward(self, x):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.relu(self.bn_final(out))
    out = F.adaptive_avg_pool2d(out, 1)
    return self.fc(out.view(out.size(0), -1))
```

Retain the existing Kaiming-normal initialization for convolutional and linear weights and default BatchNorm affine initialization. Do not add zero-gamma residual initialization, projections, dropout, stochastic depth, width, depth, or any other architecture feature.

## Parameter Count

Convolution and classifier parameters remain 268,346. BatchNorm affine parameters are redistributed:

| Component | Trainable parameters |
|---|---:|
| Stage 1 block BNs | 192 |
| Stage 2 block BNs | 352 |
| Stage 3 block BNs | 704 |
| Final `BatchNorm2d(64)` | 128 |
| Total BN affine | 1,376 |

Expected total: `268,346 + 1,376 = 269,722`, exactly equal to EXP-004. Running means, running variances, and batch counters are buffers and do not contribute to `num_params`.

Static verification must assert:

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES)
assert sum(p.numel() for p in model.parameters()) == 269_722
assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
```

## Data, Optimizer, and Worker Policy to Preserve

Outside the model classes, retain the current EXP-004 code exactly:

- `weak_train_tf`: random crop, horizontal flip, tensor conversion, existing normalization;
- `strong_train_tf`: the same operations plus `RandAugment(num_ops=1, magnitude=7)` before tensor conversion;
- begin with the strong loader and keep it through `LR_HOLD_FRACTION == 0.8`;
- keep `make_train_loader`, eight configured workers, pinned memory, shuffle, drop-last, persistent workers, and `multiprocessing_context="forkserver"` unchanged;
- after the crossing strong batch, break at the existing predicate, perform the scheduled evaluation, call `shutdown_train_loader`, verify old workers stopped, collect, and build the weak loader exactly once;
- keep `randaugment_enabled` state and the one-time switch log unchanged;
- hard-label `F.cross_entropy`; no label smoothing or Mixup;
- batch size 128;
- SGD with `lr=0.1`, ordinary momentum 0.9, weight decay `1e-4`, and no Nesterov;
- hold `lr=0.1` through 80%, then step to `0.01` and cosine-decay to `1e-4` over the final 20%;
- evaluation checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense once-per-epoch tail evaluation, and terminal evaluation;
- seed 42, maximum-step guard, timing boundaries, evaluator, and summary output.

Do not add an independent augmentation-switch constant or carry over EXP-005's 75% change. Strong augmentation and the LR transition must remain aligned at 80%.

## Hypothesis and Expected Impact

**Primary hypothesis:** composing same-width full preactivation with the accepted plateau-only RandAugment recipe will increase `best_test_acc` from 92.30% to at least 92.40%, with a plausible outcome around 92.42-92.62% (+0.12 to +0.32 points). Clean identity paths should make the strongly augmented high-LR problem easier to optimize while preactivation BN regularization improves representation quality; the existing weak low-LR tail should then adapt BN buffers and refine those representations for clean test inputs.

**Secondary expectations:**

- `num_params` remains exactly 269,722 and peak VRAM remains close to 330.1 MB.
- The augmentation switch occurs once at about 80.0%, with the crossing batch still using `lr=0.1`, all strong workers stopped, and the next phase using the weak loader and low-LR schedule.
- Optimizer exposure remains within approximately 3% of EXP-004's 38,358 steps.
- If preactivation helps the composition, the largest interpretable separation should appear after the weak switch; strong-phase evaluator accuracy is suppressed by the expected strong/clean distribution mismatch and should not be used alone to reject representation quality.

The gain estimate is intentionally conservative. EXP-004 already captures a large augmentation benefit, leaving less headroom, and full-preactivation evidence is strongest at much greater depth.

## Throughput and Fixed-Budget Feasibility

The architecture keeps every convolution shape and count fixed and still executes 19 BatchNorm and 19 ReLU operations per forward pass. It adds no forward pass, loss work, model copy, data transformation, or synchronized GPU operation. BN/ReLU tensor locations change, so kernel scheduling is not guaranteed identical, but a large GPU-step cost is not expected.

EXP-004 measured strong-loader throughput at 165.5-175.8 batches/s versus approximately 127.9 optimizer steps/s over the full run. That leaves host-side augmentation headroom if preactivation is modestly faster, while the accepted loader switch costs about 2.6 seconds outside counted training. The fixed 300-second training budget and 600-second wall limit remain feasible.

Use these diagnostics:

- Expected optimizer-step band: approximately 37,207-39,509 (within +/-3% of 38,358).
- A reduction beyond 5% (below about 36,440 steps) is a material throughput confound even if the run remains valid.
- Expected total runtime is near EXP-004's 340.7 seconds and comfortably below 600 seconds.
- Expected VRAM remains near EXP-004's 330.1 MB because parameter and activation shapes are essentially unchanged.

The loader switch, evaluations, and input production remain outside the synchronized per-step timing as implemented by the accepted harness. Do not move timing boundaries during this experiment.

## Risks and Failure Modes

- **Shallow-depth non-transfer:** nine residual units may not need the improved gradient path enough to move top-1. The original paper's strongest gains occur at much greater depth.
- **Over-regularized composition:** RandAugment already supplies strong invariance pressure; preactivation BN regularization may raise training difficulty without adding complementary test accuracy.
- **Strong/weak BN shift:** preactivation changes where running statistics are collected. The accepted 20% weak tail should adapt them, but a final BN plus input-side block BNs may respond differently from post-activation BN. Do not add a separate BN recalibration pass, which would change the fixed-budget intervention.
- **High-LR interaction:** moving normalization before convolutions changes activation and gradient distributions. The fixed `lr=0.1` schedule may not be optimal for preactivation, but changing it would defeat causal attribution.
- **Option-A transition limitation:** zero-padded transition shortcuts preserve exact isolation but differ from projection-based preactivation implementations. Learned projections are a separate experiment.
- **Kernel-order throughput loss:** EXP-003 showed that seemingly small operations can reduce fixed-budget steps. Report step count and do not attribute a metric change solely to representation if exposure falls materially.
- **Implementation drift:** `bn1` using `out_channels`, shortcutting the preactivated tensor, retaining stem/post-add ReLU, omitting final BN/ReLU, or changing the loader switch would test another architecture.
- **Misreading strong-phase accuracy:** EXP-004 jumped 6.83 points immediately after returning to weak inputs. Low strong-phase clean-test accuracy is expected and does not independently diagnose learned representation failure.
- **Fixed-seed variation:** architecture changes GPU timing and can indirectly perturb asynchronous data delivery. Run the predeclared fixed seed once; do not reroll or repeat for a favorable augmentation stream.

## Why the Composition Is Controlled

EXP-004 is the moving baseline and the current `train.py` already contains its accepted data, optimizer, worker, and evaluation policy. EXP-005 shows that boundary tuning is harmful, so the 80% phase composition is held fixed. The proposed diff should be confined to `BasicBlock`/`ResNet`: normalization and activation ordering plus the required final BN.

No augmentation strength, phase duration, loader lifecycle, loss, batch size, optimizer, schedule, momentum, weight decay, seed, validation, width, depth, shortcut capacity, initialization, averaging, or precision change is bundled. Exact parameter parity and expected step parity make the result interpretable as the incremental value of preactivation on the best accepted data recipe.

## Implementation and Verification

### Preflight

1. Confirm the moving baseline is 92.30%; improvement therefore requires at least 92.40%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM.
3. Confirm no stale `run.log` or renamed run-log variant exists.
4. Implement only the architecture changes above in `train.py`.
5. Run syntax, Ruff/pre-commit, diff, and tracked-scope checks.
6. Run the CPU shape and exact parameter-count assertions.
7. Inspect the diff to confirm all transform definitions, `make_train_loader`, `shutdown_train_loader`, switch predicates, optimizer, LR logic, loss, seed, and evaluation logic are unchanged.

### Execution

Run exactly once with required redirection:

```bash
uv run train.py > run.log 2>&1
```

Supervise without streaming the full log and terminate as failure if wall time exceeds 600 seconds.

### Post-run Checks

- Require exit code 0 and one complete finite ten-field summary.
- Require `best_test_acc >= 92.40%` for improvement.
- Require counted training near 300 seconds and total runtime below 600 seconds.
- Require `num_params == 269722`.
- Confirm one augmentation switch at about 80.0%, old worker count equals the configured worker count, all old workers exited, and no duplicate switch occurred.
- Confirm the crossing batch used strong augmentation and `lr=0.1`, while the subsequent weak phase used the `0.01`-to-`1e-4` cosine tail.
- Confirm no epoch has more than one evaluator call and terminal evaluation matches the summary epoch.
- Compare `num_steps` to 38,358, total runtime to 340.7 seconds, peak VRAM to 330.1 MB, and best/final accuracy and test loss to EXP-004.
- Interpret early strong-phase and weak-tail trajectories separately; the clean-test jump at the switch is part of the accepted recipe.
- Do not retry with a different seed. Record the result and remove `run.log` before another experiment.

## Decision Rule

- **Improvement:** accept the composition only at `best_test_acc >= 92.40%` with all protocol and integrity checks passing.
- **No improvement with preserved throughput:** reject same-width preactivation on the accepted RandAugment recipe at this shallow fixed horizon; retain EXP-004 unchanged.
- **No improvement with material step loss:** reject the implementation under the wall-clock objective and record throughput as a confound rather than concluding representation quality alone failed.
- **Accuracy improves but misses 92.40%:** formal verdict remains no-improvement; do not rerun or tune the seed.
- **Invalid/crash/worker leak:** repair only the implementation or lifecycle fault and rerun the same predeclared design.

## Evidence

- `TASK.md` and `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only `train.py` may change; one H20, fixed counted-time budget, fixed evaluator, no seed hacking, and 600-second total limit.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`: preserve the 80% strong-augmentation boundary, worker-side throughput, bounded evaluation, and accepted long high-LR exploration.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`: EXP-004 is the 92.30% moving baseline; EXP-005 boundary tuning regressed.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/004/04-analysis.md`: accepted RandAugment recipe, 38,358-step exposure, one-switch behavior, runtime, VRAM, and phase-wise accuracy.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/005/04-analysis.md`: the 75% switch preserved throughput but lost 0.18 points, requiring exact restoration of the 80% boundary.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/004/proposals/idea-03.md`: prior exact same-width preactivation design and parameter derivation.
- [He et al., "Identity Mappings in Deep Residual Networks," ECCV 2016](https://arxiv.org/abs/1603.05027): full-preactivation mechanism, CIFAR evidence, and shallow-depth caveat.

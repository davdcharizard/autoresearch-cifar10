# Idea: Width-2 ResNet-20 on the Accepted RandAugment Plateau Recipe

## Summary

Double every model channel width from `16/32/64` to `32/64/128` while preserving ResNet-20 depth, post-activation basic blocks, parameter-free Option-A shortcuts, and the complete accepted EXP-004 training recipe. The data policy remains `RandAugment(num_ops=1, magnitude=7)` through the 80% high-LR plateau, followed by the existing deterministic worker shutdown and crop/flip hard-label refinement tail. Optimizer, elapsed-time LR schedule, batch size, loss, worker policy, evaluator, seed, initialization rule, and timing boundaries remain unchanged.

This is a capacity-throughput trade, not a free model upgrade. The widened model has exactly 1,073,962 trainable parameters, 3.98x the current 269,722, while a local synchronized GPU preflight measured only a 1.44x step-time increase on the H20. The raw 300-second projection is 27,645 steps (3.54M examples), versus 39,921 synthetic control steps and 38,358 steps in the real EXP-004 run. Calibrating the synthetic ratio to EXP-004 yields approximately 26,563 full-run steps. The experiment is feasible, but it deliberately gives up about 28-31% of optimizer updates to gain nearly fourfold capacity.

## Diagnosis

The accepted data and optimizer policy is better established than the representation consuming it:

- EXP-002 validated an 80% `lr=0.1` exploration plateau followed by a step to `0.01` and cosine refinement to `1e-4`.
- EXP-004 composed one-operation magnitude-7 RandAugment through that boundary with a weak crop/flip tail and improved best accuracy from 91.83% to 92.30%, while retaining 38,358 steps, 99 epochs, 340.7 seconds total runtime, 330.1 MB peak VRAM, and 269,722 parameters.
- EXP-005 moved the data switch to 75% and regressed to 92.12%; EXP-006 replaced RandAugment with fixed Cutout and regressed to 91.63%. The N1/M7-through-80% data recipe should therefore remain untouched.
- Label smoothing improved NLL but not top-1 and reduced update exposure. Added per-step work must be benchmarked explicitly under this wall-clock objective.

The current model is extremely small for the available H20: 0.27M parameters and 330 MB peak allocation on a roughly 98 GB device. Its final representation has only 64 channels. With augmentation and schedule already productive, insufficient feature capacity is a leading untested limiter. Doubling width raises the number and diversity of features at every spatial scale without adding sequential depth or changing the number of residual additions.

The cost is fewer examples and updates in 300 seconds. Width 2 is worthwhile only if its per-update representation gain exceeds the loss of roughly 11,800 EXP-004 updates. That trade is the central hypothesis and must remain visible in analysis.

## Literature Basis and Transfer Limits

Zagoruyko and Komodakis define widening factor `k` as a multiplier on residual feature planes and show consistent CIFAR gains as width increases across several shallow residual depths. They also observe that parameter count and theoretical computation grow quadratically with `k`, while wider tensors use GPUs more efficiently than thousands of small sequential kernels. Their `B(3,3)` two-3x3-convolution block performed best among tested block-depth variants. Source: [Wide Residual Networks](https://arxiv.org/abs/1605.07146).

This proposal uses the paper as directional evidence, not as a reproduced WRN configuration. The paper's CIFAR WRNs use preactivation, often much larger width factors, 200 epochs, Nesterov, weight decay `5e-4`, and different preprocessing. This experiment deliberately retains the accepted post-activation block, ordinary momentum, `1e-4` weight decay, roughly 67-71 projected epochs, and plateau-only RandAugment. Importing preactivation, dropout, stronger decay, or the paper's schedule would bundle several untested mechanisms and defeat attribution.

## Exact Architecture Diff

### Width Constants

Add one model hyperparameter and derive all channel widths from it:

```python
WIDTH_MULTIPLIER = 2
STAGE_CHANNELS = tuple(
    channels * WIDTH_MULTIPLIER for channels in (16, 32, 64)
)
```

Use `c1, c2, c3 = STAGE_CHANNELS` in `ResNet.__init__`:

```python
self.conv1 = nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(c1)
self.layer1 = self._make_layer(c1, c1, num_blocks, stride=1)
self.layer2 = self._make_layer(c1, c2, num_blocks, stride=2)
self.layer3 = self._make_layer(c2, c3, num_blocks, stride=2)
self.fc = nn.Linear(c3, num_classes)
```

The resulting shape sequence is:

| Location | Output shape for batch 128 |
|---|---|
| Stem | `128 x 32 x 32 x 32` |
| Stage 1 | `128 x 32 x 32 x 32` |
| Stage 2 | `128 x 64 x 16 x 16` |
| Stage 3 | `128 x 128 x 8 x 8` |
| Pool/classifier | `128 x 128` -> `128 x 10` |

Keep `NUM_BLOCKS=3`; the network still has a stem convolution, nine two-convolution residual blocks, and one classifier, preserving the existing ResNet-20 depth convention.

### Block and Activation Order

Do not modify `BasicBlock` logic. It remains post-activation:

```text
conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> add shortcut -> ReLU
```

Both convolutions remain bias-free 3x3 operations. This is width scaling only, not a preactivation or block redesign experiment.

### Option-A Shortcut Implications

The stem and stage 1 are both widened to 32 channels, so stage 1 remains a same-shape identity shortcut with no padding. The two transition blocks behave as follows:

- Stage 2 transition: raw `32 x 32 x 32` input is sliced spatially with stride 2 to `32 x 16 x 16`, then 32 zero channels are appended to match the 64-channel residual output.
- Stage 3 transition: raw `64 x 16 x 16` input is sliced to `64 x 8 x 8`, then 64 zero channels are appended to match the 128-channel residual output.

Retain the exact existing code:

```python
shortcut = shortcut[:, :, :: self.stride, :: self.stride]
shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
```

`pad_channels` scales from 16/32 in the accepted model to 32/64. Do not add 1x1 projection shortcuts: they would add trainable capacity, GPU work, and a second architectural mechanism. Appending rather than symmetrically splitting zero channels also stays unchanged.

### Initialization

Preserve the current initializer exactly:

```python
if isinstance(m, (nn.Conv2d, nn.Linear)):
    init.kaiming_normal_(m.weight)
```

Kaiming initialization derives the appropriate variance from each wider tensor's fan-in. Keep convolution biases disabled, the classifier bias behavior unchanged, and BatchNorm defaults (`weight=1`, `bias=0`). Do not zero-initialize residual branches or copy/tile narrow weights. This remains fresh fixed-seed training, not network morphism.

Widening consumes more random draws during initialization and can therefore shift later DataLoader worker seeds/augmentation draws under the existing global RNG flow. Do not reseed or add a new generator solely to mimic EXP-004; that would change accepted data semantics. Record this inherent fixed-seed confound and avoid claiming the exact delta is capacity-only.

## Exact Parameter Count

All counts include trainable affine BatchNorm parameters and the classifier bias, but exclude non-trainable BN running buffers.

| Component | Derivation | Parameters |
|---|---:|---:|
| Stem convolution | `3 * 32 * 3 * 3` | 864 |
| Stage 1 convolutions | `3 blocks * 2 * 32 * 32 * 3 * 3` | 55,296 |
| Stage 2 convolutions | `32*64*9 + 64*64*9 + 4*(64*64*9)` | 202,752 |
| Stage 3 convolutions | `64*128*9 + 128*128*9 + 4*(128*128*9)` | 811,008 |
| Classifier | `128 * 10 + 10` | 1,290 |
| Stem BN affine | `2 * 32` | 64 |
| Stage 1 BN affine | `6 * 2 * 32` | 384 |
| Stage 2 BN affine | `6 * 2 * 64` | 768 |
| Stage 3 BN affine | `6 * 2 * 128` | 1,536 |
| **Total** | | **1,073,962** |

The current model has 269,722 parameters, so the exact increase is 804,240 parameters and the ratio is approximately 3.98x. Quadratic growth dominates because almost all weights are same-width 3x3 convolutions.

Static preflight must instantiate the model and require:

```python
assert sum(p.numel() for p in model.parameters()) == 1_073_962
assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
```

## Accepted Settings That Must Remain Fixed

Outside width-derived model channel arguments and an optional width label in startup output, keep current `train.py` unchanged:

- `RandAugment(num_ops=1, magnitude=7)` after crop/flip and before tensor conversion during the first 80% of counted training;
- exact weak transform of crop, flip, tensor conversion, and current normalization;
- strong loader first, one post-crossing break, scheduled evaluation, explicit persistent-worker shutdown, `gc.collect()`, and weak-loader reconstruction exactly at the 80% LR boundary;
- `make_train_loader` and `shutdown_train_loader`, `NUM_WORKERS=8`, batch 128, shuffle, pinning, drop-last, persistent workers, and forkserver context;
- hard-label `F.cross_entropy`;
- SGD with `lr=0.1`, ordinary momentum 0.9, weight decay `1e-4`, and no Nesterov;
- `lr=0.1` through 80%, then the current step to `0.01` and elapsed-time cosine to `1e-4`;
- evaluation checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense at most once-per-epoch evaluation in the tail, and terminal evaluation;
- seed 42, timing boundaries, maximum-step guard, fixed evaluator, and summary schema.

Do not import the WRN paper's preactivation, dropout, stronger weight decay, Nesterov, reflection padding, ZCA preprocessing, or 200-epoch schedule. Do not change batch size to recover throughput; that is a separate optimizer-noise experiment.

## Measured Synthetic-GPU Feasibility

The EXP-007 local H20 preflight used pinned host inputs, nonblocking H2D transfer, the exact batch-128 SGD forward/backward/update path, CUDA synchronization, 50 warmup steps, and 200 timed steps. It measured:

| Base width | Parameters | Mean synchronized step | Raw 300s steps | Raw examples | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 16 | 269,722 | 7.515 ms | 39,921 | 5.11M | 330.1 MB |
| 24 | 605,026 | 9.251 ms | 32,428 | 4.15M | 427.0 MB |
| 32 | 1,073,962 | 10.852 ms | 27,645 | 3.54M | 599.2 MB |

The synthetic width-16 projection is 4.1% above EXP-004's actual 38,358 steps. A ratio-calibrated width-32 estimate is:

```text
38,358 * 7.515 / 10.852 = approximately 26,563 steps
```

This corresponds to about 68.1 epochs (`26,563 / 390`), approximately 21,250 strong-phase steps and 5,313 weak-tail steps, or about 54.5 strong epochs plus 13.6 weak epochs. The elapsed-time LR schedule still traverses its complete trajectory despite the lower step count.

### Fixed-Time Gates

The measured width-32 candidate passes the following predeclared feasibility gates:

- candidate/control synchronized-step ratio no greater than `1.50`; measured `10.852 / 7.515 = 1.444`;
- EXP-004-ratio-calibrated projection at least `26,000` optimizer steps, preserving at least 67.8% of accepted exposure; measured estimate approximately 26,563;
- at least 13 projected weak-tail epochs for low-LR refinement and BN adaptation; measured estimate approximately 13.6;
- raw candidate throughput at least 85 batches/s, below the measured strong-loader supply of 165.5-175.8 batches/s; measured approximately 92.1 steps/s;
- peak allocated VRAM below 2 GB, leaving overwhelming H20 margin; measured 599.2 MB;
- no OOM, non-finite loss, CUDA error, or shape mismatch during warmup/timing.

If the benchmark is rerun during planning because the GPU/software state changed, use the same fresh-process protocol and require all gates again. Do not relax a failed gate based on expected accuracy. The preflight establishes feasibility only and must not be counted as evidence that top-1 will improve.

## Why Width 2 Instead of Width 1.5

Width 24 is the lower-risk operating point: it measured 9.251 ms and projects 32,428 raw steps, with 605,026 parameters and 427.0 MB VRAM. Calibrated to EXP-004, it would retain roughly 31,160 steps, about 81% of accepted exposure.

Width 32 remains the stronger isolated proposal for three reasons:

1. It is the direct integer `k=2` capacity intervention motivated by the WRN literature and nearly quadruples convolutional capacity, whereas width 24 provides only about 2.24x parameters.
2. H20 parallel efficiency is favorable: 3.98x parameters cost only 1.44x synchronized step time. Width 2 therefore tests whether the current representation is genuinely capacity-limited rather than making a marginal adjustment likely to fall near the 0.10-point decision threshold.
3. It still clears a meaningful fixed-budget floor: about 26.6k calibrated updates, 3.40M calibrated examples, and more than 13 weak-tail epochs. EXP-004's weak switch produced a 6.83-point immediate recovery and peaked later, so the shorter tail is risky but not obviously inadequate.

This defense does not make width 2 strictly dominant. If EXP-007 fails with clear optimization lag and throughput exactly as projected, width 1.5 becomes the principled follow-up because it preserves roughly 4,600 additional calibrated updates while still more than doubling capacity.

## Accuracy Hypothesis

**Primary hypothesis:** width-2 ResNet-20 composed with the accepted N1/M7 plateau and weak hard-label tail will raise `best_test_acc` from 92.30% to at least 92.40%, with a plausible range of 92.50-93.00% (+0.20 to +0.70 points). The wider stages should learn a richer set of augmentation-stable features and reduce capacity underfit enough to outweigh the loss of approximately 11.8k updates.

**Secondary predictions:**

- The strong-to-weak switch occurs once at about 80.0%; the crossing batch uses `lr=0.1`, and the weak phase uses the complete `0.01`-to-`1e-4` cosine tail.
- `num_params` equals 1,073,962 and peak VRAM remains near the measured 599 MB.
- Full-run steps land approximately in the 26,000-28,000 range, with about 67-72 epochs and 13-14 weak-tail epochs.
- Total wall time remains roughly 335-380 seconds. Fewer epochs reduce the number of dense-tail evaluator calls, partially offsetting the wider evaluator model; the 2.6-second worker transition remains unchanged.
- Strong-phase clean-test accuracy may remain low because EXP-004 established a large augmentation-distribution mismatch. The decisive comparison is the weak-tail peak, not the final strong checkpoint alone.

The published WRN gains are not used to predict a larger number because this run has far fewer epochs, post-activation blocks, different regularization, and a strict fixed-time loss of update exposure.

## Confounds

- **Capacity and update count are inseparable under the objective:** width changes both representation and how many batches fit in 300 seconds. The experiment evaluates their net value, not width at equal steps.
- **Examples seen fall materially:** the raw projection decreases from 5.11M synthetic-control examples to 3.54M; the calibrated expectation is about 3.40M versus EXP-004's 4.91M.
- **Initialization/RNG stream changes:** wider tensors consume more random values before the first DataLoader iterator is created, which may change worker seeds, shuffle, and augmentation draws even though seed 42 and policy are unchanged. Do not reroll or artificially realign the stream after seeing results.
- **Same regularization at larger capacity:** weight decay remains `1e-4` and no dropout is added. RandAugment may be sufficient, but the larger model could overfit; changing decay simultaneously would prevent attribution.
- **Option-A capacity distribution:** transition shortcuts carry identity information in the first half of output channels and zeros in the new half. Width doubles both halves without adding learned projections.
- **Synthetic-to-real timing:** the microbenchmark omits DataLoader lifecycle, logging, evaluation, and real batch variation. Ratio calibration reduces but does not remove this uncertainty.
- **Evaluation count changes:** fewer epochs imply fewer dense-tail evaluations. `best_test_acc` may have fewer observation opportunities, although elapsed-time annealing and tail evaluation remain once per completed epoch.

## Failure Modes

- **Under-optimization:** 26-28k steps and roughly 68 epochs may be insufficient for a fourfold-parameter model, especially during the high-LR strong-view phase. Accuracy may lag despite greater representational ceiling.
- **Too-short weak tail:** approximately 13-14 crop/flip epochs may not fully adapt wider BN statistics or refine the clean objective; do not move the 80% switch in the same experiment because EXP-005 rejected that change.
- **Capacity does not limit ResNet-20:** the accepted model may be data/noise limited, so width adds cost without useful features.
- **Overfitting despite RandAugment:** the larger model can memorize the finite training set during the weak tail under unchanged weight decay.
- **GPU benchmark drift:** contention, clocks, or software state may make the real candidate slower than preflight. Below 26,000 steps, throughput becomes a primary failure mechanism.
- **Host starvation is unlikely but observable:** width-32 GPU demand is about 92 batches/s versus measured strong supply above 165 batches/s. If loader stalls appear, verify worker health rather than changing loader settings mid-experiment.
- **Option-A mismatch bug:** failing to scale padding to 32/64, changing pad placement, or accidentally projecting shortcuts changes shapes or tests another model.
- **OOM is unlikely but still gated:** measured 599.2 MB is tiny relative to the H20, but a memory anomaly or leak remains an abort condition.
- **Threshold variance:** 0.10 points is ten CIFAR-10 test examples. One fixed-seed run must stand; no retries or seed selection.

## Implementation Scope

Modify only `train.py`:

1. Add `WIDTH_MULTIPLIER=2` and derived stage widths.
2. Replace hard-coded model channel arguments with `32/64/128` derived values.
3. Point the classifier at 128 input features.
4. Optionally include `W2` in the startup model label; preserve summary keys.
5. Make no other source change.

Do not modify `prepare.py`, dependencies, lockfiles, evaluator, dataset files, or autoresearch tracked experiment code during execution. The proposal itself is planning metadata; the eventual experiment diff must contain only the reviewed `train.py` capacity change.

## Verification Protocol

### Static and Environment Preflight

- Confirm the moving baseline remains 92.30%, so formal improvement requires at least 92.40%.
- Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM and no compute process.
- Confirm no stale `run.log` or renamed run-log artifact.
- Run Python compilation, Ruff, pre-commit, and tracked-scope checks.
- Require the exact parameter assertion `1_073_962` and output shape `(2, 10)`.
- Inspect every model tensor shape and require Option-A transition pads of 32 and 64 channels.
- Diff against the accepted EXP-004 code and require all data, worker, optimizer, loss, schedule, seed, timing, and evaluator lines unchanged.

### GPU Microbenchmark Gate

- Use a fresh process on the idle H20 with batch 128, pinned host inputs/targets, nonblocking H2D inside timing, exact SGD/CE forward-backward-update, CUDA synchronization, no AMP or compilation, 50 warmup steps, and 200 timed steps.
- Benchmark width 16 control and width 32 candidate in separate clean model/optimizer instances under the same process protocol.
- Record mean, median, p95, raw 300-second step projection, relative ratio, non-finite/error status, and peak allocation.
- Require ratio `<=1.50`, calibrated projection `>=26,000`, projected weak tail `>=13` epochs, raw throughput `>=85` steps/s, and peak allocation `<2 GB`.
- The existing measured values (7.515 ms control, 10.852 ms candidate, ratio 1.444, calibrated 26,563 steps, 599.2 MB) pass. Do not run another full training experiment as part of proposal development.

### Eventual Full Run

After planning/review approval, execute exactly once with required redirection and a 600-second supervisor:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

The current task does not authorize running this full experiment.

### Post-run Verification

- First require `best_test_acc >= 92.40%`; otherwise record no-improvement without retries.
- If primary accuracy passes, require exit 0, ten unique finite summary keys, counted training in the accepted approximately 300-second band, and total time below 600 seconds.
- Require `num_params == 1073962` and inspect peak VRAM against the approximately 599 MB preflight.
- Require exactly one augmentation switch at/just after 80%, all old strong workers stopped, weak training resumed, and no worker leak.
- Require no epoch has more than one evaluator call and terminal evaluation matches `num_epochs`.
- Compare steps against both raw 27,645 and calibrated 26,563 projections. Report actual/control retention and images seen.
- Compare phase checkpoints, best/final gap, final loss, epochs, and weak-tail length with EXP-004. Do not interpret the strong checkpoint without the known distribution-shift caveat.
- Preserve seed 42 and never rerun a valid result for a more favorable augmentation stream.
- Remove `run.log` after recording/analyzing the completed experiment.

## Decision Rule

- **Improvement:** accept width 2 only if `best_test_acc >= 92.40%` and all integrity/runtime conditions pass.
- **No improvement with steps >=26,000:** width 2 fails as a net capacity-throughput trade at this fixed horizon; inspect whether training or test error indicates underfit versus overfit.
- **No improvement with steps <26,000:** record throughput shortfall as a primary confound and revert; do not tune batch size or schedule inside the same experiment.
- **Clear optimization lag with expected throughput:** prioritize an isolated width-1.5 follow-up, which measured 9.251 ms and should retain roughly 31.2k calibrated updates.
- **Clear overfit with successful optimization:** retain the accepted narrow model or separately review stronger weight decay; do not append dropout or decay opportunistically.
- **Invalid/crash/OOM/worker leak:** fix only the implementation or protocol fault and rerun the same predeclared configuration.

## Evidence

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: metric, only-`train.py` scope, fixed-budget protocol, H20 requirement, validation cap, and no-seed-hacking rule.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/007/01-brainstorm.md`: capacity diagnosis, width-2 seed, accepted composition, and alternatives.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv` and `03-experiment-learnings.md`: 92.30% moving baseline; accepted schedule/data patterns and rejected regularization/boundary replacements.
- `train.py`: current post-activation ResNet-20, exact Option-A shortcut, worker lifecycle, transforms, optimizer, elapsed-time schedule, and evaluator cadence.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/004/04-analysis.md`: accepted metric, 38,358 steps, 340.7-second runtime, 330.1 MB VRAM, 269,722 parameters, and strong-to-weak trajectory.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/004/02-plan.md`: exact N1/M7 loader policy, preflight methodology, switch semantics, and execution verification.
- [Zagoruyko and Komodakis, "Wide Residual Networks"](https://arxiv.org/abs/1605.07146): width definition, quadratic complexity, GPU-efficiency argument, `B(3,3)` evidence, CIFAR results, and transfer caveats.
- EXP-007 local synthetic H20 preflight supplied for proposal development: width-16/24/32 step timing, projected exposure, and peak allocation; no full training run was performed.

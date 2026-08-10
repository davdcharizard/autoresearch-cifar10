# Proposal 01: Identity-Scale Final-Stage ECA Recalibration

## Hypothesis and mechanism

Add ECA-style example-dependent channel recalibration to exactly the three `layer3` residual branches of the accepted EXP-010 width-2 postactivation ResNet-20. Each gate globally pools the 128-channel second-BN residual, mixes neighboring channel descriptors with a bias-free length-5 `Conv1d`, applies `2 * sigmoid`, and scales the residual immediately before shortcut addition. Zero Conv1d weights make every initial gate exactly one, so the accepted forward function, active residual strength, shared gradients, BatchNorm updates, and transition behavior are preserved at initialization while the 15 new weights recruit on the first backward.

The narrow final-stage scope targets class-semantic channel allocation without letting a CutMix-blended global descriptor control early feature extraction. It should retain at least 98% of EXP-010's 26,898 updates and improve late representation quality enough to raise `best_test_acc` from 94.15% to at least 94.25%.

## Exact implementation

Add an `ECAGate` whose only parameter is `nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)`, initialized with `init.zeros_`. Protect the constructor draw with `torch.random.fork_rng(devices=[])` so all shared tensors and the post-construction CPU RNG state remain aligned. Its forward is exactly:

```python
descriptor = F.adaptive_avg_pool2d(residual, 1).flatten(2).squeeze(-1).unsqueeze(1)
logits = self.channel_conv(descriptor)
gate = (2.0 * torch.sigmoid(logits)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
return residual * gate
```

Give `BasicBlock` an explicit `use_eca=False` flag. Set it true for all and only `layer3[0]`, `layer3[1]`, and `layer3[2]`. In each selected block, apply ECA after `out = self.bn2(self.conv2(out))` and immediately before the unchanged Option-A/identity shortcut addition and final ReLU. Do not gate shortcuts or post-add tensors, and do not add ECA elsewhere.

Kernel size is fixed at 5, corresponding to ECA's `gamma=2, b=1` mapping for 128 channels; it is not tuned. Three five-weight kernels add exactly 15 parameters, taking the model from 1,073,962 to 1,073,977 parameters. Put them in the existing single standard SGD group with LR 0.1, momentum 0.9, and coupled all-parameter weight decay `1e-4`. Preserve every EXP-010 data, CutMix, schedule, seed, timer, worker, evaluator, FP32, batch-128, and logging semantic. No separate LR, clipping, warmup, optimizer-state change, gate clamp, or production instrumentation is allowed.

## Why this is distinct from prior failures

- Unlike EXP-012/015, exact unit gates do not suppress or deactivate residual branches; postactivation ordering and initial branch activity stay accepted.
- Unlike EXP-014's independent raw-max classifier, this is a bounded `(0,2)` multiplicative path with no direct class logits, though first-update recruitment is still explicitly gated.
- Unlike EXP-017/021, both Option-A transitions and their spatial/channel provenance remain untouched.
- Unlike EXP-018, there is no weight averaging, BN recalibration, or reserved non-SGD tail.
- Unlike EXP-019, the augmentation policy is unchanged; preflight persists exact post-transform batches so forkserver replay drift cannot alter a verdict.
- Unlike EXP-020/022, ordinary momentum SGD and its parameter/state alignment remain unchanged. Full-forward BF16 from EXP-016 is not used.

## Preflight safety and feasibility gates

Before production, require exactly three zero `Conv1d(1,1,5)` modules, 15 added weights, one unchanged optimizer group, shared state and CPU RNG equality, exact unit gates, and bitwise-equal train/eval outputs, BN state, and shared gradients at initialization. Require finite nonzero gradients for every ECA kernel on both hard-label and valid CutMix probability-target batches, including `layer3[0]` with its unchanged Option-A shortcut.

Persist one immutable 200-batch production N1/M7/CutMix corpus before running either arm; record its SHA-256 and hard/soft counts. Replay byte-aligned control and candidate arms under identical deterministic diagnostic settings, serialize the complete report before assertions, and never regenerate the corpus to rescue a failure. Separately inspect the first update for hard and CutMix cases. Require all state finite; after step one, `max(abs(ECA weight)) <= 0.25`, all gates within `[0.75,1.25]`, and each gate mean within `[0.95,1.05]`. Across 200 batches require candidate/control loss-EMA ratio no greater than 1.5, no candidate-only prediction share above 95%, gates within `[0.5,1.5]`, and finite nonzero gradients in all three kernels. Report gate mean/p01/p99 separately for hard and soft batches. Any violation blocks production without threshold relaxation or design rescue.

Measure five alternating fresh-process control/candidate training pairs on one idle H20, using production FP32 settings, 100 warmup steps, and at least 500 synchronized timed steps spanning hard and probability targets. Time H2D, forward, loss, backward, SGD, and synchronization; report stage ratios because backward is 75.46% of accepted step cost. Advance only if candidate/control median step ratio is at most 1.02 in aggregate, no pair exceeds 1.04, p95 ratio is at most 1.04, paired-ratio CV is below 2%, peak allocation is below 700 MiB without growth, and projected exposure is at least 26,360 updates. Five evaluator-batch-256 inference pairs must have median ratio at most 1.05, and projected total wall time must remain below 540 seconds. Failure retires this fixed three-gate design; do not alter scope or kernel.

## Production decision and falsification

Run one seed-42 production job only after all gates pass. Require one H20, only `train.py` changed, 300 counted seconds, total below 600 seconds, one 80% augmentation switch with eight workers stopped, accepted evaluation cadence, finite summary, exactly 1,073,977 parameters, and at least 26,360 realized updates. Formal success is `best_test_acc >= 94.25%`.

Compare switch accuracy with EXP-010's 89.73% (87.08% is the recurring strong-underfit marker), first weak accuracy with 93.16%, and final NLL with 0.1934. Mechanism support predicts preserved switch fit, a stronger weak-tail trajectory, and lower final NLL; a top-1 pass without those signatures is a metric improvement but weak evidence for better calibration.

The hypothesis is falsified for this exact operating point if a valid full run misses 94.25%, falls below the exposure floor, repeats strong-phase suppression, or worsens late NLL despite healthy fit. Key risks are CutMix descriptor ambiguity, arbitrary channel adjacency, rapid post-identity amplification, sequential tiny-kernel backward cost, and an effect smaller than the ten-example acceptance margin. Do not retry one gate, all stages, another kernel, standard sigmoid, another placement, or a special optimizer after observing the result.

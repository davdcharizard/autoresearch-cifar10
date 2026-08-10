# Proposal 01: Identity-Initialized Final-Stage ECA

## Intervention and hypothesis

Add ECA-style, example-dependent channel recalibration to all and only the three `layer3` residual branches of the accepted width-2 postactivation ResNet-20 at commit `7c1e7d8`. For each 128-channel residual, globally average its spatial map, mix adjacent channel descriptors with a bias-free length-5 `Conv1d`, apply `2 * sigmoid`, and multiply the resulting gate into the residual immediately before the unchanged shortcut addition and final ReLU.

Zero-initializing each ECA kernel makes every initial gate exactly one. The candidate therefore begins with the accepted forward function and active residual strength rather than suppressing branches as EXP-012/015 did. Once gradients recruit the 15 ECA weights, the final stage can condition semantic channel emphasis on each image without removing ResNet-20 depth or globally widening every convolution. Restricting attention to the final 8x8 stage also keeps CutMix-blended global descriptors from controlling early feature extraction.

The primary hypothesis is that this small conditional-capacity addition preserves the accepted strong-view fit and fixed-time exposure, improves weak-tail feature allocation, and raises `best_test_acc` from 94.15% to at least 94.25%. The point prediction is at least 26,000 updates, a switch checkpoint near EXP-010's 89.73%, and a peak accuracy of 94.30-94.45% with final NLL no worse than 0.1934.

## Evidence and diagnosis

- EXP-007 established that added channel capacity is valuable under N1/M7: width 2 gained 1.25 points despite fewer updates. EXP-010 then showed that this capacity can absorb plateau-only CutMix and reach the current 94.15% frontier.
- EXP-023 rejects a global width-3/depth-14 trade, not conditional capacity: its 1.54M-parameter model entered the weak tail well but peaked at 94.00% after removing one block per stage. EXP-024 preserves all nine accepted blocks and adds capacity only where class-semantic features are most mature.
- ECA-Net reports that channel-local interaction without an SE reduction bottleneck can improve ResNets with very few weights. The transfer is plausible but not established for a shallow CIFAR network, regional CutMix targets, or a 300-second budget; local safety and timing evidence is decisive.
- The main systems risk is launch cost, not arithmetic or VRAM. Backward consumes 75.46% of the accepted step, and EXP-012 measured a 1.233x step ratio for nine identity-initialized SE gates even though their formal FLOPs were tiny. Three final-stage ECA gates remove the SE bottleneck MLP and two-thirds of the gate sites, but pooling, Conv1d, sigmoid, broadcast multiplication, and their backwards remain sequential kernel launches.
- The main optimization risk occurs after initialization. Exact unit gates preserve the step-zero function, but EXP-014 showed that apparent identity/zero-output initialization does not guarantee a safe first update. ECA's bounded `(0, 2)` multiplier and lack of direct class logits lower that risk, but gate movement and prediction concentration must be measured on immutable production batches before timing or scoring.

## Exact implementation

Add an `ECAGate` module whose sole parameter is:

```python
self.channel_conv = nn.Conv1d(
    1, 1, kernel_size=5, padding=2, bias=False
)
init.zeros_(self.channel_conv.weight)
```

Construct the `Conv1d` inside `with torch.random.fork_rng(devices=[]):` so its default constructor draw does not perturb the global CPU RNG stream before later shared modules such as `fc` are initialized. The existing model-wide `_weights_init` handles only `Conv2d` and `Linear`, so it must leave the explicitly zeroed `Conv1d` unchanged. ECA forward is exactly:

```python
descriptor = F.adaptive_avg_pool2d(residual, 1).flatten(2).transpose(1, 2)
logits = self.channel_conv(descriptor)
gate = (2.0 * torch.sigmoid(logits)).transpose(1, 2).unsqueeze(-1)
return residual * gate
```

For a residual shaped `[N, 128, 8, 8]`, this produces descriptors/logits `[N, 1, 128]` and gates `[N, 128, 1, 1]`. Kernel size 5 is pre-registered from ECA's deterministic `gamma=2, b=1` mapping at 128 channels; it is not target-tuned.

Give `BasicBlock` an explicit `use_eca=False` constructor argument and set `self.eca = ECAGate(out_channels) if use_eca else None`. Extend `_make_layer` to pass the flag and invoke it with `use_eca=True` only for `layer3`. Within a selected block, retain:

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
out = self.eca(out)
# unchanged Option-A/identity shortcut
out += shortcut
return F.relu(out)
```

Do not gate the shortcut, the post-add tensor, the stem, `layer1`, or `layer2`. Do not fuse gates across blocks or replace global average pooling.

Three five-weight kernels add exactly 15 trainable parameters, taking the model from 1,073,962 to **1,073,977 parameters**. Keep them in the existing single SGD group with LR 0.1, momentum 0.9, and coupled all-parameter weight decay `1e-4`. Preserve FP32/default-TF32 execution, batch 128, seed 42, N1/M7 plus probability-0.5 alpha-1 CutMix through 80%, the hard weak tail, LR schedule, worker lifecycle, time accounting, evaluator, and logging. No special gate LR, warmup, clipping, clamp, decay exclusion, extra evaluation, or production instrumentation is part of the experiment.

## Structural and identity preflight

Before any learned safety trajectory or H20 timing, require:

1. Exactly three bias-free `Conv1d(1,1,5,padding=2)` modules, all under `layer3`, with 15 total zero ECA parameters and 1,073,977 total model parameters. Assert no attention module exists under the stem, `layer1`, or `layer2`.
2. Unchanged stage shapes `[N,32,32,32]`, `[N,64,16,16]`, and `[N,128,8,8]`; nine `BasicBlock`s; 19 `Conv2d`s; two unchanged Option-A slice/pad transitions; one classifier; unchanged postactivation ordering.
3. After construction from seed 42, bitwise equality for every shared parameter and buffer and equality of the global CPU/CUDA RNG states between control and candidate. Require the unchanged single optimizer-group hyperparameters and expected FP32 parameter, buffer, gradient, and momentum-buffer dtypes.
4. On separate hard `[N]` and valid probability `[N,10]` targets, require finite logits/loss/gradients/update, exact unit gates, bitwise-equal initial train/eval logits, identical shared gradients, identical BN state changes, and finite nonzero accumulated gradients for every ECA kernel. Evaluation must not change model state.

Any mismatch blocks the experiment. Do not weaken equality to tolerance unless the difference is first explained as an unavoidable deterministic kernel property and an adversarial plan review explicitly approves the replacement check.

## Immutable-corpus optimization safety gate

Persist one 200-batch corpus generated by the production N1/M7/CutMix path before either arm runs; record its SHA-256, shapes/dtypes, label-row sums, and hard/soft counts. Replay byte-identical batches in fresh control and candidate processes under identical deterministic diagnostic settings. Serialize the full report before assertions, and never regenerate or reorder the corpus to rescue a veto.

Separately inspect one hard and one probability-target first update. Require all state finite; every ECA kernel must move from zero with finite nonzero norm; maximum absolute ECA weight must be at most 0.25; all gate values must remain within `[0.75, 1.25]`; and each block's gate mean must remain within `[0.95, 1.05]`. Verify ordinary momentum buffers are aligned with their live parameters after the update.

Across the full 200-batch replay, require:

- finite logits, losses, gradients, parameters, buffers, and optimizer state in both arms;
- no candidate-only top-class prediction share above 95% on any batch;
- candidate terminal debiased loss EMA no greater than 1.5x control;
- every ECA kernel receives finite, nonzero cumulative gradient on both hard and soft subsets;
- each observed gate remains within `[0.5, 1.5]`, with per-block mean in `[0.85, 1.15]`;
- no candidate shared-parameter gradient norm or update norm above 2x the matched control without a benign, localized explanation approved before timing.

Report loss-EMA, prediction concentration, shared-gradient/update ratios, ECA weight norms, and gate mean/p01/p50/p99 separately for hard and probability-target batches. These are safety vetoes, not miniature accuracy selection. Failure retires this exact initialization/scope for EXP-024; do not rescue with a lower gate LR, clamp, different scale, kernel, or block subset.

## Paired H20 timing and wall-time gate

Only after safety passes, confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB and run five alternating fresh-process control/candidate pairs. Use production batch 128, FP32 settings, ordinary SGD, immutable representative hard/probability batches, 100 conditioning steps, and at least 500 complete synchronized measured training steps per arm. Weight hard and probability-target measurements 60/40: the full run spends 80% in a roughly 50/50 hard/soft strong mixture and 20% in an all-hard weak tail. Include H2D, zero-grad, forward, loss, backward, SGD, and final synchronization. Record total, forward, backward, optimizer, p50/p95, peak allocation, pair order, and coefficient of variation.

Advance to production only if all of the following hold:

- 80/20 strong/weak weighted candidate/control mean step ratio `<= 1.035`;
- no paired mean ratio `>= 1.05`, candidate p95 `<= 1.08x` the matched control mean, and control/candidate trial-mean CV `< 2%`;
- projected exposure from EXP-010's 26,898 steps is at least **26,000 updates** (96.66% retention), with separate strong/weak projections reported;
- candidate peak allocation `< 700 MiB`, no monotone allocation growth, and no OOM/retry;
- five paired single-batch inference probes using evaluator-shaped batch-256 inputs have median ratio `<= 1.05`, and startup + 300 counted seconds + loader switch + projected full-evaluation cost remains `< 540 seconds`.

The 3.5% aggregate allowance is intentionally much tighter than EXP-012's measured 23.3% SE overhead but allows realistic tiny-kernel launch noise. The 26,000-step floor prevents a nominally lightweight attention mechanism from spending enough scarce optimization exposure to confound a marginal accuracy result. A timing failure is an invalid/no-go, not evidence against ECA's representation mechanism. Do not reduce gate count, alter memory format, compile, fuse, or otherwise rescue after observing the timing result.

## Production protocol, verification, and falsification

Run exactly one seed-42 production job after every preflight passes: one idle H20, only `train.py` changed, `uv run train.py > run.log 2>&1`, 300 counted training seconds, and termination below 600 total seconds. Require one 80% augmentation switch, eight stopped workers, no probability targets after the switch, accepted at-most-once-per-epoch evaluation cadence, all finite summary fields, exactly 1,073,977 parameters, and at least 26,000 realized updates. Do not reroll the seed or retry another ECA operating point.

Formal success is `best_test_acc >= 94.25%`. For mechanism diagnosis, compare the switch checkpoint with EXP-010's 89.73% and the recurring 87.08% strong-underfit marker, the first weak checkpoint with 93.16%, and final NLL with 0.1934. Support for the intended mechanism is preserved strong fit followed by a stronger weak-tail trajectory and equal-or-better NLL. A top-1 pass without those signatures remains a valid metric improvement, but only weak evidence for channel recalibration as the cause.

The hypothesis is falsified for this exact three-gate, length-5, identity-scale operating point if a mechanically valid run misses 94.25%, realizes fewer than 26,000 updates, suppresses the strong phase, or worsens late NLL despite healthy fit. Interpret such a result as the net effect of learned final-stage channel mixing under CutMix and the fixed-time recipe; one run cannot isolate channel adjacency, global descriptor ambiguity, or the remaining exposure loss. Do not retry one gate, all stages, standard sigmoid, another kernel size, placement, scale, or optimizer treatment inside EXP-024.

## Principal risks

- **Launch-bound overhead**: three apparently tiny modules can still reduce exposure because every block adds reduction, Conv1d, sigmoid, multiply, and backward launches; EXP-012 makes this the dominant feasibility concern.
- **CutMix descriptor ambiguity**: global pooling blends the two pasted regions, while the mixed target is area-weighted; a single channel gate may amplify whichever region dominates features rather than respect target mass.
- **Arbitrary channel locality**: neighboring learned channels have no guaranteed semantic ordering, so a length-5 Conv1d may impose an unhelpful inductive bias despite ECA's ImageNet evidence.
- **Post-identity recruitment**: zero weights preserve step zero but LR 0.1 can move the small kernels quickly; the bounded gate does not prevent harmful attenuation/amplification or coupled feedback through BN.
- **Small statistical margin**: the acceptance boundary is only ten CIFAR-10 examples above 94.15%; trajectory/NLL evidence is needed to distinguish a real mechanism from a marginal max-over-checkpoints fluctuation.

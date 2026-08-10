# Proposal 02: Identity-Scale Final-Stage ECA Recalibration

## Proposal

Add identity-initialized ECA-style channel recalibration to exactly the three residual outputs in `layer3` of the accepted width-2 postactivation ResNet-20. Each module global-average-pools the 128-channel second-BN residual tensor, applies one zero-initialized channel-axis `Conv1d(k=5)`, forms `2 * sigmoid(logit)`, and multiplies only the residual immediately before shortcut addition.

Do not add ECA to `layer1`, `layer2`, the stem, shortcuts, post-add tensors, pooling, or classifier. Do not substitute all-block attention, a different kernel, paper-standard half-scale sigmoid, SE, a learned scalar, gate clipping, another placement, or a separate attention LR if this fixed candidate fails.

## Why Final Stage Only

The accepted EXP-010 frontier is 94.15%: width-2 postactivation ResNet-20, 26,898 updates, 89.73% at the N1/M7+CutMix switch, 93.16% at the first weak checkpoint, and final equal to best with 0.1934 NLL. Width and conservative CutMix supplied the strongest local gains, showing useful feature capacity but leaving open whether high-level channels are allocated efficiently on each example.

ECA-Net learns local cross-channel interactions without SE dimensionality reduction by applying a short `Conv1d` to global channel descriptors. Published gains cover deeper ImageNet ResNets, not this fixed-time CIFAR model, so the evidence is directional. Source: `knowledge/papers/eca-net.md` and [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf).

Final-stage scope is deliberate:

- 128-channel 8x8 features are the most class-semantic tensors in this network, making local channel relationships more plausible than in early edge/texture channels.
- Early RandAugment/CutMix evidence is preserved through the stem, `layer1`, and `layer2`; a global mixed-image descriptor cannot suppress those extractors.
- EXP-017's learned transition paths improved switch fit but worsened final NLL and peak accuracy. This candidate leaves Option-A and both transition computations unchanged and asks a distinct late representation question.
- Three attention chains reduce the sequential global-pool/Conv1d/sigmoid/multiply overhead that made all-block SE cost 1.233x in EXP-012.

`layer3[0]` is included because its 128-channel residual is already a high-level stage-3 feature. ECA gates only that residual after `bn2`; it does not touch the stride-2 Option-A shortcut or change channel provenance. `layer3[1]` and `[2]` use the same placement. No block selection is contingent on timing or accuracy.

## Exact Kernel, Module, and Parameters

Use the ECA paper's deterministic mapping with `gamma=2`, `b=1`, and implementation-style odd rounding:

```python
t = int(abs((math.log2(channels) + 1) / 2))
kernel_size = t if t % 2 else t + 1
```

For `channels=128`, `t=4` and `kernel_size=5`. Each bias-free `Conv1d(1,1,5)` has five weights; three blocks add exactly **15 parameters**. Total model parameters become **1,073,977** from 1,073,962. All 15 weights join the existing single SGD parameter group with LR, momentum, and coupled decay unchanged.

Use this exact module:

```python
class ECAGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        t = int(abs((math.log2(channels) + 1) / 2))
        kernel_size = t if t % 2 else t + 1
        with torch.random.fork_rng(devices=[]):
            self.channel_conv = nn.Conv1d(
                1, 1, kernel_size, padding=kernel_size // 2, bias=False
            )
        init.zeros_(self.channel_conv.weight)

    def forward(self, residual):
        descriptor = F.adaptive_avg_pool2d(residual, 1)
        descriptor = descriptor.squeeze(-1).transpose(-1, -2)
        logits = self.channel_conv(descriptor)
        gate = 2.0 * torch.sigmoid(logits)
        gate = gate.transpose(-1, -2).unsqueeze(-1)
        return residual * gate
```

The CPU RNG fork isolates `Conv1d` constructor draws; exact zero initialization then consumes no random values. The existing `ResNet.apply` only reinitializes `Conv2d` and `Linear`, so ECA kernels remain zero and all shared tensors plus post-construction CPU RNG can remain bitwise aligned with accepted control.

Add an explicit `use_eca=False` block flag. Instantiate it `True` for all three `layer3` blocks and nowhere else. Placement is literal:

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
if self.eca is not None:
    out = self.eca(out)
# accepted identity or Option-A shortcut remains unchanged
out += shortcut
return F.relu(out)
```

## Identity-Scale Initialization

Paper-standard `sigmoid` with a zero-centered logit produces a gate near 0.5 and halves the affected residual. That is unacceptable locally: full preactivation and selective zero-gamma lowered strong fit by 2.85-3.25 points despite equal exposure. This proposal uses only `2 * sigmoid(logit)` with exact-zero kernels. Every initial logit is zero and every multiplicative gate is exactly one.

At construction, candidate output, BN updates, residual strength, and all shared gradients should be bitwise accepted. The attention kernels nevertheless recruit immediately: the derivative of `2*sigmoid` at zero is 0.5, and the globally pooled nonzero residual descriptor supplies a kernel gradient. The gate range is `(0,2)`, so learned channels may amplify as well as suppress. This weakens literal paper transfer and makes first-update scale a mandatory safety question.

## Unchanged EXP-010 Contract

Preserve width 2; batch 128; FP32 eager training/evaluation; standard SGD momentum 0.9; coupled all-parameter decay `1e-4`; LR 0.1 through 80%, step to 0.01, then cosine to `1e-4`; N1/M7 plus p=0.5 alpha-1 CutMix through 80%; hard crop/flip weak tail; seed 42; accepted workers/switch; evaluator/checkpoints; and the 300-second timer. Do not use BF16 after EXP-016's real-batch instability, and do not change transition paths after EXP-017's no-improvement.

No production gate logging or extra evaluation is allowed. Safety instrumentation lives only in disposable preflight scripts.

## Structural and Initial-Function Gate

Build accepted and candidate models from seed 42 and require:

- exactly three ECA modules at `layer3[0:3]`, each `Conv1d(1,1,5,bias=False)`, exactly 15 zero weights, and no other ECA module;
- exactly 1,073,977 parameters, all optimizer parameters present once in one unchanged SGD group;
- every shared parameter/buffer bitwise equal and post-construction CPU RNG state equal;
- all initial gates exactly one for arbitrary finite real and synthetic inputs;
- train-mode and eval-mode logits/output bitwise equal, BN counters/statistics bitwise equal, and all shared parameter gradients bitwise equal for hard and probability-target cross-entropy;
- finite, nonzero gradients for each of the three ECA kernels on both target paths;
- `layer3[0]` Option-A shortcut remains bitwise accepted and its padded-channel residual gradients remain live.

Any mismatch retires the candidate. Do not weaken equality, remove the transition block, or realign accepted weights post hoc.

## Real-Batch First-Update and CutMix Safety

Use materialized production N1/M7 batches, not Gaussian inputs; EXP-015 showed out-of-distribution concentration checks can be misleading. Run hard int64 and valid CutMix probability-target cases separately from identical accepted/candidate states.

For each target case:

1. Confirm exact pre-update logits/loss/shared gradients and finite nonzero attention gradients.
2. Apply one ordinary accepted SGD step.
3. Replay the same batch without stepping and capture each block's gate distribution, ECA weights, logits, loss, predictions, and shared gradients.
4. Backward a second distinct real batch to prove all ECA and residual parameters remain recruited.

Require:

- maximum absolute ECA kernel weight after step one at most 0.25;
- every replay gate in `[0.75,1.25]` and each module's mean gate in `[0.95,1.05]`;
- finite replay loss no more than 2x both pre-update candidate loss and aligned accepted replay loss;
- one-class prediction concentration below 95%, unless accepted control is at least as concentrated;
- finite FP32 parameters, gradients, BN buffers, and momentum buffers;
- nonzero second-batch gradient for all three ECA kernels and all six `layer3` residual convolutions;
- hard and CutMix cases independently pass; do not average them or let one compensate for the other.

Then run 200 distinct production strong batches with the normal approximately 50/50 hard/CutMix stream on paired accepted/candidate models. This length directly covers the kind of early divergence caught in EXP-016. Require candidate loss EMA no more than 1.5x control, candidate-only concentration never above 95%, all gates finite and within `[0.5,1.5]`, and all three ECA kernels still receive finite nonzero gradients. Record gate mean/p01/p99 separately for hard and soft batches to detect a CutMix-specific response, but do not tune from it.

The thresholds are catastrophic sentinels, not evidence that full-phase representation quality will improve. EXP-015 passed short checks and still underfit at 80%; only the full trajectory can establish the mechanism.

## Paired Timing, Backward, and Exposure Gate

Fifteen parameters do not imply free execution. Every selected block adds sequential global pooling, tiny Conv1d, sigmoid, multiply, and their backward kernels. Measure five alternating fresh-process accepted/candidate pairs on one idle H20. Each arm uses state-aligned models, 100 warm steps, then at least 500 synchronized batch-128 steps alternating hard/probability targets.

Time the exact production interval including pinned H2D transfer, zero-grad, forward, cross-entropy, backward, SGD, and final synchronize. Also record CUDA-event transfer, forward, loss, backward, and optimizer stages. Report each pair's mean, median, p95, images/s, CV, peak allocation, and stage ratios.

Advance only if every condition passes:

- candidate/control median synchronized training ratio at most `1.02x`, with no individual pair above `1.04x`;
- ratio-projected exposure at least **26,360 updates**, 98% of EXP-010's 26,898, or 3,374,080 presented images;
- candidate p95 at most 1.04x control p95, per-arm trial CV below 3%, and paired-ratio CV below 2%;
- measured extra time is localized to candidate model forward/backward rather than loader/target/timing mismatch; backward overhead is reported explicitly because it is 75.46% of accepted step time;
- peak allocation below 700 MiB with no monotonic growth.

Run five alternating FP32 inference pairs at evaluator batch 256, including the final partial batch. Require median inference ratio at most `1.05x` and CV below 2%. Project cold startup + 300 counted training seconds + switch + expected unchanged-cadence evaluations below 540 seconds.

If 2% training overhead, 98% exposure, or the wall projection fails, do not expand/contract block scope, fuse by hand, change kernels, or add a performance mechanism. The final-stage candidate is retired.

## Loader and Evaluation Fairness

Batch size and CPU transforms are unchanged, and a passing candidate is no faster than accepted, so loader headroom should remain. Still measure at least 1,000 real strong batches; require median iterator wait below 10% of candidate step time, p95 below 20%, correct hard/soft provenance, all eight strong workers stopped at the switch, and one integer-target weak batch after rebuild.

Keep the exact accepted evaluation policy and FP32 evaluator. Do not add gate-aware test passes or compare gated/ungated inference. Require at most one evaluation per epoch and unique evaluated epochs. Any slight reduction in epochs/test looks from overhead is part of the candidate and cannot be compensated.

## Predicted Mechanism and Risks

**Primary hypothesis:** three identity-scale final-stage ECA modules learn bounded example-dependent 128-channel residual recalibration, improve the high-level representation without suppressing early/transition learning, retain at least 98% exposure, and raise `best_test_acc` from 94.15% to at least 94.25%.

Pre-register 87.08% as the recurring strong-underfit marker and 89.0% as a healthier switch expectation. These remain diagnostics and cannot trigger a retry or configuration change. Compare switch accuracy, first weak checkpoint, tail slope, final/best gap, and final NLL to EXP-010 and EXP-017.

Risks:

- **CutMix descriptor ambiguity:** global pooling merges two class regions, so one channel gate may suppress evidence required by either probability target.
- **Initial identity is momentary:** EXP-014 proved a neutral new path can dominate after one step; `2*sigmoid` bounds the gate but does not bound learned logits or update speed.
- **Late-stage amplification:** the `(0,2)` range can inflate residuals, BN-following is absent after block addition, and repeated gates may alter calibration.
- **Channel adjacency is artificial:** nearby learned channels have no guaranteed semantic ordering, limiting what a length-5 convolution can express.
- **Narrow scope may be too weak:** three modules reduce cost and early interference but may not reach the ten-example formal threshold.
- **Sequential backward cost:** tiny operations can be launch-bound; the 75.46% backward bottleneck makes saved attention activations and kernels material despite negligible FLOPs.
- **Short checks do not protect the phase:** bounded first updates cannot rule out the long strong-phase suppression observed in EXP-015.
- **Single-seed threshold:** a 94.25% pass is formal but low-confidence causal evidence and must not be overstated.

## One-Run Verification

Only after every structural, safety, timing, exposure, loader, lifecycle, and wall gate passes:

1. Confirm the moving baseline is 94.15% at `7c1e7d8`; improvement requires 94.25%.
2. Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB and pin its visible index.
3. Verify only `train.py` differs. Run syntax/lint/pre-commit and inspect the exact final-stage-only, k=5, pre-add centered-gate diff with no production instrumentation.
4. Assert exactly 1,073,977 parameters, three zero five-weight ECA kernels in one accepted SGD group, shared state/RNG alignment, and unchanged width, batch, data, CutMix, optimizer, schedule, seed, timer, workers, and evaluator.
5. Remove stale `run.log`; launch exactly once as `uv run train.py > run.log 2>&1` under a 600-second supervisor. No valid-run retry or alternate ECA scope is allowed.
6. Require exit 0, approximately 300 counted seconds, total below 600, finite standard summary, exact parameter count, and no non-finite output.
7. Require one switch near 80%, eight stopped workers, realized CutMix near 50%, hard weak targets, and at most one evaluation on each unique epoch.
8. Require at least 26,360 actual updates for near-equal-exposure mechanism support. Record exposure ratio, strong/tail steps, epochs/evaluations, peak memory, startup, and total time.
9. Require `best_test_acc >=94.25%` for formal improvement. Report switch/first weak/best/final/NLL and full trajectory against EXP-010/017.
10. Remove `run.log` after analysis and restore accepted `7c1e7d8` on any no-go or no-improvement.

## Decision Rules

- **Preflight no-go:** any identity/RNG, hard/CutMix first-update, 200-batch stability, timing, 98%-exposure, loader, lifecycle, or wall failure blocks production.
- **Accept:** all integrity conditions pass, actual updates are at least 26,360, and accuracy is at least 94.25%.
- **Valid accuracy miss:** reject this exact three-block final-stage identity-scale ECA; do not retry all blocks, one block, another kernel, standard sigmoid, or another placement.
- **Accuracy pass below exposure floor:** report formal metric evidence, but the claimed near-equal-exposure recalibration mechanism is unsupported; mandatory analysis decides categorization without rerun.
- **Scope, runtime, seed, evaluator, non-finite, or safety violation:** invalid and revert with no rescue.

## Recommendation

Final-stage identity-scale ECA is a narrow response to the current evidence: it preserves accepted early and transition learning, targets semantic channel allocation rather than another fit-enhancing shortcut, and starts from exact accepted function/shared gradients. The scope reduces but does not eliminate CutMix ambiguity and sequential backward cost. Advance only if real hard/soft trajectories remain bounded and paired timing retains 98% exposure; otherwise retire the one fixed design.

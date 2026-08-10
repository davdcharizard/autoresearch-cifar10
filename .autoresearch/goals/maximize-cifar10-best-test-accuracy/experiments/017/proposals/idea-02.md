# Proposal 02: Identity-Scale ECA Residual Channel Attention

## Proposal

Add one Efficient Channel Attention (ECA) module to each of the nine accepted width-2 postactivation residual blocks. Each ECA module global-average-pools the block's second-BN residual output, applies one deterministic odd-width `Conv1d` across channels, forms an identity-centered channel gate, and multiplies only the residual path immediately before shortcut addition.

Keep the complete EXP-010 data, optimizer, schedule, timer, evaluator, seed, and lifecycle unchanged. This experiment has one kernel rule, one placement, and one initialization. Do not substitute paper-standard half-scale sigmoid initialization, a different kernel, selected stages, post-add attention, shortcut attention, SE, BF16, or a changed optimizer if any gate fails.

## Mechanism and Local Evidence

The accepted EXP-010 width-2 network reached 94.15% with active postactivation residual branches, 89.73% at the strong switch, 93.16% at the first weak checkpoint, and 26,898 updates. Its largest gains came from capacity under strong views and p=0.5 CutMix. ECA targets a remaining representation question: can the model allocate those channels conditionally across RandAugment and spatially mixed examples rather than treating every residual channel equally?

ECA-Net uses global average pooling plus a tiny channel-axis convolution to learn local cross-channel interactions without SE's dimensionality bottleneck. Its published ResNet evidence is ImageNet-scale, not this shallow CIFAR recipe, but the supplemental reports positive ResNet-18/34 results and the mechanism adds only a handful of parameters. Source: [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) and `papers/eca-net.md`.

Local failures dictate the initialization. Standard ECA applies `sigmoid(logit)`, whose zero-centered initialization produces gates near 0.5 and approximately halves every residual branch. EXP-012 full preactivation and EXP-015 selective zero-gamma already lowered strong fit by 2.85-3.25 points; repeating global residual attenuation is not defensible. EXP-014 further showed that nominal initial functional safety does not control the first learned update. This proposal therefore uses a single identity-scale formulation, `2 * sigmoid(logit)`, initialized with exact-zero ECA kernels and guarded on real hard/CutMix first updates.

## Exact Kernel Rule

Use the ECA paper's deterministic channel mapping with `gamma=2`, `b=1`, and the implementation-style odd rounding:

```python
t = int(abs((math.log2(channels) + 1) / 2))
kernel_size = t if t % 2 else t + 1
```

For the accepted stage widths, this fixes:

| Residual channels | ECA kernel | Blocks | Parameters |
| ---: | ---: | ---: | ---: |
| 32 | 3 | 3 | 9 |
| 64 | 3 | 3 | 9 |
| 128 | 5 | 3 | 15 |

The experiment adds exactly **33 scalar weights** and no biases, changing total parameters from 1,073,962 to **1,073,995**. Do not tune the 3/3/5 choice from timing or accuracy, force a shared kernel across stages, or omit transition blocks.

## Exact Module, Placement, and Initialization

Use one module per block:

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

The RNG fork prevents `Conv1d` constructor initialization from perturbing accepted shared-weight initialization or the post-construction CPU RNG; zeroing then fixes the attention state. Verify shared model state and final CPU RNG match an accepted control exactly after construction.

Place it literally in `BasicBlock.forward` after `bn2` and before shortcut addition:

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
out = self.eca(out)
# accepted Option-A shortcut construction remains unchanged
out += shortcut
return F.relu(out)
```

Apply this to all nine blocks, including the two stride-2 transition blocks. Attention sees only the residual output; it never gates the raw identity/Option-A shortcut, post-add tensor, stem, or classifier. At initialization, `channel_conv == 0`, so every logit is zero and every gate is exactly one. The accepted output, all shared parameter gradients, active transition channels, and residual branch strength are preserved before the first optimizer step. Unlike zero-gamma, ECA's own kernel has a nonzero derivative through `2*sigmoid` and can learn immediately.

The identity-centered gate ranges in `(0, 2)`, unlike paper-standard `(0, 1)`, so it can amplify as well as suppress residual channels. That modification is mechanistically necessary to preserve the validated initial network, but it creates a gate-growth risk controlled below. Do not clamp gates in production; a clamp would be another method.

## Unchanged Training Contract

Retain width 2, batch 128, standard SGD momentum 0.9, coupled all-parameter decay `1e-4`, LR 0.1 through 80%, step to 0.01 and cosine to `1e-4`, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, hard crop/flip weak tail, seed 42, evaluator, checkpoints, persistent workers, and 300-second counted timer. The 33 ECA weights join the same single optimizer group; no separate LR, decay exclusion, warmup, regularizer, precision mode, or schedule is added.

Evaluation uses the same model normally in FP32 through the untouched `Eval.evaluate`. Add no production gate logging or additional test call. All safety measurements belong to disposable preflight scripts.

## Structural and Exact-Identity Gate

Before GPU timing, compare accepted and ECA models built from seed 42:

- exactly nine ECA modules with kernels `3,3,3,3,3,3,5,5,5` in stage order and 33 zero weights;
- exactly 1,073,995 total parameters, with all 33 ECA weights in the accepted SGD group once and all shared state keys/shapes unchanged;
- bitwise equality of every shared parameter/buffer and equality of post-construction CPU RNG state;
- on fixed FP32 inputs in train and eval mode, all gates exactly one, logits/output bitwise equal, BN state updates bitwise equal, and every shared gradient bitwise equal for both hard and probability-target cross-entropy;
- finite, nonzero gradients for all nine ECA kernels on each target path;
- both Option-A padded channel halves remain live and receive residual-path gradients.

Any mismatch retires the candidate. Do not weaken equality to tolerance or remove a block to make it pass.

## Real-Batch First-Update and CutMix Safety Gate

Use materialized production-distribution N1/M7 batches; EXP-015 showed Gaussian safety inputs can yield false class-concentration signals. Build aligned accepted/ECA models and optimizers. Run separate hard-label and valid p=0.5 CutMix-probability cases from identical initial shared state.

For each target path:

1. Prove the pre-update forward/loss/shared gradients are exactly accepted and all ECA kernel gradients are finite/nonzero.
2. Apply exactly one ordinary accepted SGD step.
3. Replay the same batch without another update and inspect all nine gates, logits, loss, predictions, ECA weights, and shared gradients.
4. Run a second distinct real-batch backward to prove every ECA kernel and every residual convolution remains recruited.

Require all conditions:

- maximum absolute ECA kernel weight after step one is at most 0.25;
- for every block and sample, replay gates remain within `[0.75, 1.25]`, and each block's mean gate lies within `[0.95, 1.05]`;
- replay loss is finite and no more than `2x` both its own pre-update loss and the aligned accepted replay loss;
- one-class prediction concentration remains below 95%, unless the aligned accepted control is at least as concentrated;
- all ECA/shared gradients, parameters, BN buffers, and SGD momentum buffers remain finite;
- every ECA kernel has a nonzero second-batch gradient, and every block's `conv1`/`conv2` data gradients remain nonzero;
- the CutMix case independently passes the same gate limits; do not average hard and soft results or let one path compensate for the other.

Then run 64 distinct real strong batches with the normal approximately 50/50 hard/CutMix stream on paired accepted/ECA models. Require candidate terminal loss EMA no more than 1.5x control, class concentration below 95% unless control matches it, every gate finite and within `[0.5, 1.5]`, and all nine ECA kernels still receiving finite gradients. These deliberately loose sentinels catch EXP-014/016-style collapse; they do not predict full-phase accuracy. A failure cannot be rescued by clipping, smaller LR, standard sigmoid, fewer blocks, or another kernel.

## Paired Timing and Exposure Gate

ECA's nominal FLOPs and parameters are tiny, but each of nine modules adds sequential global-pool, small-Conv1d, sigmoid, and multiply work. EXP-012's all-block SE probe cost 1.233x despite modest arithmetic, so paired H20 timing is mandatory.

Run five alternating fresh-process accepted/ECA pairs on the sole idle H20. Each process uses a fresh state-aligned model/optimizer, 100 warm steps, then at least 500 synchronized batch-128 timed steps with alternating hard and probability targets. Measure the exact production interval including transfer, zero-grad, forward, loss, backward, SGD, and final synchronize. Report mean, median, p95, CV, images/s, peak allocation, plus CUDA-event forward, loss, backward, and optimizer stages.

Advance only if all pass:

- ECA/accepted median synchronized training time is at most `1.03x`, and no individual pair exceeds `1.05x`;
- ratio-projected exposure is at least **26,091 updates**, 97% of EXP-010's 26,898, or 3,339,648 presented images;
- ECA p95 step time is at most `1.05x` accepted p95, per-arm trial-mean CV is below 3%, and ratio CV is below 2%;
- the reported overhead is attributable to added model forward/backward stages rather than loader or measurement differences;
- peak allocation remains below 700 MiB with no monotonic growth.

Also run five alternating fresh-process FP32 inference pairs at evaluator batch 256, including the final partial batch. Require median ECA/control inference ratio at most `1.08x` and CV below 2%. Project cold startup + 300 training seconds + loader switch + the expected unchanged-cadence evaluations below 540 seconds, leaving 60 seconds before the absolute 600-second kill.

If the 3% training or 97% exposure gate fails, do not run production and do not test ECA on fewer blocks. Parameter/FLOP claims cannot override measured sequential-kernel cost.

## Loader and Evaluation Fairness

Batch size and data path are unchanged, and an ECA candidate passing the timing gate is slightly slower than accepted, so the measured 0.145 ms loader wait should retain headroom. Still exercise at least 1,000 real strong batches, require median iterator wait below 10% of candidate step time and p95 below 20%, and prove the exact eight-worker shutdown/rebuild followed by an integer-target weak batch.

Keep every accepted evaluation condition. Do not add an early attention diagnostic evaluation, evaluate an ungated control, or change tail checkpoints. Require at most one test pass per epoch and unique evaluated epochs. Any small epoch/evaluation-count decrease from ECA overhead is part of the fixed-time method; extra observations are forbidden.

## Hypothesis and Risks

**Primary hypothesis:** identity-scale ECA learns bounded content-dependent residual channel gates while retaining at least 97% of accepted updates and the healthy strong representation, raising `best_test_acc` from 94.15% to at least 94.25% under unchanged FP32 evaluation.

Pre-register 87.08% as the recurring strong-underfit marker and 89.0% as the healthier expectation. These are diagnostics only, not post-hoc stop/tuning rules. Compare switch accuracy, first weak checkpoint, tail slope, final/best gap, and final NLL against EXP-010.

Main risks:

- **Global descriptors conflict with CutMix.** One pooled channel gate summarizes two spatially mixed classes and may suppress evidence useful to either soft target component.
- **Identity does not guarantee safe learning.** EXP-014 proved an initially neutral path can dominate after one update. Centered ECA has bounded gate range but can still move too quickly or amplify residuals.
- **Strong-phase suppression can emerge late.** EXP-015 passed first-update and 64-step checks yet underfit by the 80% switch. Full-phase evidence remains decisive.
- **Sequential-kernel overhead.** Nine tiny attention chains can lose far more time than their 33 parameters/FLOPs imply, as the prior SE timing showed.
- **Modified ECA scaling.** `2*sigmoid` is not the paper's standard gate range. It preserves the accepted initial model but permits amplification up to 2 and weakens direct literature transfer.
- **Channel-order locality is imposed, not semantic.** Adjacent learned convolution channels have no guaranteed meaningful ordering, so a 3/5-wide channel kernel may be too restrictive.
- **Very shallow-network ceiling.** Published gains come from deeper ImageNet backbones; nine CIFAR blocks may not have enough redundant channels for a reliable 0.10-point gain.
- **Single-seed threshold is small.** A bare 94.25% pass is ten test examples and should be reported as formal but low-confidence evidence, not a precise causal effect.

## One-Run Verification

If and only if structural, real-batch safety, timing, exposure, loader, lifecycle, and wall gates pass:

1. Confirm moving baseline 94.15% at `7c1e7d8`; formal improvement requires 94.25%.
2. Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB and pin the sole visible index.
3. Verify only `train.py` differs. Run syntax/lint/pre-commit and review the exact nine-module, 3/3/5, centered-gate placement with no production diagnostics.
4. Assert 1,073,995 parameters, nine zero ECA kernels / 33 scalars, accepted shared state/RNG alignment, one SGD group, and unchanged width, batch, data, CutMix, schedule, seed, evaluator, timer, and workers.
5. Remove stale `run.log`; launch once as `uv run train.py > run.log 2>&1` under a 600-second supervisor. No valid-run retry or alternate ECA configuration is allowed.
6. Require exit 0, approximately 300 counted seconds, total below 600, finite standard summary, 1,073,995 parameters, and no non-finite training output.
7. Require one strong-to-weak switch near 80%, all eight old workers stopped, realized CutMix near 50%, no probability target in the weak tail, and at most one evaluation per unique epoch.
8. Require at least 26,091 actual updates to support the exposure mechanism. Record actual ratio, epochs/evaluations, strong/tail updates, peak memory, startup, and total wall time.
9. Require `best_test_acc >=94.25%` for improvement. Report switch, first weak, best/final, NLL, and the whole evaluation trajectory relative to EXP-010.
10. Remove `run.log` after analysis and restore accepted `7c1e7d8` on every no-go or no-improvement outcome.

## Decision Rules

- **Preflight no-go:** any identity/RNG, first-update, hard/CutMix, recruitment, timing, 97%-exposure, loader, lifecycle, or wall gate failure blocks production.
- **Accept:** all integrity/mechanism conditions pass, actual steps are at least 26,091, and accuracy is at least 94.25%.
- **Accuracy miss after valid run:** reject this exact all-block identity-scale ECA. Do not retry standard sigmoid, another kernel, another placement, or selected stages.
- **Accuracy pass below exposure floor:** report formal metric evidence, but the predeclared representation-under-near-equal-exposure claim is unsupported; mandatory analysis decides categorization without rerun.
- **Scope, runtime, seed, evaluator, non-finite, or safety violation:** invalid and revert with no fallback.

## Recommendation

Identity-scale ECA is a focused representation candidate with an explicit answer to the recurring strong-underfit failure: it leaves every accepted residual branch exactly active at initialization, then learns only 33 local channel-interaction weights. Its paper support and tiny parameter count justify development, but neither protects against CutMix-global-gate mismatch or H20 launch cost. Advance only if both real hard/soft first updates stay bounded and paired timing retains 97% exposure; otherwise retire the single fixed design without substitution.

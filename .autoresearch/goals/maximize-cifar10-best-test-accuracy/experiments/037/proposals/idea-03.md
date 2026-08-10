# Proposal: Mean-Centered Stem Convolution

## Decision and falsifiable hypothesis

Replace only the image-facing `ResNet.conv1` with a differentiable mean-centered convolution: on every forward, subtract each output filter's mean over `(in_channel, kernel_h, kernel_w)` before the unchanged convolution. Do **not** divide by the filter standard deviation, change the stored parameter, or apply the operation to any of the 18 residual convolutions. Preserve the accepted Kaiming initialization, all BatchNorm modules, postactivation graph, width/depth, ordinary momentum SGD, all-parameter `1e-4` coupled decay, N1/M7 plus p=0.5 alpha-1 CutMix strong phase, hard weak tail, elapsed-time LR schedule, evaluator, timer, seed, batch size, precision, and workers.

**Falsifiable hypothesis:** a zero-DC first-layer filter basis will suppress crop/illumination common mode while retaining edge, color-contrast, and CutMix-boundary evidence; differentiable centering will also project the stem data gradient onto the same zero-mean subspace. Because the projection is non-expansive and affects one 864-weight tensor without variance rescaling, it should avoid the amplified relative-step geometry of EXP034 and preserve at least 99% of accepted exposure. The prediction is seed-42 `best_test_acc >= 94.25%` from the 94.15% moving baseline, with strong-to-weak switch accuracy at least 89.0% and final NLL no worse than the accepted 0.1934. One valid sub-threshold run rejects this exact stem-only point; a safety or timing veto blocks production without scope, scale, or phase rescue.

## Literature and local evidence

Qiao et al., *Micro-Batch Training with Batch-Channel Normalization and Weight Standardization* ([arXiv:1903.10520](https://arxiv.org/abs/1903.10520)), reparameterize each convolution output filter as `(W - mean(W)) / std(W)` and argue that the induced gradient projections smooth the loss and gradient landscape. Their ImageNet ablation is particularly relevant: with ResNet-50 + GroupNorm, mean subtraction alone improved top-1 error from 24.81 to 23.96, whereas standard-deviation division alone reached only 24.60 and full WS reached 23.72. They also report BN+WS improving ImageNet error from 24.30 to 23.76, so WS is not inherently redundant next to BN. This is directional evidence, not a local effect-size guarantee: their strongest motivation is micro-batch training, their networks are much deeper, and their CIFAR table does not isolate BN+WS.

Brock, De, and Smith, *Characterizing signal propagation to close the performance gap in unnormalized ResNets* ([ICLR 2021](https://openreview.net/forum?id=IX3Nnir2omJ), [arXiv:2101.08692](https://arxiv.org/abs/2101.08692)), identify weight centering as a way to prevent channel-mean growth in normalization-free ResNets. The accepted network retains BN, so that result supplies a representation mechanism rather than direct recipe evidence.

The local history sharply narrows the canonical method:

- EXP029's all-Conv data-gradient centralization was trajectory-safe and removed 37-87% of Conv gradient norm, which confirms that mean directions are locally nontrivial. It was not scored because 19 reductions plus 19 subtractions cost 1.97% of fixed-budget exposure. The present candidate is not a retry of that implementation: it changes the stem's forward representation continuously and centralizes its gradient by reparameterization, while adding operations at only one site.
- EXP034's Conv fan-out initialization kept initial BN-normalized logits within 0.044% but reduced stem parameter scale to 0.306x, producing a 13.99% relative stem update and six candidate-only one-class transients. Mean-only centering has no division and leaves the stored Kaiming parameter norm and optimizer LR unchanged; its Jacobian is an orthogonal projection, so it cannot enlarge the instantaneous stem data-gradient norm.
- EXP036 shows that changing crop boundaries can amplify strong-view logits even without class collapse. This candidate preserves constant crop padding and instead removes only the uniform local component seen by the first learned filters.

## Why stem-only mean centering

For one stem filter with `I=3*3*3=27` coefficients, define

```text
P(W) = W - mean(W),     sum(P(W)) = 0.
```

The effective filter is invariant to adding the same scalar to every coefficient and is orthogonal to a locally uniform scalar shift shared by RGB channels, apart from zero-padding boundary effects. That is a meaningful image-space prior. The corresponding operation in later layers would remove an equal shift across dozens of learned channels, where channel order and common mode have no equally clear semantic interpretation. Restricting scope also avoids 18 additional reduction/subtraction sites on a 10.9 ms step.

The intervention is intrinsically bounded. `P` is an orthogonal projection, so `||P(W)|| <= ||W||` and the backpropagated raw data gradient is `P(g)`, with `||P(g)|| <= ||g||`. Under seed-42 Kaiming normal initialization the expected removed squared-weight fraction is only `1/I = 3.70%` for the stem; unlike full WS, there is no `1/std` multiplier and no 0.306x fan-out shrink. BatchNorm immediately after the stem should further damp global output-scale differences, although it does not make the functions exact because local image means, padding, BN epsilon, running statistics, and subsequent SGD differ.

Coupled decay deserves explicit treatment. SGD updates the stored raw `W` using `P(g) + lambda*W`; its constant component receives decay but no data gradient and is invisible to the forward. Momentum therefore accumulates an ordinary projected data term plus a decaying null-space term. Do not project the parameter after `optimizer.step`, exclude it from decay, rescale LR, or rewrite momentum: those would test different optimizer rules. Report both raw and effective centered norms so a harmless null-space component cannot distort safety interpretation.

## Exact implementation

Add one subclass and use it only for `ResNet.conv1`:

```python
class MeanCenteredConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight - self.weight.mean(dim=(1, 2, 3), keepdim=True)
        return self._conv_forward(x, weight, self.bias)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, width_multiplier=1):
        super().__init__()
        c1, c2, c3 = (width_multiplier * channels for channels in (16, 32, 64))
        self.conv1 = MeanCenteredConv2d(
            3, c1, 3, stride=1, padding=1, bias=False
        )
```

`MeanCenteredConv2d` remains an `nn.Conv2d`, so the accepted `_weights_init`, module traversal, state-dict key, constructor RNG draws, parameter count (1,073,962), and optimizer membership remain unchanged. `_conv_forward` preserves the installed Conv2d padding/stride/dilation/groups semantics. Do not add an epsilon, learned gain, standard-deviation division, weight hook, parametrization API, custom autograd, cache, in-place parameter mutation, or production diagnostics.

## Construction and mechanism checks

Before trajectory or timing work, independently instantiate accepted and candidate models from identical CPU/CUDA RNG states and require:

1. identical post-construction CPU/CUDA RNG, parameter/buffer keys, shapes, values, module order, 19 Conv/19 BN/one Linear, 1,073,962 parameters, and optimizer membership;
2. candidate class scope exactly one stem module, with all residual convolutions ordinary `nn.Conv2d`;
3. an FP64 oracle match for candidate stem outputs and gradients against explicit `F.conv2d(x, W-W.mean((1,2,3)))` on varied synthetic tensors;
4. effective per-filter means at most `1e-7`, raw/effective norm and removed-energy reports, and `||P(W)|| <= ||W||` for all 32 stem filters;
5. unchanged stored parameters and RNG after forward/backward, zero-mean candidate stem data gradients within tolerance, and candidate raw data-gradient norm no larger than the gradient with respect to the effective weight;
6. exact four-step momentum/decay recurrence demonstrating `P(g)+lambda*W`, including the raw/effective/null-space decomposition.

On one registered real hard and one real CutMix batch, hook pre-BN stem output, post-BN activation, logits, loss, and pooled features. Require finite values, candidate/control loss ratio in `[0.8,1.2]`, post-BN RMS ratio in `[0.8,1.2]`, logit RMS ratio below 2.0, and no candidate-only greater-than-95% class concentration. Record pre-BN changes rather than forcing near identity; removing image DC response is the intended representation change.

## Control-qualified immutable-corpus safety

Reuse without regeneration the registered EXP022 200-batch strong corpus and EXP028 64-batch weak corpus after checking their file/tensor hashes, schemas, hard/soft counts, and immutability. First run two accepted control/control pairs under the production-default backend and calculate denominator-safe absolute-plus-relative limits for logit RMS, whole gradient/update norms, stem effective-weight update, and class share. Controls must pass the frozen global catastrophic ceilings and establish candidate-specific authority before candidate replay; a noisy or zero denominator cannot veto a candidate.

Then replay accepted and candidate models over byte-identical corpora with ordinary SGD and the registered strong/weak LRs. Serialize full evidence before assertions. Production requires finite parameters, buffers, logits, losses, gradients, momentum, and diagnostics; exact BN counters; no candidate-only predicted-class share above 95%; candidate/control logit, whole-gradient, and whole-update ratios below the greater of 5x or the qualified control bound; candidate whole update below 25% of raw parameter norm; no update above 5x its preceding 16-step median; stem effective update below 25% of effective stem norm; and terminal strong/weak loss-EMA ratios at most 1.5. Report stem raw/effective/null means and norms, projected-gradient fraction, BN running variance, and hard/soft behavior. Lower short loss cannot waive a geometry veto, and a safe short replay is not accuracy evidence.

## Fixed-budget timing and runtime feasibility

The candidate adds one tiny reduction and subtraction before every stem convolution, plus their autograd path and one 864-element temporary. The measured backward bottleneck remains the unchanged convolution/BN stack; peak memory should stay near 598.7 MiB. Nevertheless, EXP029 proves launch-sized helpers are not free.

Use one conditioning process and seven counterbalanced fresh-process accepted/candidate pairs on one idle H20, with production eight-worker loaders, real strong-hard/strong-soft/weak-hard paths, 100 warmups, and at least 1,000 synchronized measured steps per arm. The candidate arm must call the production subclass. Save raw trials before assertions. Require aggregate candidate/control counted-step ratio at most `1.01`, every pair at most `1.04`, per-arm CV below 3%, projected exposure at least 26,629 steps, peak allocation below 650 MiB, no loader starvation/growth, integrated wall/count at most 1.07, and projected total runtime below 540 seconds. This 99%-exposure gate is credible for one site; failure retires the literal implementation and does not authorize all-layer, cached, fused, phase-only, or initialization-only alternatives.

Initialization and construction are outside the counted timer but the differentiable centering is inside every training step. Do not move the work outside the timer or cache a stale effective weight.

## Production integrity and verdict

Immediately before the single scored run, query the moving baseline (expected 94.15 at `7c1e7d8`); require only tracked `train.py` to differ, no stale `run.log`, passing compile/Ruff/format/scope checks, and exactly one idle H20 of approximately 97,871 MiB. Verify seed 42, transforms, CutMix collator, loader lifecycle, schedule, timer, evaluator, precision, and summary code are unchanged. Run once:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero; one finite ten-field summary; 300.0-301.0 counted seconds; total below 600 seconds; 1,073,962 parameters; one transition near 80% with eight strong workers stopped; hard weak targets; first weak LR near 0.01; 45-55% CutMix among strong batches; at most one evaluation per epoch; and no more than the accepted 19 looks. Require at least 26,629 steps absent documented system-wide contention. Diagnostics explain but never change the formal metric verdict.

- **Improvement:** every integrity condition passes and `best_test_acc >= moving_baseline + 0.10` (currently 94.25%).
- **No improvement:** the valid run finishes below that threshold, regardless of switch fit, NLL, or lower loss.
- **Invalid/crash:** construction, mechanism, control qualification, corpus safety, timing, scope, hardware, evaluator, timer, summary, or wall-limit failure.

Main risks are a null effect because BN already removes much common mode; loss of useful low-frequency/color evidence in the stem; strong-view divergence around CutMix and padded borders; and launch overhead above 1%. The literature supports the projection mechanism but not this stem-only, width-2, short-horizon operating point. No rescue is allowed by adding variance normalization, extending to other layers, weakening centering, changing decay/LR, selecting a phase, or rerunning seed/corpora.

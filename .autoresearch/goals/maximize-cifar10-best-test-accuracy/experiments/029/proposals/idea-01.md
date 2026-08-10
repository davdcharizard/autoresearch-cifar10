# Proposal: Conv2d-Weight-Only Data-Gradient Centralization

## Decision and falsifiable hypothesis

After each `loss.backward()` and immediately before the accepted `optimizer.step()`, project every `nn.Conv2d.weight` data gradient onto the per-output-filter zero-mean subspace. Preserve the accepted PyTorch momentum-SGD implementation and its all-parameter coupled weight decay, so the projection acts on the loss gradient only and decay is added afterward by SGD. Do not alter the forward graph, initialization, classifier, BN affine parameters, data/target curriculum, learning-rate schedule, evaluation cadence, timer, precision, seed, or loader lifecycle.

**Hypothesis:** removing common-mode loss-gradient drift from each convolution filter will regularize the width-2 ResNet-20 high-LR trajectory without the alternating history or parameter/state misalignment that caused EXP020/022/028 class collapse. It will preserve strong-phase fit and at least **26,629 optimizer steps**, and raise seed-42 `best_test_acc` from the 94.15% moving baseline to at least the formal **94.25%** improvement threshold. The point prediction is **94.30%** at about 26.7k steps. A protocol-valid result below 94.25% falsifies the accuracy claim; a preregistered safety or timing veto makes this exact operating point invalid rather than authorizing a modified retry.

## Evidence, local fit, and scope

Yong et al., *Gradient Centralization* (ECCV 2020, [primary paper](https://arxiv.org/abs/2004.01461); local distillation: `knowledge/papers/gradient-centralization.md`), define `P(g) = g - mean(g)` per weight vector and interpret it as projected gradient descent. They report improved vision optimization/generalization without a second forward/backward pass and find convolution-only GC sufficient in a low-resolution CIFAR-100 setting. Their evidence is not direct proof here: it uses longer training, different networks/datasets, generally `5e-4` decay, and often Conv+FC centralization rather than this 300-second CIFAR-10/CutMix recipe.

The local motivation is unusually narrow. EXP010 established the 94.15% width-2/CutMix frontier; EXP005/027 require retaining the full strong curriculum through the simultaneous 80% LR/data transition; EXP008/009 require preserving all-parameter coupled decay `1e-4`. EXP020/022/028 show that finite state, first-step matching, and coherent-signal scale do not make a global-LR optimizer intervention safe. GC is still worth testing because it preserves ordinary SGD state and cannot increase an individual Conv data-gradient Frobenius norm, but it must pass the same immutable-batch class-geometry and update-spike protections. EXP003 also shows that a nominally cheap regularizer can cost meaningful fixed-budget exposure, so the 19 reductions and 19 subtractions require fresh paired timing.

This proposal centralizes **only the 19 bias-free Conv2d weight tensors**: the stem plus two convolutions in each of nine residual blocks. The 2-D classifier gradient is excluded to avoid directly altering class competition, and all 1-D BN/bias gradients are excluded. Eligibility is by `isinstance(module, nn.Conv2d)`, never a broad dimensionality rule.

## Exact production implementation

Add this single helper to tracked `train.py`:

```python
@torch.no_grad()
def centralize_conv_weight_gradients(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            grad = module.weight.grad
            if grad is None:
                raise RuntimeError("missing Conv2d weight gradient")
            grad.sub_(grad.mean(dim=(1, 2, 3), keepdim=True))
```

Call it exactly once in every successful training iteration:

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = F.cross_entropy(outputs, targets)
loss.backward()
centralize_conv_weight_gradients(model)
optimizer.step()
```

For a Conv weight `[C_out, C_in, kH, kW]`, reducing `(1, 2, 3)` independently centers each output filter and returns `[C_out, 1, 1, 1]`. Do not reduce dimension 0, center the entire tensor, center per input channel, or center only spatial dimensions. Mutate the existing FP32 `.grad` in place under `torch.no_grad()`; do not clone/replace gradients, add hooks, add an optimizer subclass, change gradient accumulation cadence, or apply GC during evaluation. Production adds no parameter, optimizer buffer/group, RNG draw, hyperparameter, or per-step synchronizing diagnostic.

The accepted `optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)` construction remains byte-for-byte unchanged. Model parameter count must remain 1,073,962, and the model's initialized parameters and first forward outputs must match the accepted source exactly before any backward call.

## Decay and momentum ordering

Let `g_t` be the raw loss gradient of a Conv weight, `P` the filterwise projection, `lambda=1e-4`, `mu=0.9`, and `lr_t` the accepted elapsed-time LR. The helper first overwrites `.grad` with `P(g_t)`. Installed PyTorch SGD then performs:

```text
d_t = P(g_t) + lambda * w_t
b_1 = d_1
b_t = mu * b_(t-1) + d_t,  t > 1
w_(t+1) = w_t - lr_t * b_t
```

For every non-Conv parameter, the accepted recurrence remains `d_t = g_t + lambda*w_t`. `dampening=0`, `nesterov=False`, and PyTorch's undamped first momentum-buffer initialization remain unchanged.

This order is scientifically load-bearing. It is **not** `P(g_t + lambda*w_t)`: accepted coupled decay acts on the complete Conv weight after the data-gradient projection, including the filter-mean component. Therefore the experiment does not claim the paper's strict invariant-mean theorem; it tests the narrower mechanism that zeroes common-mode *data-gradient* drift while preserving the locally validated decay rule. Do not disable optimizer decay, manually add/decouple it, project the decay term, or project an existing momentum buffer.

## Static and algebraic gates

Before a model trajectory, an ignored controller must establish all of the following and serialize the evidence before assertions:

- Discover exactly 19 unique eligible Conv weights and no classifier, BN affine, bias, parameter duplicate, or missing gradient.
- On deterministic CPU FP32 fixtures with deliberately nonzero per-filter offsets, match an out-of-place FP64 projection reference within `1e-7` absolute / `1e-6` relative tolerance; prove projection idempotence and non-increasing data-gradient norm.
- Show every post-GC filter mean is bounded by `max(1e-7, 1e-6 * raw_gradient_rms)`, while all FC and BN gradients are bitwise unchanged.
- Show parameters, buffers, and CPU/CUDA RNG states do not change between the end of backward and the call to SGD.
- On a manual four-or-more-step changing-gradient sequence including an LR change and nonzero filter means, match every candidate parameter and momentum buffer to the exact recurrence above within `1e-7` absolute / `1e-6` relative tolerance. Independently show the control matches installed PyTorch SGD.
- Confirm the accepted and candidate initial `state_dict`, first logits, optimizer group structure, and BN counters are exact matches before candidate projection is applied.

Any discrepancy is a candidate/controller stop before corpus work. Fixing an independently demonstrated controller bug is allowed; changing the candidate semantics is not.

## Immutable strong-and-weak corpus gates

Reuse both existing immutable corpora; do not materialize another stream:

- **Strong:** `experiments/022/preflight-corpus.pt`, 200 exact post-policy N1/M7 batches (94 hard, 106 CutMix), file SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`, registered tensor digest `4242043f3a4cbc04c3de0c2ffbc9f78c5c01c8314ae695c2ba8e94a3e992ad40`.
- **Weak:** `experiments/028/weak-corpus.pt`, 64 exact crop/flip hard-label batches, file SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`, registered tensor digest `df97b02a24ff5f4ca17fe0697c31b2da1cbdb6e0b3c9ca14107cfe2444408eae`.

The weak corpus is valid for reuse despite EXP028's PNM veto: EXP028 persisted it before either aligned arm, recorded 64 hard batches, and reported both corpora unchanged after all 264 steps. Before use, recompute both file hashes and tensor digests, validate payload schema/count/shape/dtype/target rank, and compare them to the prior reports. Never regenerate, reorder, filter, subsample, overwrite, or mutate either file.

Start accepted and candidate models from the same seed-42 initial state with deterministic backend settings and identical ordinary SGD optimizers. Clone each corpus record into each arm so in-place device operations cannot alias source storage. Run all 200 strong batches at fixed LR 0.1, then all 64 weak hard batches at the registered low LR used by the controller. The two arms must receive byte-identical inputs and targets in identical order. The controller must not call the evaluator.

Record per step and phase: both losses and class histograms; raw and projected Conv data-gradient norms; maximum residual filter mean; removed common-mode norm by stage; effective post-decay direction, momentum, full update, and parameter norms; Conv filter means; BN counters/running-state validity; target rank; and finite state. Hash the accepted initial state, corpus tensors before/after each arm, and generated report. Fsync the report before evaluating vetoes.

Required gates are:

- all logits, losses, gradients, parameters, BN state, momentum state, and diagnostics finite; BN counters exact and optimizer state complete;
- every centralized filter mean within the registered tolerance, every Conv projected norm no greater than its raw norm beyond FP32 tolerance, and FC/BN gradients unchanged by the helper;
- no step where candidate maximum predicted-class share is above 95% while control is at or below 95%; lower loss does not waive this gate;
- terminal debiased candidate/control loss-EMA ratio at most 1.5 in each of the strong and weak phases;
- no candidate/control whole-model update-norm ratio above 2.0 and no candidate Conv update above 25% of that tensor's pre-update parameter norm;
- no candidate whole-model update above 5 times its own preceding 16-step median once that history exists;
- exact unchanged corpus and accepted-start hashes after both arms.

A gate failure retires full-strength, all-Conv, pre-decay GC for EXP029. It does not authorize excluding a layer, applying a coefficient, delaying or stopping GC, changing its dimensions/order, clipping, warming up, changing LR/decay, regenerating a corpus, or relaxing a threshold.

## Paired timing and exposure gates

GC adds 38 small GPU operations inside the counted interval. The model is small and its accepted synchronized step is about 10.9 ms, so paper-level claims of negligible overhead on larger networks are insufficient. After algebra and corpus safety pass, confirm one idle NVIDIA H20 with about 97,871 MiB and no competing process, then run one unscored conditioning process followed by **five fresh-process counterbalanced accepted/candidate pairs**. Alternate order across pairs and never reuse a trained process for the other arm.

Each arm must use the real eight-worker production strong loader, accepted batch 128, identical backend settings, at least 100 unmeasured warmups, and at least 1,000 synchronized measured production-order steps. The timed region includes nonblocking transfer, forward, cross-entropy, backward, GC only for the candidate, ordinary SGD, and `torch.cuda.synchronize()`. Also measure a registered weak hard-label segment and the strong-loader shutdown/weak-loader rebuild lifecycle. Do not use synthetic-only timing, move GC outside the timer, add candidate-only warmup, compile, enable autocast/channels-last, or retain per-step safety scans in the production timing path.

Pass only if:

- aggregate candidate/control counted-step ratio is at most **1.01**, every pair is at most **1.04**, and each arm's trial-mean CV is below 3%;
- `floor(26,898 * control_mean / candidate_mean) >= 26,629` projected production steps;
- warmed loader delivery is at least 1.2 times GPU consumption, with median iterator wait below 10% and p95 below 20% of candidate step time;
- peak allocated memory is below 650 MiB with no monotonic allocation or child-process growth;
- weak-loader rebuild is below five seconds, integrated wall/count ratio is at most 1.07, and projected total runtime is below 540 seconds;
- exactly eight strong workers stop before the first hard weak batch and zero child workers remain after final shutdown.

The aggregate fresh-pair ratio is authoritative; the historical 26,898-step projection is a consistency/exposure gate, not permission to rerun. A timing miss retires this implementation and does not authorize a custom/fused kernel, layer subset, foreach rewrite, or looser threshold within EXP029.

## Production verification and interpretation

Only after every prior gate passes, reconfirm: baseline 94.15 at `7c1e7d8`; the integration source is clean except ignored artifacts and the user's untracked `data/`; only tracked `train.py` differs; no stale `run.log` or renamed run-log exists; one idle H20 is visible; model/data/evaluator/seed/schedule/precision/lifecycle contracts are unchanged; and syntax, formatting/lint, source diff, and helper-call-count checks pass.

Run exactly one scored seed-42 production attempt:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero; one finite expected ten-field summary; `training_seconds` approximately 300.0-301.0; `total_seconds <600`; `num_params = 1,073,962`; `num_steps >=26,629`; peak VRAM below the timed bound; one switch near 80% with exactly eight workers stopped; approximately half of strong batches CutMix; exclusively hard targets after the switch; no duplicate/repeated-epoch evaluation; and no extra candidate-only evaluation. Hash and retain the implementation, preflight/timing reports, and production log until analysis is complete.

Report the 80% switch and first-weak accuracy against EXP010's 89.73% and 93.16%, best/final accuracy, final NLL against 0.1934, epochs/steps, strong/weak loss behavior, raw-to-projected gradient ratios and removed-mean norms by stage, filter-mean drift, update/momentum diagnostics, CutMix counts, evaluation count, VRAM, counted time, and wall time. Diagnostics explain the mechanism but cannot override the formal verdict:

- `best_test_acc >=94.25%` is a formal improvement; 94.25-94.35 remains positive but weak single-seed evidence.
- Better NLL with `best_test_acc <94.25%` is calibration-only no-improvement.
- Lower switch fit followed by a miss means GC removed useful high-LR filter drift.
- Similar switch fit plus improved weak-tail NLL/top-1 supports the proposed generalization mechanism.
- Early candidate-only concentration or an update veto means even a norm-nonincreasing instantaneous projection changed momentum-integrated class geometry unsafely.
- A complete valid result below the exposure expectation remains the one fixed-budget result; it is reported with the confound and is never rerun.

## Risks and no-rescue boundary

- **Scientific risk — medium-high:** direct evidence is CIFAR-100/long-horizon/different-model evidence, and this decay-preserving variant is narrower than the paper's projected-decay formulation.
- **Optimization risk — medium:** instantaneous Conv gradient norm cannot increase, but ordinary momentum integrates a different trajectory; three recent optimizer-path experiments warn that output geometry can still collapse.
- **Runtime risk — medium:** 19 tiny reductions plus 19 broadcast subtractions can be expensive relative to this unusually short FP32 step.
- **Attribution risk — low:** forward/model/data/schedule and ordinary SGD state semantics remain fixed; explicit decay ordering isolates data-gradient projection.
- **Implementation risk — low-medium:** the helper is small, but eligible tensors, dimensions, call placement, and decay ordering are load-bearing.
- **Estimated effort — medium:** production code is minimal; immutable-corpus safety and fresh paired timing dominate.

There is **no rescue tuning** in EXP029. Do not change GC strength, dimensions, layer coverage, phase coverage, call cadence, LR, momentum, decay/order, clipping, warmup, batch size, data policy, precision/layout, timing kernel, seed, corpora, or gates after observing preflight, timing, or production evidence. Do not rerun a valid completion. Any such variant is a new brainstormed experiment with a new ID.

# Proposal: Conv2d-Weight-Only Gradient Centralization

## Decision and falsifiable hypothesis

After every `loss.backward()` and before the accepted `optimizer.step()`, subtract each `Conv2d.weight` gradient's per-output-filter mean over input-channel and spatial dimensions. Apply no transformation to BN affine gradients, the classifier, biases, parameters, activations, targets, or optimizer state. Retain ordinary PyTorch SGD, momentum 0.9, all-parameter coupled weight decay `1e-4`, the elapsed-time LR schedule, model, data curriculum, evaluator, and lifecycle unchanged.

**Hypothesis:** removing common-mode data-gradient drift within each convolution filter will regularize the accepted width-2 ResNet-20 and smooth its short high-LR trajectory without reducing strong-phase fit, raising `best_test_acc` from 94.15% to at least **94.25%** while preserving at least **26,629** optimizer steps. Point prediction: **94.30%** and about 26,700 steps. A complete lower result falsifies the accuracy claim; a safety or timing veto makes this exact operating point invalid.

## Evidence and scope

Yong et al., *Gradient Centralization* (ECCV 2020, [primary paper](https://arxiv.org/abs/2004.01461)), define the projection `P(g) = g - mean(g)` per weight vector and interpret it as constrained projected gradient descent. They report improved optimization/generalization across vision models and note that convolution-only GC was sufficient on low-resolution CIFAR-100. Their CIFAR evidence used longer 200-epoch training, several larger networks, and generally weight decay `5e-4`; it does not establish benefit for CIFAR-10, ResNet-20 width 2, CutMix/RandAugment, or this 300-second horizon. Their default experiments also centralize both Conv and FC unless stated otherwise.

This proposal deliberately selects **Conv2d weights only**. The classifier's 2-D gradient is excluded to avoid directly changing class competition, and 1-D BN/bias gradients are excluded because they do not represent convolution filters. The forward graph and initial function remain bitwise accepted, unlike recent architecture interventions.

Local evidence cuts both ways. EXP-003 showed that a seemingly cheap regularizer can lose 6.7% fixed-budget exposure, so the extra reductions require paired timing. EXP-020 Nesterov and EXP-022 Lookahead produced candidate-only class transients despite finite state and lower loss; GC therefore needs immutable-corpus recurrence and concentration gates even though it adds no optimizer buffer. EXP-008/009 showed that changing all-parameter `1e-4` decay semantics hurts this model, motivating the explicit decay ordering below. EXP-027 further protects the full accepted strong curriculum through the 80% switch.

## Exact implementation and dimensions

Add one function to `train.py`:

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

Call it exactly once per successful training step:

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = F.cross_entropy(outputs, targets)
loss.backward()
centralize_conv_weight_gradients(model)
optimizer.step()
```

PyTorch stores `Conv2d.weight` as `[C_out, C_in, kH, kW]`. Therefore `(1, 2, 3)` centralizes each of the `C_out` filters independently and leaves a mean tensor `[C_out,1,1,1]`. Do not centralize over dimension 0, over the whole tensor, per input channel, or only spatial dimensions. The accepted model has exactly 19 eligible bias-free convolution weights: the stem and two convolutions in each of nine blocks. Eligibility must be module-type based, not `grad.ndim > 1`, because the latter would incorrectly include `fc.weight`.

Use in-place FP32 mutation under `torch.no_grad()` after backward. Do not clone or replace `.grad`, register backward hooks, add an optimizer subclass, use a private foreach reduction, centralize accumulated gradients at another cadence, or run GC during evaluation. No new hyperparameter, optimizer group, parameter, buffer, RNG draw, or logit path is introduced.

## Coupled weight-decay and momentum ordering

The exact candidate recurrence is pre-registered. Let `g_t` be the raw loss gradient of a Conv weight, `P` the filterwise projection above, `lambda=1e-4`, and `mu=0.9`. The explicit GC call writes `P(g_t)` to `.grad`. Unchanged PyTorch SGD then performs coupled decay and momentum:

```text
d_t = P(g_t) + lambda * w_t
b_1 = d_1
b_t = mu * b_(t-1) + d_t,  t > 1
w_(t+1) = w_t - lr_t * b_t
```

For every non-Conv parameter, `d_t = g_t + lambda*w_t`, exactly accepted. The first momentum buffer is PyTorch's undamped first direction; `dampening=0` and `nesterov=False` remain unchanged.

This ordering is intentional and must not be blurred with the paper's stated weight-decay variant `P(g_t + lambda*w_t)`. Centralizing before an unchanged PyTorch `optimizer.step()` yields `P(g_t) + lambda*w_t`: the data gradient has zero filter mean, while accepted coupled decay still acts on the full Conv weight, including its mean component. Consequently, the paper's strict constant-filter-mean theorem does not hold here; means can shrink through weight decay. This is a locally motivated isolation of data-gradient GC that preserves the validated all-parameter decay rule. Do not set optimizer decay to zero, manually add decay, centralize `g+lambda*w`, or project the momentum buffer.

## Static, algebraic, and immutable-corpus safety

Before GPU trajectory work, use analytic CPU fixtures with nonzero per-filter offsets to prove all 19 Conv gradients have near-zero means after GC, the result equals an out-of-place FP64 reference within FP32 tolerance, and FC/BN gradients are bitwise unchanged. Verify projection idempotence, non-increasing Conv data-gradient norm, unchanged parameters/RNG before `optimizer.step`, and no missing/duplicate module. On a manual four-step changing-gradient sequence with LR change and coupled decay, compare every candidate parameter and momentum buffer to the recurrence above; separately prove control still matches installed PyTorch SGD.

Reuse the immutable EXP-022 post-policy corpus rather than generating a favorable stream: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt`, SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`, containing 200 exact N1/M7 batches (94 hard, 106 CutMix). Recompute the file and tensor digests before use, clone identical records into explicit accepted/candidate processes, use the same seed-42 initial state and deterministic backend, and never regenerate, reorder, filter, or mutate it. Serialize and fsync all evidence before evaluating veto assertions.

Train both arms for all 200 fixed-LR `0.1` steps. Require exact starting parameters, buffers, RNG declarations, optimizer structure, and batch/target hashes. Record every loss, class histogram, raw/projected Conv gradient norm, removed-mean norm, effective post-decay direction, update norm, momentum norm, parameter/filter means, BN counter, and finite-state check. The candidate must satisfy:

- all 19 post-GC data-gradient filter means are within `max(1e-7, 1e-6 * raw_gradient_rms)` at every registered check;
- FC and BN gradients remain bitwise equal to their pre-GC values; GC is RNG-neutral and changes no parameter before SGD;
- manual and installed candidate recurrence agree within `1e-7` absolute / `1e-6` relative tolerance;
- every parameter, buffer, gradient, momentum tensor, logit, loss, and diagnostic remains finite, with exact BN counters and complete momentum state;
- no step has candidate maximum predicted-class share above 95% while control is at or below 95%; lower candidate loss cannot clear this gate;
- candidate terminal debiased loss EMA is at most 1.5x control, no candidate/control full-update norm spike exceeds 2x, and no Conv update exceeds 25% of its pre-update parameter norm;
- corpus and accepted initial-state hashes remain unchanged after both arms.

Also replay registered weak hard batches if available to ensure the projection remains finite across the target-form/augmentation boundary. The controller calls no evaluator. A failure does not authorize centralizing fewer layers, delaying GC, changing decay ordering, clipping, warm-up, or relaxing the concentration rule.

## Timing and exposure

GC adds 19 small reductions and 19 broadcast in-place subtractions inside counted time. The paper reports negligible relative overhead on ResNet-50, but this workload's accepted synchronized step is only about 10.9 ms and optimizer/reset is 1.67%; kernel-launch overhead can matter disproportionately. Architecture and activation memory are unchanged, so the expected memory increment is only reduction temporaries.

After confirming one idle 97,871 MiB H20, run one unscored conditioner and five alternating fresh-process accepted/candidate timing pairs. Each arm uses the real eight-worker strong loader, identical backend state, at least 100 warmups and 1,000 synchronized production-order steps including transfer, forward, loss, backward, GC where applicable, SGD, and synchronize. Then measure a weak segment, shutdown/rebuild lifecycle, and allocation stability. Do not amortize GC outside the timer, use synthetic-only batches, compile, enable autocast/channels-last, or give candidate-only warmup.

Require aggregate candidate/control counted-step ratio `<=1.01`, every pair `<=1.04`, per-arm trial-mean CV `<3%`, and `floor(26898 * control_mean / candidate_mean) >=26,629`. Require warmed loader delivery at least 1.2x GPU consumption, median iterator wait below 10% and p95 below 20% of candidate step time, peak allocation below 650 MiB, no worker/allocation growth, weak rebuild below five seconds, integrated wall/count `<=1.07`, and projected total runtime below 540 seconds. Exactly eight strong workers must stop before a hard weak batch, followed by zero live children. A timing miss retires this implementation; it does not authorize a fused/custom kernel or fewer centralized layers inside EXP-028.

## Mechanism and discriminators

The expected mechanism is removal of each output filter's common-mode loss-gradient component. This constrains data-driven filter-mean drift, reduces raw update norm along that direction, and may smooth high-LR optimization while preserving the accepted forward representation and full target curriculum. Because coupled decay remains unprojected, evidence supports this narrower mechanism, not the paper's exact invariant-weight-subspace theorem.

Interpret outcomes prospectively:

- An early candidate-only concentration or update spike means even a norm-nonincreasing projection changed class geometry unsafely under accepted momentum.
- Lower loss but no top-1 gain echoes EXP-020/022: optimization smoothness is not sufficient generalization evidence.
- Better NLL but `best_test_acc <94.25` is calibration-only no-improvement, as EXP-003 warned.
- Healthy safety but lower switch fit indicates GC removes useful high-LR filter drift and conflicts with the short strong phase.
- Similar switch fit, improved weak-tail NLL/top-1, at least 26,629 steps, and `>=94.25` supports Conv data-gradient centralization as the mechanism. A 94.25-94.35 single-seed result remains formally positive but noise-consistent weak evidence.
- A result below the exposure floor is attribution-confounded even if accuracy clears the metric; it must not be rerun.

Report switch and first-weak accuracy, best/final accuracy and NLL, epochs/steps, raw-to-projected gradient ratios by stage and phase, filter-mean drift, update/momentum norms, CutMix counts, evaluation count, VRAM, and wall time. Diagnostics cannot change the formal verdict.

## Production, abort gates, and risks

Only after every gate passes, reconfirm baseline 94.15 at `7c1e7d8`, only `train.py` tracked-modified, no stale run log, one idle H20, syntax/Ruff/format/diff checks, and exact unchanged model/optimizer/data/schedule contracts. Run seed 42 exactly once as `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Require exit zero, ten finite summary fields, approximately 300 counted seconds, total below 600, 1,073,962 parameters, at least 26,629 steps, one 80% switch with eight workers stopped, valid CutMix rate, hard weak targets, and no repeated evaluation epoch. Accept only `best_test_acc >=94.25`; never rerun a valid completion.

Abort for any tracked edit outside `train.py`; any forward/model/data/evaluator/seed/schedule/precision/lifecycle change; centralizing FC/BN/bias or the wrong Conv dimensions; applying GC after decay/momentum; changing accepted weight decay; missing/duplicate GC calls; source/RNG/corpus mismatch; recurrence, finite-state, concentration, loss, update, timing, exposure, memory, wall, or worker gate failure; or GPU contention. No tail-only switch, layer subset, strength coefficient, clipping, warmup, alternate seed, paper-order decay, or threshold rescue is allowed.

- **Scientific risk — medium-high:** the paper's strongest low-resolution evidence is CIFAR-100 with longer training and different models/decay; this local variant also preserves decay outside the projection.
- **Optimization risk — medium:** projection cannot increase raw Conv gradient norm, but momentum integrates a directionally different trajectory and local optimizer changes have produced class transients.
- **Runtime risk — medium:** 38 tiny GPU operations can exceed the paper's relative-overhead claim on this unusually small, fast network.
- **Implementation risk — low-medium:** the code is small, but dimension choice and coupled-decay ordering are scientifically load-bearing.
- **Estimated effort — medium:** production is minimal; recurrence, immutable-corpus safety, and paired timing carry most of the work.

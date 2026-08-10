# Proposal: Full-Run Eligible-Weight Gradient Centralization

## Summary

Starting from EXP-002, apply coefficient-free Gradient Centralization (GC) once
after every loss backward and immediately before the existing PyTorch Nesterov
momentum update. Materialize the parent's coupled L2 term on every gradient
first. For each `nn.Conv2d.weight`, subtract the mean of each output filter's
regularized direction over its input-channel and spatial axes. For
`nn.Linear.weight`, subtract each output row's mean over its input-feature axis.
Compute the 17 row-mean tensors separately, then apply their broadcasts with
one `torch._foreach_sub_` call. Leave BatchNorm affine parameters and every
bias uncentralized.

This is an isolated optimizer-geometry experiment. It adds no model parameter,
activation, target change, forward pass, persistent training state, stochastic
draw, or tunable coefficient. Preserve EXP-002's WRN-16-4, front-loaded CutMix,
drop path, time-cosine learning rate, BF16/channels-last path, fixed seed 42,
data stream, evaluator, and 300-second charged budget exactly. All reduction and
audit work occurs inside the charged step. Every GPU command exposes physical
GPU 0 only through `CUDA_VISIBLE_DEVICES=0`.

## Motivation and Evidence

EXP-002 raised accuracy from 94.62% to 95.23% with early probabilistic CutMix,
but its final 95.19% and final CE loss 0.2044 leave stable-generalization
headroom. A narrow CutMix/drop-path sweep from the same node did not confirm its
selected improvements (EXP-003), so this proposal changes optimization geometry
rather than retuning the validated regularization package. EXP-004 later showed
that a gradient-geometry intervention can help this lineage, but periodic SAM
cost a second forward and reduced exposure from 27,950 to 25,560 steps. GC is an
earlier-branch, one-backward alternative and does not alter SAM because EXP-002
contains no SAM path.

The ECCV 2020 Gradient Centralization paper interprets per-output zero-mean
weight gradients as projected optimization and reports smoother optimization
and generalization gains across vision tasks. Its official CIFAR `SGD_GC`
implementation applies coupled weight decay before GC and momentum; EXP017
matches that ordering. The experiment and promoted
distillations are at `experiments/017/papers/gradient-centralization.md` and
`knowledge/papers/gradient-centralization.md`; the primary source is
<https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2471_ECCV_2020_paper.php>.
The official optimizer source is
<https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/SGD.py>.
The evidence establishes plausibility and a standard axis convention, not a
guaranteed effect under this shallow pre-activation WRN, BatchNorm, CutMix,
drop-path, Nesterov, and wall-clock schedule.

The intervention targets the goal's compute constraint directly: it uses the
existing backward signal and performs only linear gradient reductions and
in-place subtraction. It should retain close to EXP-002's useful data exposure
if H20 kernel-launch and reduction overhead are small. Unlike adding a second
objective or trajectory estimator, it has no coefficient or state horizon to
tune against the test metric.

## Fixed Mechanism

### Eligibility and exact axes

Build the eligible list once after model construction by iterating
`model.named_modules()` and selecting the `weight` object of every
`nn.Conv2d` and `nn.Linear`. Do not select parameters merely by a broad name
substring, and do not centralize any tensor with fewer than two dimensions.
For the frozen EXP-002 model, assert the inventory before training:

- 16 convolutional weight tensors plus one classifier weight tensor;
- 17 eligible tensors and 2,745,264 eligible elements;
- 2,266 output rows/filters centralized per optimizer step;
- all 13 BatchNorm weight/bias pairs and the 10-element classifier bias are
  excluded.

For an eligible gradient after coupled L2 has been materialized as
`d = g + WEIGHT_DECAY * parameter`:

```python
reduce_dims = tuple(range(1, d.ndim))
row_mean = d.mean(dim=reduce_dims, keepdim=True)
# After collecting every eligible d and row_mean:
torch._foreach_sub_(eligible_directions, row_means)
```

Thus a convolutional gradient with shape `[C_out, C_in, K_h, K_w]` is centered
independently for each of its `C_out` filters over `(C_in, K_h, K_w)`. The
classifier gradient `[10, 256]` is centered independently for each class row
over its 256 input features. The output dimension is never mixed across
filters/classes. Gradients and parameters are FP32 even though the forward uses
BF16 autocast; the parent has no `GradScaler`, `torch.compile`, or CUDA graph
capture. Do not cast them down, flatten across output rows, normalize their
magnitude, clip them, or introduce an epsilon.

Use `torch.no_grad()` and in-place subtraction on `parameter.grad`; never use
`.data`. Every eligible parameter must have a non-`None` gradient on every
call; a missing gradient is an integrity failure rather than permission to skip
a tensor. Check every value for finiteness in the integration/preflight traces
and on the fixed sparse production audit steps. Do not add 17 extra finiteness
reductions or host synchronizations to every metric step merely for auditing.
The helper is called exactly once on every optimizer step, including early
clean, early CutMix, and late clean steps.

### Order with CutMix and Nesterov

Keep the parent's loss construction unchanged. A CutMix batch computes its one
area-weighted CE scalar and performs one backward; GC transforms the gradient of
that combined scalar, not two constituent gradients independently. A clean
batch similarly uses the unchanged hard-label CE. Integration is exactly:

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast(...):
    outputs = model(inputs, drop_scale=current_drop_scale)
    loss = parent_cross_entropy(outputs, targets_a, targets_b, adjusted_lam)
loss.backward()
# Reproduce coupled L2 outside SGD, then centralize eligible directions.
torch._foreach_add_(all_gradients, all_parameters, alpha=WEIGHT_DECAY)
centralize_eligible_weight_directions(eligible_weights, audit=...)
optimizer.step()
```

The optimizer is the existing PyTorch `optim.SGD` with the same learning rate,
momentum, dampening, Nesterov setting, parameter order, and state initialization,
but with its internal `weight_decay=0` because the identical coupled-L2 term is
materialized immediately before the step. For each parameter, the intended
order is:

1. form the raw data-loss gradient `g` from hard CE or weighted CutMix CE;
2. for every parameter, form the regularized direction
   `d = g + WEIGHT_DECAY * parameter` using one `torch._foreach_add_`;
3. for eligible weights only, form `d_gc = d - row_mean(d)`;
4. update the momentum buffer with `d_gc` for eligible tensors and `d` for
   excluded tensors, apply the existing Nesterov look-ahead,
   and update the parameter with the current time-scheduled learning rate.

This matches the official `SGD_GC` ordering and makes the eligible direction
entering momentum zero-row-mean. The pre-add replaces, rather than duplicates,
PyTorch SGD's coupled decay. Deterministic smokes must prove that moving decay
outside SGD leaves every excluded parameter and momentum buffer equal to the
parent reference, and that eligible updates match an explicit
`decay -> centralize -> momentum -> Nesterov` reference. Do not replace or
subclass SGD, alter `momentum=0.9`, effective `weight_decay=1e-4`,
`nesterov=True`, learning-rate scheduling, zeroing order, or optimizer state
initialization.

### Parent-preservation contract

Do not change the architecture, initialization order, parameter count, input
normalization, crop/flip, loader, batch size, CutMix constants or dedicated RNG
streams, drop-path schedule/global CUDA RNG, BF16 autocast, channels-last
layout, synchronization/timing boundary, epoch definition, or evaluation.
GC performs no RNG operation, so identical parent and candidate work through
`loss.backward()` must have identical logits, loss, raw gradients, model
buffers, and CPU/CUDA RNG states. Divergence begins only when the candidate
centralizes eligible gradients before the optimizer step.

All GC operations remain after the parent's `t0` and before its post-optimizer
`torch.cuda.synchronize()`, making their cost part of `training_seconds`. No
`torch.compile`, custom extension, new dependency, or phase gating is allowed.
`torch._foreach_sub_` is the fixed implementation, not an approximation:
deterministic CPU/GPU smokes must prove bitwise equality to 17 independent
broadcast `sub_` calls before timing. If the 17 reductions plus one foreach
subtraction are too slow, the experiment is rejected at preflight; do not
silently centralize fewer layers or only the final quarter.

## Mechanism Hypothesis and Counter-Hypothesis

**Hypothesis.** Removing each output unit's mean regularized direction across
its incoming weights will improve conditioning and suppress correlated filter
drift throughout both mixed and clean training. Because it preserves the
gradient's within-filter contrasts, all model/data exposure, and the validated
CutMix dose, it should improve the late solution rather than trade accuracy for
fewer optimizer steps. The testable formal prediction is
`best_test_acc >= 95.33%`, at least +0.10 points over EXP-002's 95.23%, while
retaining at least 27,000 steps.

**Counter-hypothesis.** Pre-activation BatchNorm already removes much of the
activation-scale/mean pathology that GC addresses, and Nesterov momentum plus
weight decay may dominate the small projected component. The removed mean can
also carry useful coordinated evidence, especially for the 10 classifier rows;
discarding it may slow fitting or interact poorly with area-soft CutMix
gradients. Finally, 17 reductions and broadcasts every step may consume enough
of the wall-clock budget to erase any generalization benefit. Under this view,
removed-component audits will prove the code active but accuracy will remain
below 95.33% or the preflight will reject the method for exposure loss.

## Instrumentation and Removed-Component Audits

Add lightweight integer counters on the host and sparse, fixed-cadence numeric
audits. Counters do not synchronize CUDA. Set `GC_AUDIT_EVERY = 512` before any
measurement; audit the first optimizer step and every 512th one-based step.
Do not change this cadence after seeing overhead or metrics.

For every audited tensor, reuse the already computed pre-subtraction
`row_mean`. Accumulate GPU-side FP64 scalars inside the charged timer until the
terminal summary:

- raw regularized-direction squared norm `sum(d^2)` before subtraction;
- removed-component squared norm
  `sum(row_mean^2) * product(d.shape[1:])`;
- centralized-gradient squared norm after subtraction;
- maximum absolute post-transform row mean;
- count of nonfinite values before and after transformation.

These establish the orthogonal decomposition
`||d||^2 ~= ||d_gc||^2 + ||broadcast(row_mean)||^2` without retaining full
gradient copies. Check the identity tightly in deterministic smokes and report
its aggregate relative error in the run. Synchronize/read audit scalars only
outside the charged loop after training; no per-step `.item()`, norm print, or
host branch may be added.

Print a terminal GC audit line that records:

- `calls`, eligible tensor count, eligible element count, and output-row count;
- total tensor transforms (`17 * calls`), element transforms
  (`2,745,264 * calls`), and centralized rows (`2,266 * calls`);
- GC steps split into early CutMix, early clean, and late clean paths;
- number of sparse audit samples;
- aggregate regularized/removed/centralized L2 values and removed/regularized norm ratio,
  decomposition relative error, and maximum post-transform row-mean residual;
- separate convolution-only and classifier-only regularized/removed energy and
  removed/regularized ratios;
- excluded tensor count and nonfinite count.

At exit require `gc_calls == num_steps`, `gc_cutmix_steps ==
cutmix_applied_batches`, `gc_early_clean_steps == cutmix_eligible_batches -
cutmix_applied_batches`, `gc_late_clean_steps == num_steps -
cutmix_eligible_batches`, and the three GC path counts to sum to `num_steps`.
Require the removed norm to be finite and strictly positive, FP64 decomposition
relative error at most `1e-5`, post-transform maximum absolute row mean at most
`1e-6`, and zero nonfinite values on sparse audit steps. Preserve and report the parent's CutMix
counts independently. These audits demonstrate mechanism and dose; they do not
authorize an early or metric abort based on a finite removed/regularized ratio.
For interpretation after the sole run, an aggregate removed/regularized ratio at or below
1% supports the BN-redundancy counter-hypothesis; between 1% and 5% is an
ambiguous moderate intervention; at or above 5% it establishes that GC removed
substantial signal, so a null implicates useful coordinated directions instead.
The separately reported classifier ratio determines
whether a follow-up conv-only child is scientifically motivated.

## Accuracy-Blind GPU-0 Preflight

Run one decisive preflight before the metric launch. First verify physical GPU
0 is the approximately 97,871 MiB NVIDIA H20, then set
`CUDA_VISIBLE_DEVICES=0` and require exactly one visible CUDA device with the
same UUID. Materialize the exact EXP-002 parent `train.py` from its recorded
commit in `/tmp`; do not modify another tracked file. Import each module without
calling `main`, immediately replace its evaluator with a guard that raises, and
install test-loader iteration counters before running either trace; those
counters must remain zero. Preflight may inspect finite training loss and
gradients but must never calculate or expose validation/test accuracy.

### Mathematical and integration checks

1. In FP64 on CPU and FP32 on GPU, test representative 4-D convolution and 2-D
   linear directions after coupled L2. Require reconstruction of the regularized direction from the
   centralized plus broadcast-mean components, per-output post means near zero,
   orthogonality, unchanged shape/stride/dtype/device, no mutation of
   explicitly excluded 1-D tensors, and bitwise equality between the fixed
   foreach subtraction and 17 independent broadcast-subtraction references.
2. Construct the full BF16/channels-last WRN and assert the exact 17-tensor,
   2,745,264-element, 2,266-row inventory. Check finite/nonzero gradients and
   the `1e-5` decomposition / `1e-6` residual bounds for one clean and one
   forced CutMix backward.
3. From equal model/optimizer copies and shared real CIFAR batches, replay the
   same global CUDA state. Require parent/candidate logits, losses, raw
   gradients, BN buffers, CutMix choices/geometry/permutations, and CPU/CUDA RNG
   states to match through backward. After the candidate materializes decay,
   require its excluded directions to equal the parent's explicit pre-momentum
   regularized reference and only its eligible directions to differ by GC.
4. Compare two candidate updates, including a nonempty momentum-buffer case,
   against an explicit FP64/FP32 Nesterov reference in the exact order
   `add coupled decay -> centralize eligible directions -> momentum -> Nesterov`.
   Require optimizer parameters and buffers to match within dtype-appropriate
   tolerances. Separately require excluded parameter/buffer parity against the
   parent's internal-decay SGD. Confirm one optimizer/BN update and no new
   optimizer state.
5. Run at least 1,024 consecutive production-order candidate steps covering
   early clean, forced CutMix, and late clean/drop-path-decay settings. Require
   all counter identities, finite losses/gradients/state, positive removed norm,
   unchanged parameter/state inventory, RNG neutrality of GC itself, and no
   CUDA allocation growth after warmup other than normal optimizer state.

### Parent-relative charged-latency gate

After warmup, run five fixed alternating-order paired rounds on shared,
materialized real-CIFAR batches. Each arm uses the actual model, BF16
channels-last forward/backward, parent-equivalent Nesterov optimizer, production
synchronization, and no evaluator. Each round contains exactly 44 early CutMix,
45 early clean, and 31 late clean steps, approximating EXP-002's observed
10,257/10,411/7,282 path counts without metric access. Candidate rounds add
full-run GC and its fixed sparse audit cadence; parent rounds do not. Do not
drop or replace a completed numeric round.

From the first complete five-round vector, proceed only if:

- all structural/math/RNG/Nesterov checks pass and test-loader iterations are
  zero;
- `(max(parent_round)-min(parent_round))/median(parent_round) <= 0.04`;
- paired ratio MAD divided by its median is at most `0.015`;
- median candidate/parent charged latency is at most `1.03`, and every paired
  round ratio is at most `1.06`;
- `27,950 / median_ratio >= 27,000` projected optimizer steps and
  `floor(projected_steps / 195) >= 138` projected epochs;
- projected total runtime is below 600 seconds. Candidate peak allocation,
  including optimizer and audit state, is reported; only OOM or insufficient
  headroom is a memory failure.

The dispersion bounds are deliberately looser than EXP-016's over-sensitive
0.5% ratio-dispersion gate while still rejecting an unstable timing harness.
The first complete numeric gate is decisive. Retry only for an exception,
malformed/missing output, or demonstrably invalid assertion before a full
numeric vector exists. A numeric failure rejects this proposal before metric
evaluation; do not optimize the helper, reduce eligibility/dose/audits, or
rerun the gate.

## Sole Metric Run and Thresholds

After all preflight gates pass, remove stale `run.log`, reconfirm physical GPU
0 and single-device visibility, and launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

Do not stop, select, or retune based on finite training loss or intermediate
test accuracy. Abort only for wrong hardware, exception, CUDA/OOM, nonfinite or
audit failure, process/liveness failure, or the 600-second cap. Do not retry the
metric with another seed, GC scope, phase, or implementation.

Apply these preregistered outcomes:

- **Formal local improvement:** `best_test_acc >= 95.33%`, at least +0.10 over
  parent EXP-002 at 95.23%, with all integrity conditions satisfied.
- **Noise-limited mechanism context:** `best_test_acc >= 95.53%`, a +0.30-point
  gain with final-16 context, is only a weak single-seed reason to consider a
  follow-up. A 95.33-95.52 result retains its tree verdict with still weaker
  causal evidence.
- **Local no-improvement:** a trustworthy result below 95.33%. There is no
  coefficient or fallback variant in this experiment.
- **Global context:** 95.61% matches the current goal-wide best; 95.71% exceeds
  it by the goal's 0.10-point resolution. A 95.33-95.60 result is a valid local
  tree improvement but not a new global best.
- **Stable mechanism support:** in addition to formal improvement, final test
  accuracy is at least 95.29% (0.10 above the parent's 95.19 final), the final
  CE loss does not exceed the parent's 0.2044, and the final-16 evaluation mean
  is reported with its range and best-minus-tail premium. Failure of this
  secondary bar does not override the formal tree verdict.
- **Composability:** a formal pass that also reaches 95.53 may motivate a future
  GC-on-EXP-011 test toward 95.71, but one seed cannot license it mechanically;
  a narrow local pass does not by itself justify stacking GC with SAM and EMA.

Integrity requires exit 0, `training_seconds` in `[299.5, 301.0]`,
`total_seconds < 600`, exactly one evaluator call/evaluation line per epoch,
the complete frozen summary, `num_params = 2,748,890`, fixed seed 42, intact
CutMix/drop-path configuration, only tracked `train.py` changed, and every GC
inventory/path/removed-component audit reconciled. A realized step count below
27,000 falsifies the throughput projection and weakens the mechanism claim but
is not permission to rerun; report it alongside the formal accuracy outcome.

Before protocol cleanup, durably transcribe the full summary, all epoch-tail
accuracies, CutMix/GC counts, removed-component statistics, preflight round
vector, projected/realized exposure, GPU identity, evaluation count, and scope
evidence into the execution record for Claude-only adversarial result review.

## Risks and Interpretation Limits

- The paper's broad vision evidence may not transfer to a shallow pre-activation
  network with BatchNorm on nearly every convolutional input. GC may be largely
  redundant with the parent's normalization and Nesterov dynamics.
- EXP017 matches the official coupled-L2 ordering, so the regularized eligible
  direction entering momentum is centralized. Momentum/Nesterov can still make
  the final applied update differ from an instantaneous projected-gradient
  interpretation; report this honestly in any causal claim.
- The classifier's row mean may be useful, and CutMix's soft-target gradient can
  respond differently from hard CE. Path-split counters show dose but do not
  identify which path caused an outcome.
- Nominal arithmetic is small, but 17 mean/subtraction pairs per step can be
  launch-bound. The accuracy-blind gate protects useful exposure without
  assuming that linear FLOPs imply zero overhead.
- The formal +0.10 threshold is below previously observed 0.14-0.29-point
  selected-run variability. One fixed-seed run establishes a protocol-valid
  local tree result, not broad statistical superiority; final/tail and CE-loss
  context must accompany the selected maximum.
- A null result rejects this exact full-run, all-eligible composition on
  EXP-002. It does not prove all GC variants ineffective and does not authorize
  a same-experiment conv-only, late-only, raw-gradient-only, or GC+EMA retry.

## Verification Checklist

1. Confirm the exact EXP-002 parent, 95.23 metric, frozen configuration, clean
   tracked scope, and physical/visible GPU-0 identity.
2. Prove exact axes, reconstruction, orthogonality, residual, eligible/excluded
   inventory, and FP32 in-place semantics.
3. Prove unchanged CutMix/drop-path/RNG/BN behavior through backward, excluded
   update parity, and exact decay-before-centralization Nesterov state updates.
4. Pass the single accuracy-blind five-round latency/exposure/memory gate with
   evaluator guards and zero test iterations.
5. Run exactly one fixed-seed 300-second metric process on GPU 0.
6. Reconcile GC calls/path dose/removed component with steps and CutMix counts,
   enforce once-per-epoch evaluation and runtime/scope constraints, then apply
   the 95.33 formal threshold without retuning.

# Proposal: Training-Only Fourth-Block Companion Classifier

## Summary

Add one lightweight companion classifier to the unchanged EXP-002
PreAct WRN-16-4. Tap the output of the fourth residual block (zero-based block
index 3), where the activation has shape `[B, 128, 16, 16]`; apply ReLU, global
average pooling, and one `Linear(128, 10)` layer. During the first 75% of
charged training, optimize its cross-entropy jointly with the inherited final
classifier. The companion receives exactly the same hard or CutMix-weighted
targets as the final classifier. Its coefficient is fixed at 0.15 through 50%
progress and then linearly tapers to zero at 75%. The head is never executed or
consulted by the frozen evaluator.

This is one preregistered intermediate-supervision test, not a coefficient,
attachment-point, normalization, or head-capacity sweep. Preserve EXP-002's
2-2-2 block allocation, final classifier, front-loaded probability-0.5 CutMix,
drop path, Nesterov SGD, cosine schedule, BF16/channels-last execution, seed 42,
300-second charged budget, and once-per-epoch evaluator. Every GPU command must
set `CUDA_VISIBLE_DEVICES=0` and verify that the one visible device is physical
GPU 0, the approximately 98 GB NVIDIA H20.

## Fixed Architecture and Schedule

Use these constants without adjustment after any preflight measurement:

```python
COMPANION_BLOCK_INDEX = 3
COMPANION_CHANNELS = 128
COMPANION_WEIGHT = 0.15
COMPANION_FULL_END = 0.50
COMPANION_END = 0.75
COMPANION_INIT_SEED = 42017
```

The exact coefficient schedule, evaluated from the inherited charged-time
`progress` at step entry, is:

```python
def companion_weight(progress):
    if progress < COMPANION_FULL_END:
        return COMPANION_WEIGHT
    if progress < COMPANION_END:
        fraction = (COMPANION_END - progress) / (
            COMPANION_END - COMPANION_FULL_END
        )
        return COMPANION_WEIGHT * fraction
    return 0.0
```

Thus the full-dose region is `[0, 0.50)`, the linear release is
`[0.50, 0.75)`, and the inherited clean final quarter is exactly companion-free.
The time-integrated coefficient is 0.09375 over an ideal continuous run. The
0.15 peak is deliberately moderate because this six-block network is not
demonstrably gradient-starved; the taper follows the Deeply-Supervised Nets
principle that companion influence need not persist at convergence while
aligning the complete turn-off with EXP-002's validated CutMix boundary.

Attach after the fourth residual block, not after stage 1 and not inside the
final 256-channel stage. This is the end of the 128-channel middle stage: it
provides a direct gradient through the stem and first four blocks while leaving
the last two blocks and final classifier to learn the main objective. Define:

```python
companion_logits = companion_fc(
    F.adaptive_avg_pool2d(F.relu(block4_output), 1).flatten(1)
)
```

Do not add companion BatchNorm, convolution, dropout, temperature, or hidden
MLP. The head adds exactly `128 * 10 + 10 = 1,290` trainable parameters, taking
the model from 2,748,890 to 2,750,180 parameters, a 0.047% increase. It adds one
pool, one very small matrix multiply, and one additional loss branch during
the first 75% of charged time. Its arithmetic cost is tiny relative to the
backbone, but launch and backward overhead must be measured rather than inferred
from parameter or MAC counts.

## Target and Loss Semantics

Factor the inherited main loss into one helper and call that helper separately
for the final and companion logits. For a clean batch:

```text
L_main = CE(main_logits, targets_a)
L_aux  = CE(companion_logits, targets_a)
L      = L_main + w(progress) * L_aux
```

For a CutMix batch, reuse the exact `targets_a`, `targets_b`, and clipped-area
`adjusted_lam` already returned by EXP-002's unchanged helper:

```text
L_main = adjusted_lam * CE(main_logits, targets_a)
       + (1 - adjusted_lam) * CE(main_logits, targets_b)

L_aux  = adjusted_lam * CE(companion_logits, targets_a)
       + (1 - adjusted_lam) * CE(companion_logits, targets_b)

L      = L_main + w(progress) * L_aux
```

Do not detach the tapped representation, redraw a permutation or lambda,
convert CutMix batches to hard labels, apply label smoothing, or rescale either
head according to observed losses or gradient norms. The area-weighted target
is only an approximation to the class content seen by a receptive field, but
using anything different for the companion would confound intermediate
supervision with a new target policy. Preserve the parent's dedicated CutMix
CPU and CUDA generators and consume no new stochastic draws during training.

Use the inherited optimizer group and weight decay for the companion weights
and bias; do not introduce a separate learning rate or no-decay exception.
When `w(progress) == 0`, request only the main logits so the pool, head, and
auxiliary CE do not execute in the final quarter. With gradients set to `None`,
SGD then leaves the inactive head unchanged, including coupled weight decay.

## Parent Initialization and Evaluator Isolation

Adding a module must not perturb the parent's main-model initialization or its
subsequent global RNG stream. Initialize all inherited modules in their
unchanged order first. Then construct and initialize the companion inside an
isolated CPU RNG context, using a dedicated CPU generator seeded with 42017
for Kaiming-normal weight initialization and a zero bias. Restore the enclosing
CPU RNG state exactly. Do not call a global `manual_seed` for the head and do
not let its construction advance the parent RNG.

Keep the production `forward(inputs, drop_scale=0.0)` contract unchanged: it
returns only the final logits tensor. A training-only explicit flag or method
may return `(main_logits, companion_logits)` when requested. The default path
must skip every companion operation. The frozen `Eval.evaluate(model, device)`
therefore receives the same `[B, 10]` main-logit tensor as EXP-002, performs the
same clean cross-entropy, and remains the sole source of `best_test_acc`,
`final_test_acc`, and `final_test_loss`. Never evaluate companion logits,
average the heads, select between them, or run a second evaluator call.

## Mechanistic Rationale

Deeply-Supervised Nets (`experiments/017/papers/deeply-supervised-nets.md` and
`knowledge/papers/deeply-supervised-nets.md`) reports that companion hidden-layer
classification objectives can make intermediate features more discriminative,
improve gradient delivery, and improve CIFAR classification without retaining
the auxiliary head at inference. EXP-002 already has a strong input-space
regularizer, but its main loss reaches the first four blocks only through two
additional high-level residual blocks. A direct middle-stage objective may
make those shared features linearly class-informative earlier, improving the
main classifier's final basin without another backbone forward.

The proposal preserves the successful mechanisms rather than substituting for
them. That matters because goal memory says substitution experiments identify
only the difference between removed and added mechanisms. Unlike EXP-009, this
head is one pooled path rather than four FP32 pool/standardize/MLP/sigmoid
paths; nevertheless, EXP-009's 20.7% latency failure makes a parent-relative
timing gate mandatory. Unlike EXP-010, it retains both 64-channel blocks and
the validated 2-2-2 allocation. EXP-010's 9.3% exposure gain without an
accuracy gain also warns that throughput alone is not the objective.

The counter-hypothesis is substantial. The parent already trains only six
residual blocks, BatchNorm and residual connections already deliver gradients,
and forcing linearly separable 128-channel features may remove transformations
that the final two blocks would otherwise exploit. The extra loss can also
over-regularize the same first 75% in which CutMix and drop path are active.
Consequently this is a plausible exploratory mechanism, not a high-confidence
route past the 95.61 global best.

## Required Correctness and Production Audits

Before timing, run deterministic CPU FP32 checks that establish:

- a parent and candidate constructed from the same seed have bitwise-equal
  inherited parameters/buffers and identical default-forward main logits;
- head construction leaves the global CPU RNG state unchanged and the head is
  absent from every inherited-state comparison while present exactly once in
  the optimizer inventory;
- default forward returns one `[B, 10]` tensor, training forward returns two
  `[B, 10]` tensors, and the tap is exactly block index 3 with 128 channels;
- fixed clean and CutMix examples match independent FP64/FP32 references for
  both `L_main`, `L_aux`, and `L_main + w * L_aux`, including lambda 0 and 1;
- the coefficient is exactly 0.15 before 0.50, continuous at 0.50, strictly
  decreases in the taper, is zero at and after 0.75, and the head is not called
  when its coefficient is zero;
- an auxiliary-only backward produces finite nonzero gradients in the
  companion head, stem, and blocks 0-3, but no gradients in blocks 4-5 or the
  final classifier; and
- main-only and combined backward paths are finite, the combined gradient
  differs from main-only upstream of the tap, and neither path advances any
  RNG stream beyond the inherited forward's normal stochastic-depth draws.

The production summary must print enough accuracy-independent evidence to
audit the fixed intervention after the transient log is deleted:

- attachment index/channels, head type, head parameter count and optimizer
  ownership;
- fixed peak coefficient, full-dose boundary, taper boundary, and initialization
  seed;
- full-dose, taper, and zero-dose batch counts; coefficient min/max and sum;
- clean and CutMix companion-active counts, alongside the inherited CutMix
  applied/eligible ratio;
- separate post-synchronization aggregates for main loss and active auxiliary
  loss, plus final finite head-parameter displacement from initialization;
- companion training-forward calls, default/main-only calls, evaluator calls,
  and zero companion evaluation calls; and
- epochs, steps, charged/total time, parameter count, peak VRAM, and the complete
  inherited final summary.

Avoid per-step gradient-norm reductions or `.item()` operations before the
existing CUDA synchronization. Counters and loss aggregates are observational
only and must not affect scheduling, optimization, or evaluation. Any target,
schedule, inventory, nonfinite, or evaluator-isolation audit failure must make
the run exit nonzero after printing diagnostics.

## Accuracy-Blind Physical-GPU-0 Preflight

Run one decisive correctness-and-feasibility preflight before the metric run.
Materialize the exact EXP-002 parent and candidate in the same bounded harness.
Before any GPU work, verify physical index 0 is an NVIDIA H20 with approximately
98 GB memory; under `CUDA_VISIBLE_DEVICES=0`, require exactly one visible device
with the same UUID. Monkeypatch the evaluator to raise immediately and assert
zero test-loader iterations, so no preflight result can reveal test accuracy.

Use real training batches and BF16/channels-last production paths. Exercise
clean and CutMix batches in each of the full-dose and taper regions, then the
zero-dose final-quarter path. Include at least one optimizer update in each
region. In a separate correctness prefix, replay identical parent/candidate
state up to the first forward and prove exact main-logit parity before the
candidate's auxiliary gradient intentionally changes shared weights. Verify
the fixed loss decomposition, CutMix target identity, expected gradient reach,
head update and later zero-dose immobility, finite state, optimizer ownership,
RNG neutrality of head construction, and default-forward evaluator contract.

Benchmark five alternating-order paired rounds after warmup. Weight active
full/taper and inactive final-quarter work in the inherited 50%/25%/25% charged
mixture. Report every round ratio and ratio dispersion, but do not use the
goal's previously over-tight 0.5% dispersion ceiling as a decisive gate. The
first complete numeric result is decisive; repeat only after an exception,
assertion failure, or malformed output before any numeric gate is emitted.

Proceed to the sole metric run only if all correctness checks pass and:

- parent round drift is at most 3%;
- weighted median candidate/parent charged-step latency ratio is at most 1.03,
  and no valid round ratio exceeds 1.06;
- projected optimizer steps are at least 26,500 and projected epochs at least
  136, compared with EXP-002's 27,950 steps and 144 epochs;
- projected total runtime is below 600 seconds;
- candidate peak allocation for model, optimizer, active auxiliary branch,
  and audit state is below 1.30 GiB; and
- zero evaluator/test-data access and all target, gradient-reach, schedule,
  initialization, RNG, finite-state, and evaluation-isolation checks pass.

Do not change the attachment point, coefficient, taper, head, initialization,
latency gate, or any inherited setting after a numeric preflight result. A
completed numeric failure is a preflight reject and failed leaf, not permission
to shrink the head or coefficient within EXP-017.

## Sole Metric Run and Decision Rules

After and only after the preflight passes, launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run train.py > run.log 2>&1
```

The parent EXP-002 metric is 95.23%, so the formal local improvement threshold
is `best_test_acc >= 95.33%`. The preregistered mechanism-strength target is
`>=95.53%`, a 0.30-point gain chosen because goal-wide single-run selection and
tail noise can obscure sub-0.30 effects. The existing global best is 95.61%;
`>=95.71%` is required to beat it by the goal's 0.10-point resolution. Keep
these interpretations distinct: 95.33-95.52 is a valid local tree improvement
but weak evidence for the mechanism, and 95.53-95.70 supports the mechanism
without establishing a new resolution-clearing global best.

Require exit 0, exactly one evaluator call per completed epoch, 299.5-301.0
charged seconds, total runtime below 600 seconds, at least 26,500 optimizer
steps, at least 136 epochs, and the complete summary. Require all three schedule
regions to have nonzero dose; the maximum coefficient must be 0.15, taper
coefficients must remain in `(0, 0.15)`, every step at progress `>=0.75` must
have zero coefficient and zero companion forward, and the final companion-head
parameter displacement must be finite and nonzero. The CutMix gate, target
orientation, adjusted-lambda identity, dedicated RNG streams, and inherited
once-per-epoch evaluation semantics must all pass their audits.

- With all integrity and dose conditions, `best_test_acc >=95.33%` is an
  `improvement` over EXP-002.
- A result below 95.33% after a valid, fully dosed run is `no-improvement`; do
  not retry with another coefficient, attachment, or schedule in EXP-017.
- A realized step or epoch shortfall is a failed necessary condition and must
  not trigger a metric rerun.
- Crash, timeout, wrong GPU, extra evaluation, test access during preflight,
  nonfinite state, target mismatch, head use during evaluation/final quarter,
  incomplete audits, or modifications outside `train.py` make the result
  invalid.

Transcribe the exact final summary, phase doses, loss aggregates, CutMix counts,
head displacement, evaluator counters, and audit results into `03-execute.md`
before removing `run.log`, because the log is intentionally transient.

## Risks and Falsification Value

- The shallow residual backbone may not need improved gradient delivery; the
  companion can instead overconstrain a useful non-linearly-separable middle
  representation.
- ReLU plus pooling is intentionally minimal but lacks scale normalization.
  Adding companion BatchNorm after seeing instability would be a different
  experiment, not a repair.
- Mid-level CutMix features do not correspond exactly to rectangle area, even
  though exact shared target semantics are the cleanest controlled choice.
- Small loss/gradient kernels can cost more wall time than their FLOP count
  suggests. EXP-009 makes the preflight latency gate scientifically necessary.
- The auxiliary head changes shared training dynamics but is invisible at
  evaluation, so a gain supports training-time intermediate supervision, not
  an inference ensemble or added deployed capacity.
- One fixed-seed max-selected accuracy result cannot resolve a small effect.
  The 95.53 mechanism bar and final/tail context should temper interpretation
  of a narrow formal pass.

## Verification Checklist

1. Prove only `train.py` differs from EXP-002 and the frozen evaluator is
   untouched.
2. Prove exact inherited initialization/main-logit parity and isolated head RNG.
3. Prove fixed fourth-block tap, 1,290-parameter head, and optimizer ownership.
4. Prove exact clean/CutMix target sharing, coefficient schedule, and gradient
   reach.
5. Prove the companion is inactive after 75% and absent from evaluation.
6. Pass the decisive accuracy-blind paired preflight on physical GPU 0.
7. Run exactly one fixed-seed metric launch and apply local, mechanism, and
   global thresholds without retuning.

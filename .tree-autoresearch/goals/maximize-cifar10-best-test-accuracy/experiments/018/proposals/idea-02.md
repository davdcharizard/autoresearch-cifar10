# Proposal: Training-Only Middle-Stage Companion Classifier

## Intervention

Add one auxiliary classifier to the unchanged EXP002 PreAct WRN-16-4 and use
it only during training. Tap the residual output after block index 3, the end
of the 128-channel, 16x16 middle stage. For active steps, compute

```python
companion_logits = companion_fc(
    F.adaptive_avg_pool2d(F.relu(block3_output), 1).flatten(1)
)
```

where `companion_fc` is exactly `Linear(128, 10)`. It has 1,290 parameters and
raises the parent count from 2,748,890 to 2,750,180 (+0.047%). Do not add a
companion BatchNorm, convolution, hidden layer, dropout, temperature, or a
second backbone forward. This placement is deep enough to supervise both
64-channel blocks and both 128-channel blocks directly, while leaving the two
256-channel blocks and final classifier trained solely by the main objective.
It is also far enough from the input to ask for semantic rather than local
texture discrimination.

Use the fixed constants below; this is not an attachment, coefficient, or head
capacity sweep.

```python
COMPANION_BLOCK_INDEX = 3
COMPANION_CHANNELS = 128
COMPANION_PEAK_WEIGHT = 0.15
COMPANION_FULL_END = 0.50
COMPANION_END = 0.75
COMPANION_INIT_SEED = 42018
```

Preserve EXP002's architecture outside this head, CutMix, drop path, LR
schedule, Nesterov SGD, weight decay, BF16 autocast, channels-last layout,
batching, seed 42, 300-second charged budget, and once-per-epoch evaluation.
Only `train.py` may change. Every GPU command must expose physical GPU 0 only
with `CUDA_VISIBLE_DEVICES=0` and verify the visible device is the approximately
98 GB NVIDIA H20.

## Loss and Schedule

Compute the coefficient from the inherited charged-time `progress` at step
entry:

```python
def companion_weight(progress):
    if progress < 0.50:
        return 0.15
    if progress < 0.75:
        return 0.15 * (0.75 - progress) / 0.25
    return 0.0
```

The coefficient is therefore fixed at 0.15 over `[0, 0.50)`, linearly tapers
over `[0.50, 0.75)`, and is exactly zero over `[0.75, 1.00]`. Its ideal
time-integrated value is 0.09375. The moderate peak acknowledges that this is
only a six-block residual network, so vanishing gradients are not established
as a limiter. The taper makes the companion an early representation-shaping
signal rather than a permanent requirement for intermediate linear
separability. Turning it off at 75% also leaves EXP002's validated clean,
drop-path-decay tail exactly main-loss-only.

Factor the inherited target calculation into a deterministic helper that is
called independently on main and companion logits. For a clean batch,

```text
L_main = CE(main_logits, targets_a)
L_aux  = CE(companion_logits, targets_a)
L      = L_main + w(progress) * L_aux
```

For a CutMix batch, both heads must receive the exact same target pair and the
same clipped-area lambda already produced by the unchanged parent helper:

```text
L_main = lam * CE(main_logits, targets_a)
       + (1 - lam) * CE(main_logits, targets_b)
L_aux  = lam * CE(companion_logits, targets_a)
       + (1 - lam) * CE(companion_logits, targets_b)
L      = L_main + w(progress) * L_aux
```

Do not redraw a permutation or lambda, detach the tapped representation,
convert the companion target to a hard label, or adapt the coefficient from
loss or gradient magnitudes. Reusing EXP002's area-weighted target deliberately
tests intermediate supervision without confounding it with a second target
policy or RNG stream.

When `w == 0`, do not compute the pooled feature, companion logits, or auxiliary
CE. Because the parent already calls `zero_grad(set_to_none=True)`, the head's
gradients remain `None` and coupled SGD weight decay cannot move the dormant
head during the final quarter. The head shares the parent's optimizer group,
LR, momentum, and weight decay while active; it gets no special optimizer
settings.

## Initialization and Evaluation Isolation

Construct and initialize all inherited modules in their original order before
creating the companion. Create and Kaiming-initialize the companion inside an
isolated CPU RNG context seeded with `COMPANION_INIT_SEED`, with zero bias, and
restore the enclosing RNG state exactly. Candidate and parent built from seed
42 must therefore have bitwise-identical inherited parameters and buffers, and
head construction must not advance the crop, shuffle, drop-path, or CutMix RNG
streams.

Keep the production call `model(inputs, drop_scale=...)` unchanged: it returns
only the final `[B, 10]` logits and skips all companion computation. A separate
explicit training-only flag may return `(main_logits, companion_logits)` while
the coefficient is positive. The frozen `Eval.evaluate(model, device)` must
always take the default path. Companion logits must never be evaluated,
ensembled, averaged with final logits, used for checkpoint selection, or
reported as test accuracy. Evaluation inputs, labels, loss, cadence, and
`best_test_acc` selection remain exactly EXP002's.

## Evidence and Mechanistic Case

Deeply-Supervised Nets (`knowledge/papers/deeply-supervised-nets.md`, AISTATS
2015) supports training-time companion classification objectives as a way to
make intermediate representations discriminative and provide a more direct
supervised gradient, with the auxiliary head removed at inference. This
proposal adapts that mechanism conservatively: one pooled linear head, one
backward, no extra backbone pass, and no inference use.

EXP002's CutMix gain shows that its remaining error responds to regularization,
while the goal's system understanding says stable generalization rather than
memory is limiting. The companion could complement CutMix by requiring the
middle-stage representation itself to support the same mixed-label semantics,
instead of leaving all discrimination to the last two blocks. Unlike a wider
inference model, the added capacity is explicitly disposable.

The counter-case is strong enough to make the experiment informative. Residual
connections and BatchNorm already deliver gradients through this shallow
network; early linear separability may constrain useful nonlinear features;
and adding an auxiliary objective during the same interval as CutMix and full
drop path may over-regularize. The head can also reduce optimizer exposure if
its apparently small kernels incur launch overhead. A null or negative result
would therefore reject this fixed middle-stage supervision recipe, not deep
supervision in every architecture.

## Correctness and Accuracy-Blind Preflight

Before a metric run, perform deterministic CPU FP32 checks for all of the
following:

- bitwise equality of every inherited candidate/parent parameter and buffer,
  unchanged global CPU RNG state after head construction, and exact default-
  forward main-logit equality on fixed inputs;
- exactly one 128-to-10 companion head with 1,290 parameters, present exactly
  once in the optimizer and absent from the inherited-state comparison;
- default forward returning one `[B, 10]` tensor and the active training path
  returning two `[B, 10]` tensors tapped after block 3;
- independently recomputed clean and CutMix `L_main`, `L_aux`, and combined
  loss, including lambda 0 and 1 edge cases and exact target orientation;
- schedule values at both boundaries and interior points: 0.15 before 0.50,
  continuous at 0.50, strictly decreasing during the taper, and zero from
  0.75 onward;
- auxiliary-only backward producing finite nonzero gradients in the head,
  stem, and blocks 0-3, with no gradients in blocks 4-5 or the final head; and
- no companion call or head movement on zero-dose optimizer steps.

Then run one decisive accuracy-blind preflight on physical GPU 0. Replace the
evaluator with a guard that raises, guard test-loader iteration, and assert zero
test examples, evaluations, or accuracy values. Exercise real BF16,
channels-last training batches in the three production workload classes:
active CutMix, active clean, and inactive clean. Verify finite forward,
backward, and optimizer state; the exact loss decomposition; CutMix target
identity; expected gradient reach; head movement while active and immobility
while inactive; and the default evaluator-facing contract.

After warmup, benchmark five alternating-order parent/candidate paired rounds
using the inherited time mixture: 37.5% active CutMix, 37.5% active clean, and
25% inactive clean. Synchronize around complete charged steps, include data
mixing and optimization, and reset equivalent model/optimizer/RNG state for
each comparison. Report all round latencies, ratios, parent drift, robust ratio
dispersion, peak allocation, and projected 300-second dose. The first complete
numeric result is decisive; retry only an exception, assertion failure, or
malformed output occurring before a numeric gate is emitted.

Proceed to the sole metric launch only if every correctness check passes,
parent timing drift is at most 4%, median weighted candidate/parent step ratio
is at most 1.03, no round ratio exceeds 1.06, and the projection retains at
least 27,000 optimizer steps and 138 epochs versus EXP002's 27,950 steps and
144 epochs. Projected total runtime must remain below 600 seconds. Record VRAM
but do not reject solely for memory because the goal defines it as a soft
consideration. Do not tune the attachment, coefficient, taper, head, or gates
after seeing any numeric preflight result.

## Production Dose Audit and Decision

The production summary must record, without changing optimization or exposing
test information:

- attachment index/channels, head type, parameter count, initialization seed,
  and optimizer ownership;
- fixed peak coefficient and boundaries; full-dose, taper, and zero-dose step
  counts; coefficient minimum, maximum, and sum;
- companion-active CutMix and clean counts, inherited CutMix applied/eligible
  counts, companion forward count, and zero companion forwards in the final
  quarter;
- synchronized aggregate main and active auxiliary losses, plus finite nonzero
  final head displacement from initialization;
- evaluator calls, completed epochs and steps, charged/total time, peak VRAM,
  parameter count, and the complete inherited metric summary.

Keep audit reductions outside the critical per-step path where possible; do
not retain one CUDA tensor per step or add gradient-norm synchronizations. All
companion work is inside the existing charged timer. Require exit 0, one and
only one evaluation per completed epoch, 299.5-301.0 charged seconds, total
runtime below 600 seconds, a complete final summary, and internally consistent
schedule and dose counters. The dose floors are mechanism-interpretation
context, not permission to override the goal's formal accuracy verdict after a
valid completed run.

The parent metric is 95.23%, so the formal falsifiable prediction is:

> Training-only middle-stage supervision will produce
> `best_test_acc >= 95.33%` while satisfying the fixed-time and evaluation
> constraints.

Treat `95.33-95.52%` as a formal local improvement but weak, noise-limited
mechanism evidence. `>=95.53%` is the preregistered stronger-support threshold
because the goal history finds sub-0.30-point single-run differences difficult
to distinguish from selection noise. `>=95.71%` clears the current 95.61%
global best at the goal's required resolution. A valid result below 95.33%
falsifies this fixed companion recipe; failing the accuracy-blind feasibility
gate rejects it for dose/cost without making an accuracy claim.

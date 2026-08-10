# Proposal: Stage-2 Training-Only Companion Classifier on EXP-004

## Summary

Add one disposable linear companion classifier to the end of EXP-004's second
residual stage. The unchanged deployed path remains the 256-channel final
classifier. During training only, the 128-channel activation after residual
block index 3 is ReLU-activated, globally averaged, and classified by a
`Linear(128, 10)` head. Optimize

```text
L_joint = L_main + 0.15 * L_companion
```

on every optimizer step. The companion uses exactly the main head's hard or
area-corrected CutMix targets. On a scheduled late SAM update, the same joint
objective is used on both the unperturbed pass that defines the perturbation
and the perturbed pass whose gradient drives the sole SGD update. The head is
included in the SAM parameter inventory, then discarded entirely by the
default/evaluation forward path.

This is a single fixed deep-supervision hypothesis, not an attachment-point,
head-capacity, coefficient, phase, or seed sweep. It preserves EXP-004's WRN,
front-loaded CutMix, period-two clean-tail SAM, drop path, Nesterov SGD, cosine
schedule, BF16/channels-last path, fixed seed 42, evaluator, and 300-second
charged-training budget. Only `train.py` may change.

## Exact Architecture and Attachment

EXP-004 has six residual blocks with the following stage structure:

```text
blocks 0-1: 64 channels, 32x32   (stage 1)
blocks 2-3: 128 channels, 16x16  (stage 2)
blocks 4-5: 256 channels, 8x8    (stage 3)
```

Tap the tensor returned by `blocks[3]`, after the second 128-channel residual
block and before `blocks[4]` performs the 128-to-256 downsampling transition.
For a training batch, this tensor has shape `[B, 128, 16, 16]`. Compute:

```python
stage2_features = out
companion_logits = companion_fc(
    F.adaptive_avg_pool2d(F.relu(stage2_features), 1).flatten(1)
)
```

The ReLU is consistent with the pre-activation network's use of activated
features at the next block boundary, while avoiding a new stateful
normalization path. Do not add companion BatchNorm, convolution, dropout,
temperature scaling, an MLP, or a projection head. `companion_fc` has exactly
`128 * 10 + 10 = 1,290` trainable parameters, increasing the parent count from
2,748,890 to 2,750,180 (0.047%). It belongs to the same SGD parameter group as
the backbone and final classifier, with the inherited learning rate, momentum,
Nesterov setting, and coupled weight decay.

Keep the public/default forward contract unchanged:

```python
model(inputs, drop_scale=...) -> main_logits  # [B, 10]
```

Add an explicit training-only request such as `return_companion=True` that
returns `(main_logits, companion_logits)`. Only the charged training loop may
set that flag. The default path must neither pool stage-2 features nor call the
companion head. Consequently the frozen evaluator receives the same main-logit
tensor as EXP-004, and the companion contributes no inference compute, no
ensemble prediction, and no checkpoint selection signal.

## Fixed Coefficient and Rationale

Use these fixed constants:

```python
COMPANION_BLOCK_INDEX = 3
COMPANION_CHANNELS = 128
COMPANION_WEIGHT = 0.15
COMPANION_INIT_SEED = 42021
```

The main loss retains coefficient 1.0 and the companion has a constant 0.15
coefficient for the entire charged run. A 15% dose is deliberately subordinate:
for initially comparable cross-entropies, about 87% of the scalar joint loss is
still the deployed head's objective. It is nevertheless large enough to send a
direct, persistent classification gradient through the stem and first four
blocks. The value reuses the conservative peak dose developed before any
companion-head metric was observed in EXP-017's unexecuted deep-supervision
proposal; it is not selected from EXP-021 test accuracy or a coefficient
preflight.

A constant coefficient is preferable here to tapering the head away at 75%.
EXP-004's defining improvement comes from the clean final-quarter SAM phase.
Turning the companion off exactly at that boundary would test early
deep-supervision followed by the unmodified SAM objective, whereas the stated
hypothesis is that a more discriminative stage-2 representation should also
shape the final basin. Keeping 0.15 fixed makes both SAM passes optimize one
well-defined joint objective and avoids the mathematically inconsistent case
where the first SAM gradient includes a loss that the second-pass update does
not.

Do not alter the coefficient after preflight timing, loss-scale diagnostics,
intermediate validation, or final accuracy. A loss- or gradient-normalized
coefficient, warmup, decay, stage switch, or a second head is a different
experiment.

## Target Semantics

Factor the existing target logic into a deterministic helper applied
separately to the main and companion logits. It must preserve the parent's
exact cross-entropy defaults.

For an ordinary hard-label batch:

```text
L_main = CE(main_logits, targets_a)
L_companion = CE(companion_logits, targets_a)
L_joint = L_main + 0.15 * L_companion
```

For an early CutMix batch, reuse the exact `targets_a`, `targets_b`, and
rectangle-area-corrected `adjusted_lam` produced by EXP-004's unchanged helper:

```text
L_main = adjusted_lam * CE(main_logits, targets_a)
       + (1 - adjusted_lam) * CE(main_logits, targets_b)

L_companion = adjusted_lam * CE(companion_logits, targets_a)
            + (1 - adjusted_lam) * CE(companion_logits, targets_b)

L_joint = L_main + 0.15 * L_companion
```

Do not detach the stage-2 tensor, redraw a CutMix permutation or lambda, use a
hard dominant label for the companion, add smoothing, or reinterpret the
rectangle at the intermediate resolution. Area weighting is an imperfect
description of content in a stage-2 receptive field, but sharing the parent's
target is the controlled choice; a different auxiliary target policy would
confound deep supervision with a second augmentation mechanism.

EXP-004 guarantees that CutMix ends before SAM starts. Assert the inherited
invariant that a scheduled SAM batch has `targets_b is None`. Both SAM passes
therefore use the same hard target tensor:

1. Save CUDA RNG state immediately before the first forward. Compute main and
   companion logits, form the hard-label `L_joint`, and backpropagate it. This
   complete joint gradient defines the normalized SAM perturbation.
2. Perturb every trainable parameter, including `companion_fc.weight` and
   `companion_fc.bias`, using EXP-004's global FP32 norm and exact snapshots.
3. Clear first-pass gradients, replay the saved CUDA RNG, and suppress
   BatchNorm running-stat tracking exactly as the parent does. Recompute both
   heads and the identical hard-label `L_joint` at perturbed parameters.
4. Backpropagate the perturbed joint loss, restore all parameters and
   BatchNorm flags exactly, then execute the sole Nesterov-SGD update.

The first-pass auxiliary gradient must not be carried directly into optimizer
state, and the second pass must not silently fall back to main-only CE. Either
mistake changes the stated joint-objective SAM method. The progress loss may
continue to report the second-pass joint loss on SAM steps because that is the
loss whose gradient drives the update.

## Initialization and RNG Isolation

The companion must not perturb EXP-004's inherited initialization or any
subsequent data, CutMix, or stochastic-depth random stream. Construct and
initialize every inherited module in its original order, including the final
classifier, and run the existing `self.apply(self._weights_init)` before adding
the companion. Then construct `companion_fc` while saving and restoring the
global CPU RNG state, and overwrite its parameters using a dedicated CPU
`torch.Generator` seeded with `42021`:

```python
parent_rng_state = torch.get_rng_state()
try:
    companion_fc = nn.Linear(128, 10)
finally:
    torch.set_rng_state(parent_rng_state)

companion_generator = torch.Generator(device="cpu").manual_seed(42021)
init.kaiming_normal_(companion_fc.weight, generator=companion_generator)
init.zeros_(companion_fc.bias)
```

The implementation may encapsulate this pattern, but it must not call a new
global `manual_seed`, and state restoration must be exception-safe. The
dedicated generator is not retained or used during training. Head pooling,
linear projection, and cross-entropy consume no random draws, so for equal
input state the candidate's main forward consumes exactly the same CUDA RNG as
the parent; the normal SAM replay continues to reproduce the first pass's
drop-path masks.

Before the charged timer, save detached copies of the two initialized companion
parameters solely for a final finite displacement audit. These copies add about
5 KiB and are not optimizer parameters or SAM snapshots.

## Mechanistic Rationale

Deeply-Supervised Nets reports that an intermediate companion classification
objective can make hidden representations directly discriminative and improve
CIFAR classification while removing the head at inference. The later
Auxiliary Training work independently supports disposable training-time
classifiers, although its largest results use corruption, selective BatchNorm,
distillation, and classifier alignment that are intentionally absent here.
Those papers establish a plausible mechanism, not a transferable effect size.

EXP-004 is a useful local test because it already fits strongly: it reached
95.40%, ended at its best checkpoint, and reduced final loss to 0.1654 despite
using only 25,560 optimizer steps. Its remaining problem is therefore more
consistent with representation/generalization quality than simple failure to
optimize. The stage-2 tap directly shapes the features shared by the first four
blocks while leaving the two stage-3 blocks free to transform them for the
deployed classifier. It adds no backbone forward and does not replace the
validated CutMix or SAM mechanisms.

Attaching earlier, after the 64-channel stage, would impose linear class
separability on low-level features and lengthen the direct-gradient path less
usefully. Attaching after stage 3 would be nearly redundant with the existing
final pooled classifier. The end of stage 2 is the one fixed point that has
semantic capacity, a meaningful two-block downstream path, and a cheap
128-dimensional head.

The intended causal claim is narrow: if the live main model improves while the
head is never evaluated, persistent stage-2 companion supervision improved the
EXP-004 training package. It does not establish that gradient starvation was
the only cause, that the auxiliary logits are better predictions, or that the
same head would help without CutMix/SAM.

## Charged Compute and Feasibility

All companion pooling, logits, losses, backward work, and SAM snapshot/update
work must remain between the existing per-batch `t0` and CUDA synchronization.
The 300-second `TIME_BUDGET_S` is unchanged. The head adds one small global pool,
a `256 x 128` by `128 x 10` matrix multiply, and one ten-class cross-entropy per
ordinary update. A period-two SAM pulse performs the branch twice. If EXP-004's
25,560-step and 2,449-pulse exposure were unchanged, the candidate would execute
about 28,009 companion forward/loss branches. The arithmetic and parameter
storage are tiny, but the pool, loss, and backward launches can be latency-bound
on H20, so feasibility must be measured rather than inferred from FLOPs.

Run one accuracy-blind paired preflight using the exact EXP-004 and candidate
production paths on physical GPU 0. Use real train batches, BF16,
channels-last, clean and CutMix ordinary updates, and clean SAM updates. Prevent
all evaluator/test-data access. After warmup, benchmark at least five
alternating-order parent/candidate rounds weighted by EXP-004's approximate
75% ordinary early phase plus 25% late phase with half of late updates using a
second SAM pass. Report every ratio and parent drift. Proceed only if all
correctness checks pass, median charged-step ratio is at most 1.03, no valid
round exceeds 1.06, parent drift is at most 3%, projected exposure is at least
24,000 optimizer steps, projected total time is below 600 seconds, and peak
allocation is safely below 1.30 GiB.

The first complete numeric preflight is decisive. A numeric gate failure is a
failed feasibility leaf, not permission to taper the head, lower the
coefficient, move the tap, or omit it from SAM. Only an exception, assertion
failure, or malformed result before numeric gates are emitted may be repaired
and rerun according to the experiment protocol.

## Required Correctness Checks

Before the paired timing result, prove the following without touching the test
set:

- A parent and candidate created from identical seed-42 state have bitwise
  equal inherited parameters and buffers. Their default-forward main logits
  and post-forward CUDA RNG states are exactly equal for a fixed input and
  drop-path scale.
- Companion construction leaves the global CPU RNG byte-for-byte unchanged;
  repeated heads from seed 42021 are identical, while no training randomness
  is drawn from the dedicated generator.
- The only new trainable tensors are the 1,290 companion parameters, present
  exactly once in the optimizer, SAM parameter list, and SAM snapshot list.
- Default forward returns one `[B, 10]` tensor and never calls the companion.
  Training forward returns two `[B, 10]` tensors, taps block index 3 at exactly
  `[B, 128, 16, 16]`, and continues from the untampered stage-2 tensor into
  block 4.
- Fixed clean and CutMix logits match independently calculated loss references
  for `L_main`, `L_companion`, and `L_joint`, including CutMix lambdas 0 and 1.
- An auxiliary-only backward produces finite nonzero gradients in the
  companion, stem, and blocks 0-3, but none in blocks 4-5 or the final
  classifier. Combined loss changes upstream gradients relative to main-only
  loss, while all required gradients remain finite.
- A full companion SAM smoke performs two companion calls, reuses the exact
  hard targets and stochastic-depth masks, updates BatchNorm buffers exactly
  once, perturbs and restores the companion with the rest of the model, and
  performs exactly one optimizer/momentum update from the second joint
  gradient.
- Injected failures during the perturbed pass restore every parameter,
  BatchNorm tracking flag, and snapshot-owned state exactly as EXP-004 does.
- No preflight evaluator call or test-loader iteration occurs.

These checks may report main/companion loss ratios and the companion-head share
of the first SAM gradient's squared norm as descriptive diagnostics, but those
values are not tuning signals and have no accuracy-dependent acceptance
threshold beyond finiteness and a nonzero companion gradient.

## Production Diagnostics

Print a fixed startup inventory and a final audit sufficient to reconstruct the
intervention after `run.log` is deleted:

- attachment block `3`, channels `128`, head type `Linear(128,10)`, head
  parameter count `1290`, total count `2,750,180`, optimizer ownership, and SAM
  snapshot ownership;
- coefficient `0.15`, initialization seed `42021`, target policy
  `shared_area_corrected`, schedule `full_run`, and evaluation policy
  `main_only`;
- optimizer-step counts split into clean ordinary, CutMix ordinary, and clean
  SAM, plus companion first-pass calls and SAM second-pass calls;
- require `companion_first_pass_calls == num_steps` and
  `companion_second_pass_calls == sam_applied_batches`; therefore total charged
  companion calls equal `num_steps + sam_applied_batches`;
- zero companion calls from the default/evaluator path, exactly one evaluation
  event per completed epoch, and no extra best-accuracy source;
- finite nonzero L2 displacement of the final companion weights and bias from
  their saved initialization, with no use of that value in scheduling;
- the inherited CutMix exposure, SAM cadence/first-step audits, complete metric
  summary, charged/total time, epochs, steps, peak VRAM, and final parameter
  count.

Counters must be observational only. Do not add per-step `.item()` calls,
gradient reductions, hooks, or synchronization beyond the parent's existing
work merely to populate diagnostics. Static checks, sparse use of values
already computed by SAM, and post-run parameter displacement are sufficient.
Any target, call-count, inventory, finiteness, restoration, RNG, or evaluator
isolation mismatch must make the run invalid rather than silently disabling
the head.

## Expected Effect and Decision Rule

The formal parent-relative threshold is `95.50%` because EXP-004 achieved
95.40%. A useful mechanism-sized target is `>=95.70%` (+0.30 points), chosen
before the run because prior goal history shows sub-0.30-point single-run and
tail variation. The current global best is 95.61%, so `>=95.71%` is required to
clear it by the goal's 0.10-point resolution. A plausible but uncertain effect
range is roughly +0.15 to +0.35 points (95.55-95.75): the literature prior is
positive and the cost is low, but this six-block residual network is much
shallower and better conditioned than the networks for which deep supervision
was originally motivated.

After a passing preflight, launch exactly one fixed-seed metric run on physical
GPU 0:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run train.py > run.log 2>&1
```

Require exit 0, 299.5-301.0 charged seconds, total time below 600 seconds,
at least 24,000 optimizer steps, one evaluation per epoch, a complete summary,
all companion/SAM/CutMix audits, and no protected-file changes. With those
conditions, `best_test_acc >=95.50%` is a local improvement. A valid result
below 95.50% is no improvement and must not be retried with a different head,
coefficient, schedule, or seed. Interpret 95.50-95.69 as a formal but
sub-mechanism-sized gain, 95.70 as mechanism-sized but just below the global
resolution bar, and at least 95.71 as a new resolution-clearing global best.
Carry final accuracy and the final 16-evaluation mean/range beside the selected
maximum so an isolated checkpoint does not overstate the effect.

## Causal Risks and Falsification Value

- The compact pre-activation WRN may not be gradient-starved. Directly forcing
  stage-2 linear separability can discard intermediate information that the
  final two blocks would otherwise exploit, worsening the stable endpoint.
- The auxiliary gradient changes shared representations and the global SAM
  perturbation direction. Because the head itself is also perturbed, this is a
  test of companion supervision integrated with EXP-004's SAM package, not a
  pure one-pass deep-supervision effect. Omitting the head from either SAM pass
  would improve attribution superficially but define an incoherent objective.
- CutMix rectangle area is exact at the input but only approximate for a
  stage-2 receptive field. Shared semantics avoid a policy confound, yet noisy
  intermediate targets may make the companion harmful during the first 75%.
- Kaiming-initialized auxiliary logits may be less calibrated than the main
  logits. The fixed 0.15 coefficient limits their scalar influence, but loss
  scale and gradient direction need not track scalar weighting exactly.
- The companion adds little arithmetic but several small kernels on every
  step and twice on SAM pulses. Launch overhead can reduce exposure enough to
  erase a representation gain; the paired charged-time preflight and realized
  24,000-step floor separate this failure mode from accuracy.
- The CVPR auxiliary-training headline gains depend on mechanisms excluded
  here, and the original deeply supervised results used different networks and
  schedules. A null result would specifically close this minimal stage-2
  companion-CE implementation, not all auxiliary training.
- A gain cannot be attributed to inference capacity because default evaluation
  never executes the head. Conversely, a small best-only gain without a lifted
  late tail is weak evidence and should not immediately be composed with EMA.

## Verification Checklist

1. Confirm EXP-004 at 95.40% is the parent and only `train.py` can differ.
2. Prove inherited initialization, main logits, and RNG consumption match the
   parent before the intended auxiliary gradient changes weights.
3. Prove the exact block-3 tap, 1,290-parameter linear head, optimizer/SAM
   ownership, isolated seed 42021 initialization, and main-only default path.
4. Prove shared hard/CutMix loss semantics and the fixed full-run coefficient
   without any loss- or metric-based adaptation.
5. Prove both SAM passes use the identical joint hard-label objective, replay
   stochastic masks, update BatchNorm once, restore exactly, and step once.
6. Pass the decisive accuracy-blind paired H20 preflight with projected and
   realized exposure gates.
7. Run one bounded fixed-seed metric launch, preserve complete diagnostics, and
   judge it against 95.50/95.70/95.71 without retuning.

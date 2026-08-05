# Proposal: Critical-Period Augmentation With Clean Fine-Tuning

## Summary

Train the existing ResNet-20 with a moderate modern regularization stack for the
first 72% of the measured 300-second training budget, then remove the extra
regularizers and finish on the baseline crop/flip distribution with hard labels.
The early phase combines RandAugment, an 8x8 mean-filled Cutout, and mixup. The
late phase is explicitly paired with lower learning rates so that the network can
recover hard-label confidence and adapt its batch-normalization statistics to the
less-distorted data distribution before the final evaluations.

This is a focused training-policy change: preserve the model, batch size, SGD,
momentum, weight decay, data normalization, seed, and evaluation cadence. All
implementation stays in `train.py` and uses existing PyTorch/torchvision APIs.

## Hypothesis

The baseline reaches 91.54% after 38,254 steps and 99 epochs, but it sees only
random crops and flips and optimizes hard-label cross-entropy throughout. Its
likely limiter is generalization, not insufficient model fitting. Moderate
augmentation and mixup during the early critical period should build smoother,
more invariant features, while removing both during the last 84 training seconds
should avoid the slow convergence and depressed hard-label confidence that can
make mixup unattractive under a short budget.

The testable prediction is `best_test_acc >= 91.64%`, with a reasonable target of
roughly 92.0-92.5%, while still completing in under 10 minutes. At baseline
throughput, the switch occurs near epoch 71 and step 27,500, leaving about 27-28
epochs and 10,700 steps for clean adaptation and convergence. Mixup may reduce
optimizer-step throughput slightly, so the schedule must be keyed to measured
training time rather than a fixed epoch or step.

## Exact Intervention

### Phase schedule

- Define `REGULARIZED_FRACTION = 0.72`; the regularized phase ends at 216 counted
  training seconds. Select the phase at epoch boundaries using
  `total_training_time / TIME_BUDGET_S`, so one transform policy is stable for
  every epoch and validation remains exactly once per epoch.
- Phase A, 0-216 seconds: baseline random crop and horizontal flip, followed by
  `transforms.RandAugment(num_ops=2, magnitude=7)`, tensor conversion,
  normalization, and an 8x8 Cutout. Implement Cutout with
  `transforms.RandomErasing(p=1.0, scale=(0.0625, 0.0625), ratio=(1.0, 1.0),
  value=0)`. Because erasing follows normalization and the configured standard
  deviation is one, `value=0` is a mean-filled patch rather than an artificial
  extreme pixel value.
- Phase B, 216-300 seconds: use only the baseline crop, flip, tensor conversion,
  and normalization. Disable RandAugment and Cutout entirely. This is
  "baseline-clean" fine-tuning rather than unaugmented memorization.
- Use separate regularized and clean transform objects. Either switch the dataset
  transform before creating each epoch iterator (workers are non-persistent in
  the current loader) or use two identically configured loaders. Do not change
  the data split or sampling policy.

### Mixup and loss

- During Phase A, draw one `lambda ~ Beta(0.2, 0.2)` per minibatch, create one
  GPU `randperm`, and train on
  `lambda * inputs + (1 - lambda) * inputs[perm]`.
- Compute one forward pass and the standard mixed-target objective:
  `lambda * CE(logits, targets) + (1 - lambda) * CE(logits, targets[perm])`.
- During Phase B, use the original inputs and ordinary hard-label
  `F.cross_entropy(outputs, targets)`.
- Do not add label smoothing in EXP-001. Mixup already supplies soft targets in
  the phase where regularization is wanted; stacking label smoothing makes the
  effective target less class-specific and adds over-regularization risk for a
  small ResNet under a short budget. Hard labels are also the mechanism by which
  the final phase restores confidence. Label smoothing is a later ablation only
  if augmentation without mixup is selected.

### Optimization alignment

- Preserve SGD at LR 0.1, momentum 0.9, weight decay `1e-4`, and batch size 128.
- Replace the step milestones, whose 48,000-step drop is unreachable in the
  measured baseline, with a time-aligned three-stage LR policy: 0.1 during Phase
  A, 0.01 from 72-90% of counted training time, and 0.001 from 90-100%.
- Apply the LR selected from `total_training_time / TIME_BUDGET_S` immediately
  before each optimizer step. This makes the policy robust if mixup changes the
  achieved step count and gives the clean phase 54 seconds of adaptation followed
  by 30 seconds of low-LR convergence.
- Preserve the fixed seed 42. There is one run of this intervention; no seed
  rerolling or result-conditioned seed selection.

## Why This Combination

The saved mixup distillation identifies a low-overhead way to directly reduce the
baseline's generalization gap, but warns that mixup may delay peak hard-label
accuracy under a short training budget. The saved Time Matters distillation
provides the key scheduling mechanism: augmentation and mixup deliver much of
their benefit during an early critical period and can be removed later without
losing that benefit. The saved RandAugment distillation supports a small fixed
operation count and shared magnitude without policy-search dependencies.

The combination is stronger than any component alone because the early image
transformations teach spatial and appearance invariance while mixup constrains
behavior between examples; the clean phase specifically addresses their shared
convergence risk. `magnitude=7`, `alpha=0.2`, and an 8x8 mask are intentionally
moderate. This avoids treating three individually useful regularizers as if their
maximum strengths were additive.

Sources:

- `experiments/001/papers/mixup.md` (Zhang et al., ICLR 2018)
- `experiments/001/papers/randaugment.md` (Cubuk et al., NeurIPS 2020)
- `experiments/001/papers/time-matters-regularization.md` (Golatkar et al.,
  NeurIPS 2019)
- `goals/maximize-cifar10-test-accuracy/04-results.tsv` (91.54% baseline)

## Budget and Constraint Fit

- The baseline measured 38,254 steps in 300.1 counted training seconds, about
  7.84 ms/step. Mixup adds only a permutation, one image interpolation, and a
  second target-loss calculation around a single model forward/backward pass.
  A modest step-count reduction is acceptable because the policy reserves 28%
  of time for faster clean batches.
- RandAugment and Cutout execute in the existing eight data-loader workers, and
  input-fetch time is outside the harness's counted training interval. They can
  increase total wall time, but should remain comfortably below the 10-minute
  hard timeout; total time and achieved epochs must still be checked.
- Mixup requires approximately one additional image batch plus a permutation,
  negligible against 98 GB H20 memory. The model and parameter count are
  unchanged.
- Evaluation remains once after each epoch. `prepare.py`, the evaluator, dataset,
  test-time path, and dependencies remain unchanged.

## Risks and Diagnostics

- **Over-regularization:** RandAugment, Cutout, and mixup can compound. The late
  clean phase and moderate strengths are the main mitigation. If training loss
  remains high through the switch or accuracy jumps only after regularization is
  removed, the next ablation should remove Cutout first, then reduce the
  regularized fraction to 60-65%.
- **Distribution handoff:** A sudden transform/loss switch can briefly destabilize
  optimization. Aligning the first LR drop with the switch and retaining crop/flip
  in both phases limits this shock. Log a one-line phase transition with elapsed
  time, epoch, step, and LR so the handoff can be verified.
- **CPU transform cost:** RandAugment may extend total wall time even though data
  loading is not charged to the 300-second counter. If total runtime approaches
  10 minutes, RandAugment is the first component to simplify; the run must be
  killed and classified as failed if it crosses 10 minutes.
- **Fewer steps:** If mixup substantially lowers throughput, time-based phase and
  LR transitions still occur as intended. Record `num_steps` and compare it with
  38,254 to distinguish an optimization failure from lost training volume.

## Verification

1. Confirm one NVIDIA H20 is visible and preserve seed 42.
2. Remove stale `run.log`; run `uv run train.py > run.log 2>&1`.
3. Verify the logged phase switch occurs at the first epoch boundary after about
   216 counted seconds and that evaluation appears no more than once per epoch.
4. Confirm `training_seconds` is approximately 300, total runtime is below 600
   seconds, and the summary is present.
5. Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `num_epochs`,
   `num_steps`, and `peak_vram_mb`.
6. Count the experiment as an improvement only if `best_test_acc >= 91.64%`.

## Effort and Expected Value

Estimated implementation effort is medium: two transform policies, a phase gate,
mixup loss, and a time-aligned LR update. The architecture and evaluation surface
remain untouched, so the behavioral blast radius is limited to training. The
expected value is high for a first modernization experiment because it directly
targets generalization while explicitly designing around the 300-second
convergence constraint.

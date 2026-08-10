# Proposal: Low-Cost Soft-Target Regularization Before Stronger Augmentation

## Summary

Use label smoothing as the first controlled modernization of the ResNet-20
training objective. Change only the existing cross-entropy call to use
`label_smoothing=0.1`, while leaving the input pipeline, optimizer, learning-rate
schedule, batch size, model, seed, and evaluation cadence unchanged.

This is the best first experiment among label smoothing, Mixup/CutMix, and
RandAugment because the baseline already takes 595.4 seconds end to end. The
hard ten-minute limit therefore leaves only 4.6 seconds of wall-clock headroom.
Label smoothing has no host-transform cost and only a very small loss-computation
cost. Mixup/CutMix consumes part of the counted 300-second GPU training budget,
while RandAugment adds per-image host work that can increase the uncounted but
hard-limited wall-clock time. A one-line label-smoothing experiment is also much
easier to interpret than a stack of interacting regularizers.

## Mechanism

The baseline trains against a one-hot target distribution. With smoothing
strength `epsilon = 0.1`, the loss uses a target distribution that assigns most
mass to the correct class while distributing a small amount across all classes.
This discourages increasingly confident logits after an example is already
classified correctly. For a small ResNet trained for roughly 99 epochs, that can
reduce overfitting and improve the representation without requiring more data,
parameters, or optimizer steps.

The three candidate techniques regularize in materially different ways:

| Technique | Regularization mechanism | Runtime effect in this harness | Main experimental risk |
|---|---|---|---|
| Label smoothing | Softens each target toward the uniform class distribution | No host work; negligible extra GPU loss work | Effect may be modest; `0.1` may slightly underfit |
| Mixup | Interpolates pairs of images and computes a lambda-weighted loss for both labels | GPU permutation, image blend, and two target losses occur inside the timed step | Fewer steps and slower convergence under 300 seconds; alpha interaction |
| CutMix | Replaces an image rectangle and mixes labels by pasted area | GPU permutation, mask/rectangle work, and mixed-label loss occur inside the timed step | More code and variance; small 32x32 rectangles can be destructive |
| RandAugment | Applies randomly selected image transforms at a shared magnitude | Per-sample CPU/PIL transform work occurs outside `t0`, but increases total wall time | Baseline has only 4.6 seconds before the ten-minute cutoff; distortions can be too strong |

Mixup has good CIFAR-10 evidence and is the most attractive follow-up from this
group. A conservative later test would use `alpha=0.2` and a lambda-weighted pair
of cross-entropies, without label smoothing. CutMix is less compelling as the
first mixing test because its rectangle geometry adds implementation choices on
32x32 inputs. RandAugment is also evidence-backed, but it should not be attempted
until transform throughput and total runtime can be kept safely below the hard
limit. None should be combined with label smoothing initially: mixing already
creates soft targets, so stacking the methods risks excessive regularization and
would make the source of any gain or regression ambiguous.

## Exact Proposed Change

In `train.py`, replace:

```python
loss = F.cross_entropy(outputs, targets)
```

with:

```python
loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)
```

Make no other training-code changes in experiment 001. In particular:

- keep `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` unchanged;
- keep random crop and horizontal flip as the only input augmentations;
- keep batch size 128, SGD hyperparameters, and step milestones at 32,000 and
  48,000;
- keep `MAX_STEPS`, model architecture, loader workers, and evaluation once per
  epoch unchanged;
- do not add Mixup, CutMix, RandAugment, a second loss term, or a smoothing
  schedule in the same run.

`torch.nn.functional.cross_entropy` is already imported and supports label
smoothing, so this requires no dependency or data-pipeline change.

## Hypothesis

Moderate label smoothing will reduce overconfident fitting late in training and
raise `best_test_acc` from 91.67% to at least 91.80%, satisfying the required
0.1-percentage-point improvement threshold, while preserving approximately the
baseline's 99 epochs and 38,525 optimizer steps. A realistic expected range is
roughly +0.1 to +0.4 percentage points; a larger gain is possible but should not
be assumed for a single low-cost regularizer.

The prediction is specifically about `best_test_acc`, not calibration or test
loss. Smoothed training can improve accuracy while changing loss values in ways
that are not directly comparable to the hard-target baseline.

## Expected Benefit

- Near-zero implementation complexity and no new failure-prone data path.
- No extra VRAM of practical significance and no CPU transform overhead.
- Regularization applies throughout the run and targets a known weakness of
  one-hot cross-entropy: continually rewarding excessive class confidence.
- The result is highly attributable. A success establishes that the baseline is
  regularization-limited; a failure cleanly rules out this smoothing strength
  before testing richer augmentations.
- It preserves Mixup and RandAugment as independent follow-up experiments rather
  than consuming their upside in an uninterpretable combined run.

## Risks and Failure Modes

1. **Underfitting under a short schedule.** ResNet-20 may need most of its limited
   optimization budget to fit the training set, and `epsilon=0.1` may weaken the
   useful target signal. This would appear as lower accuracy through the late
   evaluations, not merely a different train-loss scale.
2. **Small effect relative to run noise.** The gain may be below 0.1 percentage
   points. The seed must not be changed to rescue the result.
3. **Tiny step-throughput regression.** The smoothed loss performs additional
   reduction work. Even a small slowdown could reduce the number of steps within
   300 counted seconds or move the last evaluation to an earlier learning state.
4. **Ten-minute wall-clock fragility.** Although label smoothing does not add
   host work, the baseline's 595.4-second total leaves little operational margin.
   Normal evaluation or system jitter could still cross the 600-second cutoff.
5. **Loss comparability.** Smoothed cross-entropy has a different optimum from
   hard-target cross-entropy. Training-loss values should not be used alone to
   claim improvement or diagnose failure.

## Confound Controls

- Compare against the recorded unmodified baseline of 91.67%, 99 epochs, 38,525
  steps, and 595.4 seconds total; do not rerun with alternate seeds.
- Treat the loss change as the sole independent variable. Retaining the exact
  loader, crop/flip pipeline, schedule, optimizer, and model prevents augmentation
  and optimization changes from being credited to smoothing.
- Record `num_steps`, `num_epochs`, `training_seconds`, `total_seconds`, and peak
  VRAM alongside accuracy. A material step-count decrease is a runtime confound
  and should be reported even if accuracy improves.
- Use the unchanged once-per-epoch evaluator and the printed `best_test_acc` as
  the only decision metric. Do not increase evaluation frequency.
- Require the normal numeric summary and total completion within ten minutes.
  A timeout is a failed experiment, not a result to extrapolate.
- Do not stack label smoothing with Mixup/CutMix. Their overlapping soft-target
  mechanisms would prevent attribution and may over-regularize the small model.

## Fixed-Budget Feasibility

The baseline averages about 128.4 optimizer steps per counted training second
(`38,525 / 300`) and about 7.79 ms per step. The proposed operation is implemented
inside the existing fused PyTorch loss path and does not copy or transform input
images. It should therefore consume far less budget than Mixup or CutMix and
should leave the 32,000-step learning-rate drop reachable. The 48,000-step drop
is already unreachable in the baseline and remains unchanged as a control.

The limiting feasibility concern is total runtime rather than VRAM or the 300
training seconds. Validation and other excluded work account for approximately
295.4 seconds beyond the counted training budget. Because only 4.6 seconds remain
before the hard timeout, the run must be monitored against elapsed wall time and
killed at ten minutes as required. This concern is strongest for RandAugment:
its host-side per-image transforms can extend wall time even though their cost is
outside the per-step `t0` measurement. Label smoothing avoids that specific
hazard. No compilation, extra startup, or new data scan is introduced.

## Suggested First Experiment Scope

Run exactly one trial with `label_smoothing=0.1` and no other code changes. The
success criterion is `best_test_acc >= 91.77%` in a valid run, with 91.80% or
higher expected after two-decimal reporting. Also verify that the step count stays
close to the baseline and that the process prints its final summary before the
ten-minute limit.

Decision after the run:

- If it improves by at least 0.1 percentage points, retain label smoothing as a
  validated low-cost component, but still test Mixup separately before combining
  anything.
- If it is neutral or worse, revert the one-line change. The next experiment from
  this family should be Mixup with `alpha=0.2`, hard-label weighted cross-entropy,
  and no label smoothing.
- Defer RandAugment until there is a measured plan for maintaining wall-clock
  headroom, and defer CutMix until plain Mixup establishes that batch mixing is
  beneficial under the time budget.

## Evidence

- Zhang et al., *mixup: Beyond Empirical Risk Minimization* (ICLR 2018): convex
  input/target combinations improve generalization on CIFAR-10 architectures;
  distilled in `experiments/001/papers/mixup.md`.
- Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a
  Reduced Search Space* (NeurIPS 2020): strong augmentation without policy-search
  cost; distilled in `experiments/001/papers/randaugment.md`.
- Muller et al., *When Does Label Smoothing Help?* (NeurIPS 2019): soft targets
  can improve generalization and calibration, while interactions with other
  soft-label methods require care; distilled in
  `experiments/001/papers/label-smoothing.md`.

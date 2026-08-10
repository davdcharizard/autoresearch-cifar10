# Proposal: Conservative Mixup on the Validated Long-Plateau Schedule

## Summary

Compose one conservative generalization intervention with the successful EXP-002
optimizer schedule: apply standard minibatch Mixup with `alpha=0.2` on every
training batch, while retaining the 80% `lr=0.1` hold, step to `0.01`, cosine
decay to `1e-4`, ordinary SGD momentum, model, crop/flip pipeline, persistent
workers, and bounded evaluation schedule exactly as they are.

EXP-002 reached `91.83%` after the hard-label training objective had already
driven training loss low. The remaining limiter is therefore generalization, not
failure to optimize the current objective. Mixup directly targets that gap by
training the network to behave linearly between examples and by replacing
overconfident one-hot supervision with pairwise soft targets. `alpha=0.2` is at
the conservative end of the paper-supported range: most draws remain close to
one endpoint, preserving a strong class signal and convergence speed under the
fixed 300-second training budget.

## Mechanism

For a batch of inputs `x` and labels `y`, draw one
`lambda ~ Beta(0.2, 0.2)`, randomly permute the batch, and construct

```text
x_mix = lambda * x + (1 - lambda) * x[permutation]
L = lambda * CE(model(x_mix), y)
    + (1 - lambda) * CE(model(x_mix), y[permutation])
```

The same scalar lambda is used for the entire minibatch. This is the canonical,
minimal Mixup form and adds no model parameters. It discourages memorization and
sharp decision boundaries between training examples, which is complementary to
EXP-002's long high-learning-rate exploration and terminal low-rate refinement.

The composition has a deliberate division of labor. The existing schedule keeps
240 counted seconds of high-rate exploration and 60 seconds of low-rate
refinement. Mixup changes only the examples and target objective seen inside
those windows. Because schedule progress is calculated from
`total_training_time / TIME_BUDGET_S`, modest per-step overhead reduces the
number of updates but does not silently shorten either time window.

## Exact Candidate Implementation

Add one hyperparameter next to the existing training constants:

```python
MIXUP_ALPHA = 0.2
```

After `device` is created, construct a GPU-resident Beta distribution once:

```python
mixup_concentration = torch.tensor(MIXUP_ALPHA, device=device)
mixup_distribution = torch.distributions.Beta(
    mixup_concentration, mixup_concentration
)
```

Inside the current timed training step, after inputs and targets are transferred
to the GPU and before the forward pass, replace only the hard-label forward/loss
block:

```python
permutation = torch.randperm(inputs.size(0), device=device)
mixup_lambda = mixup_distribution.sample()
mixed_inputs = (
    mixup_lambda * inputs + (1.0 - mixup_lambda) * inputs[permutation]
)

optimizer.zero_grad()
outputs = model(mixed_inputs)
loss = mixup_lambda * F.cross_entropy(outputs, targets) + (
    1.0 - mixup_lambda
) * F.cross_entropy(outputs, targets[permutation])
loss.backward()
optimizer.step()
```

Keep lambda as a scalar CUDA tensor. Do not call `.item()` in the timed path:
that would introduce a needless device synchronization. Sampling both lambda and
the permutation on the GPU also avoids consuming the CPU RNG stream used by
DataLoader shuffling, so the host-side data-order mechanism remains as close as
possible to EXP-002. The GPU RNG is already fixed once with
`torch.cuda.manual_seed(42)`; no seed is changed or retried.

The two cross-entropy calls are intentionally retained instead of introducing a
manual soft-target loss. With only ten logits per example, their incremental
cost is small relative to the ResNet forward/backward pass, and the canonical
form minimizes correctness risk. Do not clamp lambda, force it above `0.5`, add
a per-batch application probability, or resample based on class identity; those
would be additional policy choices rather than the proposed single standard
intervention.

## Parameters and Scope

- **Mixup alpha:** `0.2`.
- **Application:** every training minibatch, throughout the full 300-second
  budget.
- **Pairing:** one random in-batch GPU permutation per minibatch.
- **Lambda granularity:** one scalar draw per minibatch.
- **Loss:** lambda-weighted sum of two ordinary index-target cross-entropies.
- **Evaluation:** unchanged clean CIFAR-10 test images and hard-label evaluator.
- **Schedule:** preserve all EXP-002 constants and elapsed-time LR logic exactly.

This first run is not an alpha sweep. An alpha of `0.2` is strong enough to test
the mechanism but less likely than `0.4` or larger values to delay fitting in the
short horizon. An always-on policy is easier to interpret than introducing both
alpha and a Mixup probability. If the result later shows underfitting, reducing
alpha or disabling Mixup only during the 20% refinement tail can be evaluated as
separate follow-ups rather than folded into this experiment.

## Evidence and Rationale

Zhang et al., *mixup: Beyond Empirical Risk Minimization* (ICLR 2018), trains on
convex combinations of image pairs and their one-hot targets, encouraging local
linearity and reducing memorization. The work reports CIFAR-10 generalization
gains across multiple residual and convolutional architectures. The local
distillation records `alpha=0.1-0.4` as a common range and identifies Mixup as a
low-memory, dependency-free fit for this repository:
`knowledge/papers/mixup.md`.

Local experimental evidence identifies the schedule to preserve. EXP-002 held
`lr=0.1` for 80% of counted time, then stepped to `0.01` and cosine-decayed to
`1e-4`; it improved `best_test_acc` from `91.67%` to `91.83%`, completed 38,629
steps in 300.0 training seconds, and finished in 336.0 total seconds. The goal
learnings explicitly classify the long high-rate plateau followed by low-rate
refinement as a reusable pattern. Keeping it fixed avoids repeating EXP-001's
confounded short-hold schedule and lets this experiment attribute any change to
Mixup.

## Hypothesis and Expected Impact

The testable hypothesis is that `alpha=0.2` Mixup will reduce hard-label
overfitting while retaining enough endpoint-dominant examples to converge within
the fixed horizon, increasing `best_test_acc` from `91.83%` to at least `91.93%`.
A plausible target range is `91.95-92.25%` (a gain of roughly 0.12-0.42 percentage
points), with the best checkpoint expected during the existing low-rate tail.

The expected gain is modest rather than transformational because ResNet-20 has
limited capacity and only about 100 epochs of updates. The proposal favors a
high probability of preserving EXP-002 performance over a stronger Mixup setting
that might offer more asymptotic regularization but converge too slowly in 300
seconds.

## Runtime and Fixed-Budget Implications

Mixup adds one GPU Beta sample, one `randperm`, one indexed batch read, an image
blend, and a second ten-class cross-entropy reduction per optimizer step. The
mixed image tensor is about 1.5 MiB at batch size 128 (`128 x 3 x 32 x 32` in
FP32), so VRAM growth should be small relative to the H20's capacity. No host
transform, extra DataLoader pass, new evaluation, compilation stage, or package
is introduced.

All Mixup work occurs after the current `t0`, so its cost is honestly charged to
the fixed 300-second training budget. A small decrease from EXP-002's 38,629
steps is expected; approximately 37,500-38,500 steps is a reasonable diagnostic
range. The elapsed-time schedule still guarantees the same 80%/20% time split.
If the run falls materially below 37,000 steps, throughput loss becomes a serious
confound and should be called out even if accuracy is close to the baseline.

EXP-002 used 336.0 total seconds, leaving roughly 264 seconds before the hard
600-second limit. Mixup adds no CPU/PIL work and may slightly reduce the number
of completed epochs and tail evaluations, so total-wall feasibility is strong.
The standard ten-minute supervisor remains mandatory, but unlike host-side
RandAugment this proposal does not threaten the wall limit through work excluded
from `total_training_time`.

## Failure Modes and Risks

1. **Convergence is slower than the regularization benefit.** Soft targets may
   require more updates than fit in 300 seconds, especially when used through the
   low-rate tail. Accuracy could remain below EXP-002 even if the method would
   win under a longer epoch budget.
2. **The 80% high-rate plateau and Mixup over-regularize together.** EXP-002
   relies on prolonged high-LR noise as useful implicit regularization. Adding
   input/target mixing may make that regime too exploratory for this small model.
3. **Throughput falls enough to change optimization exposure.** Mixup's GPU
   operations are counted, so fewer batches and epochs are processed even though
   the LR time fractions remain fixed.
4. **BatchNorm sees interpolated image statistics.** Training BatchNorm state is
   fitted on mixed images while evaluation uses clean images. Conservative alpha
   limits, but does not eliminate, this train/evaluation distribution shift.
5. **Training loss is no longer comparable.** Weighted mixed-target loss has a
   higher irreducible value than hard-label CE. A higher logged loss is not by
   itself evidence of failed optimization.
6. **Single-run metric variance masks a small gain.** The seed and run count must
   remain fixed. Do not reroll to turn a sub-threshold result into an improvement.
7. **Implementation-induced synchronization.** Calling `.item()` on the CUDA
   lambda or doing CPU-side sampling inside the step can add overhead or perturb
   RNG behavior; the proposed tensor-only path avoids both.

## Confound Controls and Excluded Interventions

Treat the committed EXP-002 `train.py` as the moving baseline and change only the
Mixup constant, distribution construction, batch permutation/blend, and weighted
loss. Specifically, do not add or alter any of the following:

- label smoothing, CutMix, RandAugment, AutoAugment, erasing, or additional crop
  policies;
- optimizer type, momentum/Nesterov setting, weight decay, batch size, model
  width/depth, initialization, or normalization;
- `LR`, `ANNEAL_START_LR`, `MIN_LR`, `LR_HOLD_FRACTION`, or the elapsed-time
  schedule formula;
- DataLoader worker count, persistent workers, shuffle behavior, or host-side
  transforms;
- evaluation checkpoints, dense-tail evaluation, test preprocessing, or
  `Eval.evaluate()`;
- random seeds, multiple trials, checkpoint selection outside the existing
  `best_test_acc` protocol, dependencies, or files other than `train.py`.

Mixup and label smoothing should not be stacked in this run because both soften
targets and can over-regularize a small model. Likewise, do not combine Mixup and
CutMix behind a random switch: that would test a policy family rather than the
specific mechanism and would obscure a negative result.

## Verification

Before execution:

1. Confirm the diff touches only `train.py` and contains only the scoped Mixup
   additions above; confirm `prepare.py`, dependency files, schedule constants,
   model, loader, and evaluator are unchanged.
2. Run static compilation and the repository's Ruff/pre-commit checks.
3. Confirm exactly one NVIDIA H20 with approximately 98 GB VRAM is selected.
4. Confirm there is no stale completed `run.log` or renamed run-log variant.

Execute one run only with the fixed seed:

```bash
uv run train.py > run.log 2>&1
```

Monitor it without streaming the full log and kill it at 600 seconds if it has
not exited. A valid run must print a unique numeric summary, record approximately
300 seconds of counted training, run evaluation no more than once per epoch, and
finish below 600 seconds total.

Decision criteria:

- **Improvement:** `best_test_acc >= 91.93%`, at least 0.10 percentage points
  above the moving baseline of `91.83%`, with all integrity checks passing.
- **No improvement:** valid completion below `91.93%`; do not rerun with another
  seed.
- **Runtime-confounded:** valid or invalid result with a surprisingly large
  step-count loss (especially below about 37,000); record throughput as the main
  diagnostic before judging stronger Mixup.
- **Failure:** crash, missing/nonfinite summary, scope violation, more than one
  evaluation in an epoch, wrong hardware, counted-budget violation, or timeout.

Record `best_test_acc`, final accuracy/loss, training and total seconds, epochs,
steps, and peak VRAM. Compare the full late-evaluation trajectory to EXP-002,
while recognizing that mixed training loss is not numerically comparable to the
hard-label baseline.

## Follow-Up Interpretation

- If this experiment improves, retain `alpha=0.2` Mixup as a validated component
  of the EXP-002 schedule. Any later alpha tuning should be a separate experiment.
- If accuracy rises during the tail but ends below the threshold, a later
  Mixup-off terminal refinement is a plausible test, but it must not be retrofitted
  into this run.
- If accuracy is uniformly lower while throughput is healthy, the likely cause
  is over-regularization or insufficient finite-horizon fit; revert Mixup before
  testing another generalization intervention.
- If throughput is the dominant failure, optimize the same Mixup implementation
  or use a cheaper soft-target regularizer in a later experiment rather than
  stacking compensating changes now.

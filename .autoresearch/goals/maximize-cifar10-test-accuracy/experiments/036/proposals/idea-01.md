# Proposal: Exclude Only the Linear Classifier Weight from Decay

## Exact Treatment

Keep the accepted `67c8e98` learner unchanged except for optimizer allocation:
`fc.weight` receives zero weight decay for the full run, while every
convolution weight continues to receive coupled SGD weight decay `5e-4` and
all existing BatchNorm parameters and biases remain at zero decay.

Implement the allocation from `model.named_parameters()` so name and rank are
both explicit:

```python
named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
decay_params = [
    p for name, p in named_params if p.ndim >= 2 and name != "fc.weight"
]
no_decay_params = [
    p for name, p in named_params if p.ndim < 2 or name == "fc.weight"
]
```

The existing two SGD groups and all their other options remain unchanged. Do
not add runtime switches, norm telemetry, new constants, or another optimizer
group. The accepted model has 987,098 parameters: 984,752 currently decayed
matrix parameters, of which `fc.weight` is exactly 1,280 (`[10,128]`). The
candidate therefore has 983,472 continuously decayed parameters and 3,626
zero-decay parameters. `fc.bias` remains in the latter group as before.

## Rationale and Distinction from EXP007

The accepted run nearly interpolates its hard-label tail yet ends at 94.22%
accuracy and 0.2523 test loss, so representation generalization and final
decision boundaries, rather than training fit, are limiting. The classifier is
only 0.13% of decayed parameters but is the direct class-boundary map. Removing
its coupled shrinkage lets its ten class vectors track the already-regularized
128-dimensional representation without weakening convolutional feature decay.

EXP007 does not test this allocation. It removed decay from all matrix weights
during the last 35%, scored 93.74%, and worsened loss to 0.3244. This proposal
preserves continuous `5e-4` decay on 983,472 convolution parameters and changes
only 1,280 classifier parameters from the first optimizer step. It therefore
tests where decay belongs, not when global decay should end. The perturbation
is still material: ignoring gradients and momentum, the accepted schedule's
rough mean LR of 0.101 across 25,978 steps would impose about
`exp(-5e-4 * 0.101 * 25978) = 0.27` isolated multiplicative shrinkage.

All accepted architecture, initialization, FP32 arithmetic, batch 256,
time-based `0.2 -> 0.002` LR, momentum 0.9, Nesterov, alpha-0.2 batch-shared
mixup through 65%, worker-safe N1/M5 RandAugment through the first exhausted
epoch at or after 65%, crop/flip, seed 42, evaluator cadence, and limits remain
unchanged.

## Semantic Preflight

Use an evaluator-free disposable harness and an independent
`git show 67c8e98:train.py` oracle. Before timing or scoring, require:

- the production diff is only the named optimizer allocation above and
  `prepare.py` is byte-identical;
- accepted and candidate model topology, initialization, all parameter bytes,
  987,098 count, loader/transforms, schedule, loss paths, constants, and
  post-construction CPU/CUDA RNG states are exact;
- every trainable parameter appears exactly once in each optimizer; accepted
  group counts are 984,752/2,346 and candidate counts are 983,472/3,626;
- candidate decay tensors are exactly all rank-2-or-greater tensors except the
  uniquely named `[10,128]` `fc.weight`; candidate zero-decay tensors are
  exactly all rank-below-2 tensors plus that same object;
- both groups retain accepted LR, momentum, Nesterov, and other defaults, with
  live decay values exactly `[5e-4, 0.0]`;
- from cloned state and identical fixed CPU/CUDA RNG, one complete early-mixup
  step has bitwise-equal inputs, lambda, permutation, logits, losses, gradients,
  and pre-step RNG state in both arms; the same holds for a hard-label step;
- after either one-step probe, every parameter except `fc.weight`, including
  `fc.bias`, is bitwise equal; each arm's `fc.weight` update matches an
  independent PyTorch SGD/Nesterov oracle with decay `5e-4` versus `0.0`, the
  two weights differ nontrivially, and both momentum buffers are finite;
- post-step CPU/CUDA RNG states are bitwise equal, proving parameter grouping
  itself consumes no randomness; restoring candidate state and RNG reproduces
  the entire candidate step exactly;
- cutoff probes around 65%, source comparison, and static evaluator-call audit
  confirm mixup/RandAugment transitions and at-most-once-per-epoch evaluation
  remain accepted.

Print measured group memberships, counts, update deltas, and equality results
before assertions. A failed semantic condition closes this exact
implementation without a score; do not repair it by broadening name patterns
or changing other decay values.

## Throughput and Exposure Gate

Although total tensor count is unchanged, moving one tensor between CUDA SGD
multi-tensor groups could alter launch efficiency. On the idle H20, compare
independent accepted and candidate modules from equal model/optimizer states.
Time complete production-equivalent early-mixup and hard-label steps including
H2D, LR writes, zeroing, mixup work when active, forward, loss, finite guard,
backward, Nesterov step, and final synchronization. Use at least 20 warmups and
three alternating balanced windows of at least 50 steps per arm, with fresh
deterministic fixtures per replicate.

Print every window and all derived values before assertions. Require finite
times and population CV no greater than 5% for every arm. Compute:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms + 0.35 / accepted_hard_ms)
projected_passes = 133.00736 * retention
```

Proceed only if retention is at least `0.9774` and projected passes are at
least `130.0`. A stable miss is final: do not rerun the gate, relax it, or alter
the allocation. No loader benchmark is needed because transforms, workers,
batch shape, and consumer code are source-identical.

## Sole Scored Run and Decision Contract

After all gates pass, reconfirm baseline 94.32% at `67c8e98`, one idle NVIDIA
H20, local CIFAR-10, frozen `prepare.py`, no stale log, and the exact production
diff. Run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one finite complete summary, 300.0-300.1 counted seconds, total
below 600 seconds, 987,098 parameters, the accepted ordered mixup and exhausted-
epoch RandAugment transitions, unique every-fifth-epoch evaluations plus the
final partial epoch, and no traceback, OOM, worker, or non-finite error. Record
realized exposure as `num_steps * 256 / 50000`.

The objective succeeds only at `best_test_acc >= 94.42%`, exactly 0.10 points
above the accepted 94.32%. Pre-register `final_test_acc >= 94.32%` and
`final_test_loss <= 0.2523` as non-decisive support that the endpoint and
boundary quality did not degrade. Neither can rescue a primary-metric miss;
a primary success remains valid if they fail but the mechanism must be called
fragile.

A valid score below 130 realized passes counts and cannot be rerun, but it is
operationally inconclusive for the intended normal-exposure mechanism. Timeout,
malformed output, semantic contamination, or wrong transition is a crash.

## One-Run Family Closure

If a valid run with at least 130 passes scores below 94.42%, retain decay on
`fc.weight` and close the classifier **under-decay** family: do not try
`1e-4`, `2.5e-4`, late classifier-only disabling, a different cutoff,
classifier-bias decay, another seed, or a norm-conditioned rescue. This result
does not by itself close increased classifier decay or a separately motivated
stage-wise allocation, because those act in the opposite direction.

If the candidate succeeds, it may replace the accepted allocation solely on
the primary metric and constraints. Success does not license an adjacent decay
sweep without a new independent mechanism. A pre-score failure closes only the
exact zero-decay allocation under its tested protocol.

## Risks

- The accepted classifier decay may be important precisely because no
  normalization follows `fc`; removing it can grow logit norms, overconfidence,
  and test loss without changing useful top-1 boundaries.
- EXP007's severe loss regression is contrary directional evidence, even
  though its scope and timing differ substantially.
- Only 1,280 parameters change, so the effect may be below the ten-example
  acceptance margin despite nontrivial cumulative shrinkage.
- Optimizer grouping is deterministic, but time-based cutoffs can land one step
  apart if tiny runtime differences accumulate; exposure and transition audits
  remain binding.
- One fixed seed cannot establish average treatment effect, and no reroll is
  permitted.

## Falsifiable Hypothesis

If coupled shrinkage of the final class vectors unnecessarily constrains
boundary fitting while convolutional decay already regularizes the learned
representation, excluding only `fc.weight` from decay will retain at least 130
projected and realized passes and raise fixed-seed `best_test_acc` from 94.32%
to at least 94.42%, with final accuracy at least 94.32% and final loss no worse
than 0.2523. A valid normal-exposure miss falsifies zero/less classifier decay
as a useful standalone refinement and closes that under-decay family.

## Local Evidence

- `experiments/007/04-analysis.md`: disabling decay on every matrix for the
  final 35% scored 93.74% and 0.3244 loss, establishing the need to preserve
  continuous convolution decay.
- `experiments/027/04-analysis.md`: accepted `[2,2,3]` plus early RandAugment
  scored 94.32% best, 94.22% final, 0.2523 loss, and 133.00736 passes.
- `03-experiment-learnings.md`: continuous matrix decay and the accepted
  capacity/augmentation interaction are protected; compute-neutral
  generalization changes are preferred.
- `02-system-understanding.md`: the learner nearly interpolates training while
  test boundary quality limits the objective; compute, not memory or I/O, is
  binding.

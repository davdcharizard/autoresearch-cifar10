# Proposal: Selective 144-Channel Stage 3

## Summary

Replace only the accepted WRN-16-2 final-stage width, changing stage widths
from `[32,64,128]` to `[32,64,144]`. Keep two pre-activation blocks in every
stage and retain the complete accepted FP32, batch-256, SGD/Nesterov,
time-based cosine, alpha-0.2 batchwise mixup through 65%, crop/flip, seed-42,
and evaluation recipe.

This is a smaller version of EXP-010's `[32,64,160]` treatment, which scored
94.11% at 132.16 passes: directionally above the 94.07% accepted result but
below the required 94.17%. Width 144 adds 128,032 parameters and 8,175,776
convolution/linear MACs over accepted, approximately half of width 160's
269,888-parameter and 17,236,288-MAC increments. Its only defensible premise
is therefore a different fixed-budget operating point: retain enough of the
late-capacity signal while recovering materially more optimization exposure.

That premise is weak and must be fail-closed. Score only if a matched H20
production-path preflight projects at least **137.0 dataset passes** and at
least **96.5% throughput retention** versus a same-process accepted control.
The two gates are nearly equivalent when calibrated against 141.9 accepted
passes, but both must pass. If they fail, reject this proposal without a
scored run and do not lower the floor.

## Why This Is Probably an Adjacent Retry

EXP-010 is unusually direct prior evidence. Its exact selective-width
mechanism completed stably, exceeded its 120-pass interpretation floor, and
gained only 0.04 points. Its preregistered falsification rule explicitly said
not to try widths 144, 176, or 192 after observing the result, and its final
analysis says another width is not justified as an immediate
result-conditioned sweep. The current proposal violates the spirit of that
closure unless it establishes a materially different exposure regime before
scoring.

There is also no evidence for an interior optimum at 144. Width 160's positive
delta could be seed-scale evaluation noise; a smaller tail supplies less of
the hypothesized class-separating capacity; and BF16 plus fixed-MAC depth
redistribution show that extra exposure alone does not improve accuracy. A
144-channel result would also be difficult to interpret causally because both
capacity and exposure move toward accepted at once.

The proposal is therefore technically feasible but strategically weak. It
should rank below a genuinely new generalization or noise-scale intervention.
If selected anyway, the strict exposure gate and one-run closure below are
necessary to keep it from becoming a width sweep.

## Exact Architecture and Cost

Use the accepted stem and the first two residual stages unchanged:

| Component | Accepted | Candidate |
|---|---:|---:|
| Stem | 16 | 16 |
| Stage 1 at 32x32 | 32, 32 | 32, 32 |
| Stage 2 at 16x16 | 64, 64 | 64, 64 |
| Stage 3 at 8x8 | 128, 128 | 144, 144 |
| Final BN / classifier input | 128 | 144 |

The candidate tail must have these exact shapes and semantics:

- `layer3[0].bn1`: 64 channels;
- `layer3[0].conv1.weight`: `[144,64,3,3]`, stride 2, padding 1;
- `layer3[0].bn2`: 144 channels;
- `layer3[0].conv2.weight`: `[144,144,3,3]`, stride 1, padding 1;
- `layer3[0].shortcut.weight`: `[144,64,1,1]`, stride 2;
- `layer3[1].bn1` and `bn2`: 144 channels;
- `layer3[1].conv1.weight` and `conv2.weight`: `[144,144,3,3]`,
  stride 1, padding 1;
- `layer3[1].shortcut is None`;
- final BN has 144 channels and `fc.weight` is `[10,144]` with bias `[10]`.

There is no extra block, bottleneck, gate, dropout, auxiliary head, or changed
forward ordering. The candidate must have exactly **819,706** trainable
parameters, versus 691,674 accepted: +128,032 or +18.5%. Counting convolution
and linear MACs per image gives exactly **109,282,720**, versus 101,106,944:
+8,175,776 or +8.1%. H20 memory is not a concern; shape-specific measured
step time is authoritative because prior equal-MAC layouts differed about 20%
in H20 throughput.

## Initialization and RNG Isolation

A direct `[32,64,144]` constructor under seed 42 is not a controlled
comparison. PyTorch module constructors consume shape-dependent random draws
before `WideResNet.apply(_weights_init)` begins, so even same-shaped upstream
weights can shift. Implement accepted-first mutation instead:

1. Under the normal global seed-42 state, construct and initialize the exact
   accepted `[32,64,128]` graph with the unchanged whole-model
   `self.apply(_weights_init)` call.
2. Only after that initialization completes, enter
   `torch.random.fork_rng(devices=[])` and set only the CPU default generator
   with `torch.random.default_generator.manual_seed(26026)`. Do not call
   `torch.manual_seed`, which would also perturb CUDA generators.
3. Inside the fork, construct a new two-block `64 -> 144 -> 144` `layer3`, a
   144-channel final BN, and a `Linear(144,10)` classifier in that order. Apply
   the unchanged `_weights_init` recursively to the new `layer3`, then to the
   new BN and classifier in that exact order, and install them in place of the
   accepted tail.
4. Exit the fork before `.to(device)`. The experiment-local seed is fixed in
   advance and is not varied after timing or accuracy. It isolates one new
   tail; it is not a seed search.

This construction must leave `conv1`, `layer1`, and `layer2` parameters and
buffers bitwise identical to an accepted seed-42 oracle. It must also leave
post-construction CPU and CUDA RNG states bitwise identical to the accepted
path, preserving DataLoader shuffling, crop/flip draws, mixup coefficients,
and permutations. The complete new tail must be bitwise identical to an
independent seed-26026 oracle that performs constructor defaults followed by
the exact production initialization sequence. All candidate parameters must
appear exactly once in the existing optimizer groups: matrices in decay and
BN/linear biases in no-decay.

## Matched H20 Feasibility Gate

Use an evaluator-free preflight in a separate process. It must not construct
or call the real `Eval`, inspect CIFAR test labels, report accuracy, or write
`run.log`. Require exactly one visible NVIDIA H20 and benchmark only accepted
`[32,64,128]` and candidate `[32,64,144]`; do not time other widths.

Before timing, verify the exact topology, parameter/MAC counts, common-state
identity, post-construction CPU/CUDA RNG identity, seed-26026 tail oracle,
finite `[256,10]` FP32 logits/loss, finite gradients, one SGD/Nesterov update,
and exact optimizer membership. Fail closed on any mismatch.

Time the complete production counted body with fixed pinned host inputs and
targets: nonblocking host-to-device copies, LR/group writes, zero-grad,
Beta/randperm mixup where applicable, FP32 forward and cross entropy, finite
guard, backward, optimizer step, and final CUDA synchronization. Give each
topology an independent initially identical training RNG stream. Warm each
path for 25 mixup steps, then measure three 50-step windows per topology and
regime in balanced order. Measure mixup at 50% progress and hard labels at 80%
progress; report every window rather than only the aggregate.

For each topology and regime, use the median window mean as its center and
require population CV at most 5%. Compute:

```text
weighted_ms = 0.65 * mixup_median_ms + 0.35 * hard_median_ms
retention = accepted_weighted_ms / candidate_weighted_ms
projected_passes = 141.9 * retention
```

Proceed to the one scored run only if all semantic checks pass, all losses are
finite, every CV is at most 5%, retention is at least **0.965**, and projected
passes are at least **137.0**. Since `137 / 141.9 = 0.96547`, the pass gate is
the binding requirement. The slight rounded-retention redundancy makes the
expected operating regime explicit. Do not repeat a stable timing failure,
change window sizes, or relax either gate.

The 137-pass floor is higher than EXP-010's 132.16 realized passes by 4.84.
Without that recovery, width 144 has not demonstrated the proposed distinct
capacity/exposure balance and is merely an adjacent width retry.

## One Scored Run and Decision Rule

After a passing preflight, audit that only `train.py` differs from accepted and
that FP32, batch 256, seed 42, transforms, persistent workers, alpha-0.2
batch-shared mixup through 65%, SGD/Nesterov, selective `5e-4` decay,
time-aligned LR with 0.002 floor, and evaluation cadence are unchanged. Remove
stale `run.log`, then run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one H20, exact `[32,64,144]` topology and 819,706 parameters,
finite loss, one mixup transition near 195 counted seconds, 300.0-300.1 counted
seconds, total wall time below 600 seconds, no more than one evaluation per
epoch, and a complete final summary. Compute realized passes as
`num_steps * 256 / 50000`.

The decision is fixed before execution:

- Accept only `best_test_acc >= 94.17%` in the sole valid run.
- Any valid score below 94.17% is formal no-improvement, including 94.16%; do
  not rerun, change the local initialization seed, or try widths 136, 152,
  160, 176, or 192.
- If realized exposure is at least 137 passes and accuracy is below 94.17%,
  close selective width 144 and the immediate width-neighbor family: the
  proposed balance was present and insufficient.
- If preflight passed but realized exposure falls below 137, the run still
  counts and cannot be retried; classify the accuracy result normally but call
  the mechanism inconclusive because the preregistered operating regime did
  not materialize.
- Timeout, non-finite loss, OOM, wrong topology/count/RNG, missing summary, or
  invalid evaluation cadence is a crash, not permission for a rescue run.

## Expected Outcome and Risks

An optimistic interpolation is that width 144 recovers roughly half of
width 160's measured 6.9% exposure loss while preserving enough added tail
capacity to move another 0.06 points. That is not strong evidence: capacity
benefit need not scale smoothly, fixed-seed top-1 noise is comparable to the
observed delta, and the 64-channel stage-2 output can remain the true
bottleneck. The strict gate protects only feasibility; it cannot turn a weak
architecture premise into a strong one.

Other risks are H20 kernel cliffs at 144 channels, undertraining of the locally
initialized larger tail, and confounding capacity with recovered updates. The
new fixed local seed improves attribution of upstream weights and training RNG
but introduces a different tail initialization from EXP-010, so results cannot
be read as a clean width-response curve. This further supports treating the
proposal as low priority and terminal if attempted.

## Evidence

- `experiments/010/04-analysis.md`: `[32,64,160]` scored 94.11% at 132.16
  passes, a +0.04 near miss with adequate exposure, and explicitly closed
  adjacent width tuning.
- `experiments/010/02-plan.md` and `03-execute.md`: exact width-160 topology,
  matched production-path timing protocol, 0.923362 retention, and stable H20
  execution.
- `03-experiment-learnings.md`: selective width is promising but insufficient;
  more exposure alone and adjacent architecture changes have repeatedly missed
  the acceptance margin.
- `project-notes/project-insights.md`: channel/resolution shape must be timed
  directly on H20 because equal MAC counts can differ about 20% in speed.
- `knowledge/papers/wide-residual-networks.md`: width is a credible CIFAR
  capacity lever in general, but it does not establish an optimum at selective
  stage-3 width 144.


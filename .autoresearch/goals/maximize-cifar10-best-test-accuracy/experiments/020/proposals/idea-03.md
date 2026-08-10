# Idea: Low-Rate Batchwise Stochastic Depth

## Verdict and Exact Candidate

This idea is feasible only as a conservative shallow-network policy. During the
first 80% strong N1/M7+CutMix phase, independently skip each of the six
non-entry same-width residual branches with fixed probability `p=0.05`, one
Boolean mask per block per batch. The selected blocks are
`layer{1,2,3}[1]` and `layer{1,2,3}[2]`. Keep all three stage-entry blocks,
including both stride-2 Option-A transitions, active on every batch. Disable
stochastic depth completely for the weak tail and evaluation.

When a branch survives strong training, scale only its residual output by
`1/(1-p) = 1/0.95`; the shortcut is never scaled. When dropped, return the
same-shape shortcut input directly and do not execute either convolution or BN.
At evaluation and in the weak tail, execute the accepted block without scaling.
This inverted policy preserves expected pre-add residual contribution during
strong training while making the deployed graph exactly EXP-010.

No probability schedule, per-sample mask, transition dropping, width recovery,
fallback rate, or rescue variant is part of EXP-020.

## Why This Scope

EXP-010 remains the 94.15% frontier with 26,898 updates, a healthy 89.73%
switch, p=0.5 CutMix, and a hard weak tail. Full preactivation and selective
zero-gamma reduced switch fit by 2.85-3.25 points; a shallow ResNet-20 cannot
tolerate aggressive depth reduction. Keeping stage-entry blocks active preserves
initial feature formation and prevents the padded output channels of Option A
from becoming zero. Disabling drops in the weak tail gives every branch the
complete low-LR refinement interval.

Hayou et al., *Regularization in ResNet with Stochastic Depth* (NeurIPS 2021),
support residual-branch sampling as explicit/gradient regularization and note
that true full-branch skipping can save training compute. The evidence does not
select a ResNet-20 probability. Five percent is pre-registered as the smallest
round rate likely to produce measurable regularization and kernel savings while
retaining 95% of strong updates in each selected branch.

Source: `experiments/020/papers/stochastic-depth-regularization.md` and
[the paper](https://papers.nips.cc/paper_files/paper/2021/file/82ba9d6eee3f026be339bb287651c3d8-Paper.pdf).

## Exact Model and Mask Semantics

Add constants `STOCHASTIC_DEPTH_PROBABILITY = 0.05` and a dedicated CPU
`torch.Generator` seeded exactly 42 without consuming global CPU/CUDA RNG. Pass
one shared generator reference to the six selected blocks. Do not use the global
CPU generator: model-forward draws would shift later sampler/worker seeds. Do
not use CUDA masks: converting them to a host branch decision would synchronize.

Each selected `BasicBlock` tracks Python integer strong-survive/drop counters
for final provenance only. Its forward logic is:

```python
if (
    self.training
    and self.stochastic_depth_enabled
    and torch.rand((), generator=self.drop_generator).item()
        < STOCHASTIC_DEPTH_PROBABILITY
):
    self.strong_drop_count += 1
    return x

out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
if self.training and self.stochastic_depth_enabled:
    self.strong_survive_count += 1
    out = out / (1.0 - STOCHASTIC_DEPTH_PROBABILITY)
out += x
return F.relu(out)
```

The actual implementation should sample once into a named Boolean to avoid a
second draw. Only stride-1 equal-channel blocks may use this path; assert
`need_pad=False`. Returning `x` is forward-equivalent to `ReLU(x)` because every
selected input follows a post-add ReLU and is nonnegative.

At the 80% loader switch, set all six `stochastic_depth_enabled` flags false
before the first weak optimizer step. `model.eval()` also disables sampling via
`self.training=False`; evaluations consume no mask RNG and execute the full,
unscaled accepted graph. `model.train()` during the strong phase re-enables only
the training-mode condition, not a disabled weak-tail flag.

Parameter count and state-dict tensors remain exactly 1,073,962. The dedicated
generator and counters are nonpersistent Python state, consume no model RNG,
receive no gradient, and enter no optimizer group. With seed 42, all accepted
parameters and post-construction global CPU/CUDA RNG must remain bitwise aligned.
The mask stream is continuous through the strong phase, never reset by epoch or
evaluation, and unused in the weak tail.

## BN, Optimizer, and Effective Updates

A dropped block executes no BN. Its `bn1` and `bn2` running statistics and
`num_batches_tracked` remain unchanged for that batch. A surviving block updates
both BNs once with normal momentum; evaluation never updates them. At run end,
each selected block's two BN counters must equal
`strong_survive_count + weak_batch_count`. Unselected blocks must count every
training batch.

Keep `optimizer.zero_grad(set_to_none=True)`. Dropped parameters therefore have
`grad is None`; ordinary PyTorch SGD skips their momentum, coupled decay, and
parameter update entirely. On surviving or weak batches, accepted SGD momentum
0.9 and all-parameter decay `1e-4` apply normally. This conditional optimizer
behavior is part of stochastic depth and must not be replaced by zero gradients,
which would still advance decay/momentum.

EXP-010 used about 21.5k strong and 5.4k weak updates. At accepted exposure, a
selected branch expects `0.95*21.5k + 5.4k = about 25.8k` effective updates, or
roughly 96% of its accepted count. Conditional speedup should raise global steps
and partially recover that loss. The launch gate requires at least 25,900
projected effective updates per selected branch.

## Actual Compute-Savings Prior

The six selected blocks each cost approximately 18.87M forward MACs; together
they are about 113.25M, or 70.2% of the accepted 161.3M forward path. Five-percent
skipping therefore removes about 5.66M MACs per strong image: 3.51% of accepted
forward work during 80% of counted time, or 2.81% schedule-weighted. Backward and
saved-activation work is also skipped, but mask/Python/scale overhead reduces the
real gain. Output multiplication without conditional execution is invalid: it
would regularize but save no convolution/BN backward work.

## Persistent Real-Batch Safety Gates

Following EXP-019's replay finding, materialize and serialize one exact sequence
of 400 production N1/M7 batches, including hard and CutMix probability targets,
before launching paired control/candidate processes. Record file hash, shapes,
target ranks, and mix count; both processes must load identical tensors.

Require:

1. In eval mode, aligned control/candidate logits are bitwise equal and neither
   model advances the mask generator or BN counters.
2. In strong train mode, each selected block is observed both dropped and kept;
   aggregate survival lies in `[0.93,0.97]` and each block in `[0.90,0.99]`.
3. On a forced disposable dropped case, no selected residual kernel launches,
   parameters have `grad is None`, momentum/weights/BN counters are unchanged,
   input gradient follows the shortcut, and output equals input exactly.
4. On a forced kept case, branch gradients and BN updates are finite, residual
   scaling is exactly `1/0.95`, and shortcut values are unscaled.
5. Across all 400 identical batches, losses/logits/parameters remain finite,
   candidate terminal loss EMA is at most 1.5x control, and candidate does not
   exceed 95% one-class predictions unless control does too.
6. After disabling the feature, each selected block's weak-batch output matches
   the accepted `ReLU(bn2(conv2(...)) + x)` formula using the same candidate
   tensors; all six branches execute, scaling is absent, and no generator draw
   occurs.

Serialize evidence before assertions. A failed gate blocks production; do not
change probability or mask seed after observing it.

## Paired H20 Timing and Exposure Gates

Use five alternating fresh-process pairs on the sole idle H20. Each trial runs
an exact synthetic 80/20 schedule: 800 strong masked steps then 200 full weak
steps, alternating hard/probability targets, after 100 warmups. Include H2D,
zero-grad, conditional forward, CE, backward, SGD, and synchronization. Record
per-block masks, mean/median/p95, CV, memory, global and effective projections.

Require:

- schedule-weighted candidate/control mean ratio `<=0.99`, proving at least 1%
  real speedup rather than theoretical masking;
- projected global steps `>=27,169` and each selected branch's projected
  surviving-strong plus full-weak updates `>=25,900`;
- aggregate and per-block survival within the real-batch bounds;
- candidate p95 `<=1.03x` control, trial CV below 3%, finite state, and peak
  allocation no more than 16 MiB above control;
- eval-mode inference ratio `<=1.01`, exact full-graph logits, projected total
  wall below 540 seconds, and projected evaluator count no more than 19.

More than 19 production evaluations is invalid because speed must not grant
extra best-metric opportunities. If timing or effective-update gates fail, do
not substitute output masking, raise width, or increase the drop rate.

## Hypothesis and Failure Modes

**Hypothesis:** 5% batchwise skipping of six ordinary strong-phase branches will
add mild architectural-noise regularization while converting skipped backward
work into at least 1% more global updates, preserve at least 25,900 effective
updates per selected branch, and raise `best_test_acc` from 94.15% to at least
**94.25%**. A plausible success range is 94.25-94.45%.

Key risks:

- ResNet-20 is shallow; even 5% dropping can recreate the recurring strong
  underfit signature. A switch below 87.08% supports this mechanism.
- `1/0.95` amplifies residual and CutMix gradients on surviving batches;
  expectation preservation does not preserve variance or post-ReLU expectation.
- Selected BN statistics see only surviving batches, while evaluation combines
  every branch. This train/eval state mismatch can hurt calibration.
- Momentum and coupled decay freeze on drops, changing optimization as well as
  depth. Faster global steps do not guarantee equal per-branch learning.
- Batchwise masks regularize less diversely than per-sample drop path, but
  per-sample masks cannot conditionally skip kernels.
- Returning identity changes downstream activation distributions, and all six
  masks interact combinatorially; `0.95^6 = 73.5%` of strong batches execute the
  complete six-branch set.
- Extra epochs can add evaluation opportunities; the 19-evaluation cap is
  mandatory.
- External evidence is stronger for deeper ResNets. A bare pass is weak
  single-seed evidence and never authorizes reroll or rate tuning.

Mechanism diagnostics are switch accuracy versus 89.73% and the 87.08% marker,
first weak versus 93.16%, final NLL versus 0.1934, global/effective updates,
per-block BN/mask counts, exposure, and best-final slope. They cannot override
the primary metric.

## One-Run Verification

After every gate passes, run exactly once with
`CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`. Require
exit zero, finite summary, 300 counted seconds, total below 600, 1,073,962
parameters, one 80% switch with eight stopped workers, 45-55% CutMix, hard weak
targets, unique epochs, at most one evaluation per epoch and at most 19 total,
valid per-block masks/BN counters, and only reviewed `train.py` changes.

Accept only `best_test_acc >=94.25%` with all integrity conditions. A correct
lower run is valid no-improvement and the stochastic-depth diff is reverted. A
mechanical defect may be repaired only without changing mask scope, probability,
seed, scaling, phase, or optimizer semantics. There is no fallback experiment.

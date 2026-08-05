# Proposal: ResNet-D Projection Shortcuts at Both Downsampling Transitions

## Recommendation

Test one exact, medium-confidence architectural change: at the first block of
`layer2` and `layer3`, replace the shortcut's phase-selective `1x1, stride=2`
projection with fixed `AvgPool2d(kernel_size=2, stride=2)` followed by the same
bias-free `1x1, stride=1` projection. Leave both stride-2 main branches and the
stride-1 `16 -> 32` projection in `layer1[0]` unchanged. Preserve the accepted
`a7c42dc` model, pooled residual head, data path, optimization, schedules,
seeds, time budget, and evaluator cadence in every other respect.

This is a cheap ResNet-D-style change to how the two resolution transitions
carry information. It adds no learned capacity and does not revisit a closed
masking or input-padding treatment. Its prior is credible but not strong: a
box average uses all four spatial phases and can reduce shortcut aliasing, but
it can also blur small CIFAR features and change the main/shortcut amplitude
balance. One strictly qualified score is warranted; no pooling variant sweep
is warranted.

## Exact Production Change

Keep the existing convolution registered as `self.shortcut` so its parameter
name and parameter traversal position remain accepted. Add a parameterless
pool used only when the block stride is exactly two:

```python
self.shortcut = (
    nn.Conv2d(
        in_channels,
        out_channels,
        1,
        stride=1 if stride == 2 else stride,
        bias=False,
    )
    if stride != 1 or in_channels != out_channels
    else None
)
self.shortcut_pool = (
    nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else None
)
```

The forward shortcut is exactly:

```python
shortcut_input = (
    self.shortcut_pool(preactivated)
    if self.shortcut_pool is not None
    else preactivated
)
shortcut = (
    self.shortcut(shortcut_input) if self.shortcut is not None else x
)
```

Do not put a pool on an identity shortcut, pool the raw pre-BN block input,
move or alter the main-branch stride, add padding/ceil mode, rescale the pooled
tensor, or insert normalization/activation around the projection. Defaults
must remain `padding=0`, `ceil_mode=False`, and `divisor_override=None`.

The resulting shapes are fixed:

| Block | Preactivated input | Shortcut pool | Projection/output | Main output |
|---|---:|---:|---:|---:|
| `layer1[0]` | `[N,16,32,32]` | none | `[N,32,32,32]` | `[N,32,32,32]` |
| `layer2[0]` | `[N,32,32,32]` | `[N,32,16,16]` | `[N,64,16,16]` | `[N,64,16,16]` |
| `layer3[0]` | `[N,64,16,16]` | `[N,64,8,8]` | `[N,128,8,8]` | `[N,128,8,8]` |

All other blocks retain literal identity shortcuts. There are exactly two
`AvgPool2d` modules, both on projection paths, and exactly three learned
projection convolutions with strides `[1,1,1]` after the change.

## Information and Aliasing Mechanism

Let `z = ReLU(BN1(x))` and let `W` be a transition projection's `1x1` weight.
The accepted shortcut is

```text
S_old[o,i,j] = sum_c W[o,c] * z[c,2i,2j].
```

It chooses one fixed phase of every `2x2` cell and discards the other three
before channel mixing. The candidate is

```text
S_new[o,i,j] = sum_c W[o,c] *
               (z[c,2i,2j] + z[c,2i+1,2j] +
                z[c,2i,2j+1] + z[c,2i+1,2j+1]) / 4.
```

Because spatial averaging and a bias-free pointwise channel map are linear and
act on different axes, pool-then-project is mathematically equivalent to
project-then-pool in real arithmetic. The selected ordering is the requested
ResNet-D form and avoids performing the projection at the larger resolution.
Every spatial value contributes deterministically with weight one quarter.

This fixed box filter removes the accepted shortcut's single-phase preference,
attenuates the highest spatial frequencies before decimation, and distributes
shortcut gradients over all four input sites. It may improve local
translation/phase stability under the accepted crop, flip, and early
RandAugment regime. It is not a perfect anti-aliasing filter: a `2x2` box has a
weak stopband, destroys within-cell phase, and may suppress class-relevant fine
detail. The stride-2 `3x3` main branch remains learned and unchanged, so the
treatment also changes the relative frequency content and variance of the two
summands. There is no post-shortcut BN to automatically restore that balance.

## Relation to Local Evidence

EXP010's selective final-stage width scored 94.11% at 132.16 passes, and
EXP011's extra `8x8` block scored 94.15% at 132.92 passes: low-resolution
capacity was directionally useful but insufficient alone. EXP026's exact
RNG-isolated early RandAugment scored 94.12% at 142.45 passes. EXP027 composed
the depth and invariance signals and improved to 94.32%, after which EXP036's
post-pooling residual MLP reached the accepted 94.48% at 130.304 passes. The
current system diagnosis therefore favors generalization improvements with
near-zero spatial cost. Filtering only the two projection shortcuts fits that
constraint and acts before the accepted capacity and pooled head rather than
adding more capacity.

This is not EXP006/EXP030-style masking. Those treatments randomly removed
elements or a whole residual branch and lost 0.55/0.41 points at normal
exposure. Here the main residual branch is always complete, the shortcut is
always complete, and every site in each `2x2` cell contributes on every pass.
Average pooling is lossy, as all downsampling is, but it replaces deterministic
phase selection with deterministic aggregation; it does not create stochastic
model depth or consume mask RNG.

It is also distinct from EXP032 reflection padding. Reflection changed
worker-side crop pixels and failed a loader-variability gate. This proposal
does not touch PIL transforms, image boundaries, workers, prefetching, or data
RNG. It operates on GPU feature maps after `BN1/ReLU`, partitions even-sized
maps without padding, and cannot create reflected pixels or loader jitter.

## Initialization, RNG, State, and Cost

The two projection weights retain shapes `[64,32,1,1]` and `[128,64,1,1]`.
The pool has no parameter, buffer, random initialization, or state-dict entry.
With `self.shortcut` constructed in its accepted position and
`shortcut_pool` registered afterward, all 52 trainable tensors retain their
names, order, shapes, initial bytes, optimizer-group order, and Kaiming draw
order. The total remains exactly `1,003,482` parameters. The isolated pooled
head seed 36036 and both post-construction global CPU/CUDA RNG states must also
remain exact. Initial logits must *not* match accepted, because the intended
shortcut function changes immediately; requiring logit identity would erase
the treatment.

The learned projection work remains exactly `524,288` MACs/image at each
transition, `1,048,576` total. The new pools produce `32*16*16 + 64*8*8 =
12,288` values/image, approximately 49,152 scalar add/scale operations. Against
about 120.0M accepted convolution/linear MACs/image, arithmetic growth is only
about 0.04%. Actual cost can be larger than that ratio because two extra small
kernels, memory traffic, and pool backward do not scale like dense MACs.
Hence exposure must be measured, not inferred.

## Fail-Closed Semantic Gate

Before timing or scoring, use an ignored evaluator-free harness and print all
diagnostics before assertions. Require:

1. The production diff against `git show a7c42dc:train.py` is confined to the
   exact shortcut construction/forward change; `prepare.py` is unchanged.
2. Exactly two pools occur only in `layer2[0]` and `layer3[0]`; `layer1[0]`
   remains a direct bias-free `16 -> 32`, `1x1`, stride-1 projection. Main
   convolution shapes, strides, padding, stages, head, and classifier match.
3. Hook every block and require the shape table above, finite FP32 logits of
   `[256,10]`, `1,003,482` parameters, and 52 trainable tensors.
4. From cloned seed-42 construction states, require every named parameter and
   buffer byte, state-dict key, optimizer group/order/option, pooled-head tensor,
   and post-construction CPU/CUDA RNG state to equal accepted.
5. On deterministic synthetic transition tensors, compare production output
   to an independent reshape-and-mean plus `einsum`/linear oracle. Include
   impulse fixtures at all four `2x2` phases; each must contribute exactly one
   quarter before projection, whereas the accepted oracle selects only phase
   `(0,0)`. Require no padding or odd-size/ceil behavior.
6. Verify the local main-branch output is byte-identical before addition, the
   candidate shortcut differs nontrivially on a phase-sensitive fixture, and
   both addition operands have exact matching shapes.
7. Compare projection-weight and preactivation gradients to an independent
   oracle. Require shortcut input gradients to distribute across all four
   sites, finite nonzero parameter gradients, ordinary coupled-decay Nesterov
   updates, and no pool parameter or optimizer state.
8. Require identical CPU/CUDA RNG after candidate and accepted forwards and
   backwards from cloned states. Audit unchanged mixup draws, worker-safe
   RandAugment transition, LR samples, sole backward/step, finite guard, and
   every-fifth-plus-final evaluation contract.

A semantic failure closes before scoring. Repair only an independently proven
implementation or verifier defect; do not change ordering, kernel, stride,
placement, scale, or which transition is filtered.

## Timing Gate

On one idle H20, compare complete accepted and candidate production-equivalent
steps in both early-mixup and hard-label regimes. Include pinned H2D, LR writes,
zeroing, mixup where active, full forward/loss/finite check/backward, coupled
Nesterov update, and synchronization. Use at least 20 warmups and two
counterbalanced `A/C/C/A` cycles, yielding four windows of at least 50 steps
per arm and regime, with identical restored model/optimizer/input/RNG state.

For every local accepted/candidate pair compute

```text
retention_i =
  (0.65 / candidate_mix_i + 0.35 / candidate_hard_i) /
  (0.65 / accepted_mix_i  + 0.35 / accepted_hard_i)
projected_passes_i = 130.304 * retention_i
```

Require all window CVs `<=5%`, early and hard paired-ratio population CVs
`<=1%`, every `retention_i >= 127/130.304 = 0.9746439096`, median projected
passes `>=127`, and candidate peak allocation `<2,048 MiB`. A stable timing
miss closes systems viability without an accuracy claim and without a rerun or
pooling rescue.

## Sole Score and Decision Contract

After the gates pass, reconfirm accepted baseline 94.48% at `a7c42dc`, one
idle NVIDIA H20, frozen evaluator/`prepare.py`, local CIFAR-10, exact
`train.py`-only scope, and no stale `run.log`. Run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, total
wall time below 600 seconds, exactly 1,003,482 parameters, correct ordered
mixup and exhausted-iterator RandAugment transitions, unique every-fifth plus
final evaluations, and no traceback/OOM/worker/non-finite signature. Compute
realized exposure as `num_steps * 256 / 50000`.

Success requires both `best_test_acc >=94.58%` (exactly +0.10 points over the
accepted 94.48%) and realized exposure `>=127` passes. Final accuracy versus
94.45% and final loss versus 0.2456 are descriptive corroboration only; they
cannot rescue a primary miss or veto a primary success. A completed score
below 127 passes still consumes the sole run and is recorded, but does not
support the near-zero-cost mechanism and may not be rerun.

## Strict Closure

A valid normal-exposure score below 94.58% closes this exact two-transition
`2x2` average-pool-before-projection treatment. Do not rescue it with only one
transition, max pooling, a `3x3`/blur kernel, padding or ceil mode, learned or
gated pooling, a residual blend, gain compensation, pool-after-projection,
changed main-branch stride, filtering layer1, a schedule/cutoff, another seed,
or a second score. A success supports only the fixed complete treatment and
does not authorize a pooling sweep. The result does not close fundamentally
different anti-aliasing architectures justified by a new diagnosis.

## Falsifiable Hypothesis and Sources

If phase-selective projection shortcuts are a material source of aliasing and
translation sensitivity under the accepted early-invariance learner, then
fixed `2x2` average aggregation at both stride-2 shortcuts will preserve at
least 127 passes and raise fixed-seed `best_test_acc` from 94.48% to at least
94.58%. A normal-exposure miss falsifies that exact claim.

Offline sources: accepted `train.py` at `a7c42dc`; `01-definition.md`,
`02-system-understanding.md`, `03-experiment-learnings.md`, and
`04-results.tsv`; EXP010/011 capacity reports, EXP026/027 invariance reports,
EXP036 accepted-head report, EXP006/030 masking failures, and EXP032 reflection
failure. No network, test data, or external evaluation was used.

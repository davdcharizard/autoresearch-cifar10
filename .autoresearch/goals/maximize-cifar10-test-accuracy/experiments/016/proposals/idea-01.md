# Proposal: Compute-Aware WRN Stage-Depth Redistribution `[1,2,3]`

## Recommendation

Retain the accepted WRN widths `[32,64,128]`, six total residual blocks, and
the complete EXP-002 training recipe, but move the second 32-channel block from
the 32x32 stage to the end of the 128-channel 8x8 stage. The final stage depths
become exactly `[1,2,3]` rather than `[2,2,2]`.

This is a fixed-compute representation-allocation experiment. A same-width
basic block has exactly the same convolutional MAC count when channel width is
doubled and spatial resolution is halved: the removed 32-channel 32x32 block
and added 128-channel 8x8 block each cost 18,874,368 convolutional MACs per
image. The candidate therefore keeps the accepted model's total
convolution/linear count at 101,106,944 MACs per image while concentrating one
nonlinear transformation at the empirically promising semantic stage.

Do not combine this treatment with width 160, a bottleneck, attention,
RandAugment, initialization changes, or optimizer changes. A valid result
below 94.17% closes this exact `[1,2,3]` allocation without adjacent topology
tuning in the same experiment.

## Diagnosis And Rationale

The accepted `[2,2,2]` WRN-16-2 reaches 94.07% after about 141.9
dataset-equivalent passes and nearly fits the training set. The strongest
remaining positive signal is specifically dense 8x8 capacity: stage-3 width
160 scored 94.11% in EXP-010, and adding one full 128-channel stage-3 block
scored 94.15% in EXP-011. Both treatments lost roughly 7% of accepted exposure,
yet moved accuracy upward. In contrast, the rank-64 post-stage-3 refinement in
EXP-012 retained 95.5% exposure but fell to 93.74%, showing that the useful
signal is not merely another residual path; dense full-width transformation
appears important.

The untested question is whether the accepted model spends one of its six
blocks at the wrong resolution. Every stage currently receives two blocks even
though later feature maps encode more class-specific channels. Removing the
second stage-1 block preserves the 16-to-32 transition block and its learned
projection, so the network still performs one full nonlinear 32x32 extraction
before downsampling. Spending the released MACs on a third 128-channel block
provides another full-rank transformation after both resolution transitions.

This trades high-resolution local refinement for low-resolution semantic
refinement without increasing arithmetic or the number of residual additions.
It follows the Wide Residual Networks principle of spending a modest depth
budget effectively, sharpened by this repository's two directionally positive
stage-3 results. The principal uncertainty is whether one stage-1 block is
enough to form robust edges/textures before the 16x16 transition.

## Exact Architecture And API Change

Replace the scalar topology constant with an explicit immutable tuple:

```python
STAGE_BLOCKS = (1, 2, 3)
```

Change `WideResNet.__init__` to accept `stage_blocks` and validate it strictly:

- require a tuple of exactly three built-in positive integers;
- reject booleans, floats, lists, zero/negative entries, and extra entries;
- pass `stage_blocks[0]`, `[1]`, and `[2]` to `layer1`, `layer2`, and `layer3`;
- log exact widths and block counts rather than deriving a conventional
  `WRN-{6*n+4}` name from a scalar.

The final module sequence must be:

- stem: unchanged `3 -> 16` 3x3 convolution at 32x32;
- stage 1: one `16 -> 32` transition/basic block at 32x32;
- stage 2: unchanged two blocks at 16x16, `32 -> 64` then `64 -> 64`;
- stage 3: three blocks at 8x8, `64 -> 128` then two `128 -> 128` blocks;
- final BatchNorm, global pooling, and `128 -> 10` classifier unchanged.

Every block keeps the accepted `PreActBlock` ordering, Kaiming initialization,
shortcut policy, BatchNorm defaults, and absence of post-add activation. There
are still exactly six `PreActBlock` instances and three projection shortcuts.

## Parameter And Compute Accounting

The accepted model has 691,674 trainable parameters. The removed identity
stage-1 block contains 18,560 parameters:

- two 32-channel 3x3 convolutions: `2 * 32 * 32 * 3 * 3 = 18,432`;
- two affine BatchNorms: `2 * (32 + 32) = 128`.

The added identity stage-3 block contains 295,424 parameters:

- two 128-channel 3x3 convolutions:
  `2 * 128 * 128 * 3 * 3 = 294,912`;
- two affine BatchNorms: `2 * (128 + 128) = 512`.

The candidate therefore has exactly **968,538 parameters**, an increase of
276,864 parameters (40.03%) despite retaining six blocks. This is expected:
parameter count rises with channel width, whereas the matching spatial
reduction keeps convolutional arithmetic constant.

For each exchanged block:

```text
stage-1 block: 2 * 32 * 32 * 32 * 32 * 9   = 18,874,368 MAC/image
stage-3 block: 2 *  8 *  8 * 128 * 128 * 9 = 18,874,368 MAC/image
```

Thus the accepted and candidate convolution/linear totals are both exactly
101,106,944 MACs per image. Training convolution arithmetic is correspondingly
matched in forward and backward. The candidate reduces BatchNorm/ReLU/addition
traffic for the exchanged activations by 4x in element count, but increases
parameter, gradient, momentum, and weight-decay memory traffic by about
277k scalars. Peak memory should remain far below H20 capacity; measured
throughput, not MAC equality alone, decides whether the implementation is
eligible for scoring.

## Initialization And RNG Semantics

A naive direct construction of `[1,2,3]` changes how many random values module
constructors and `self.apply(_weights_init)` consume before later shared
tensors are initialized. That would shift stage-2, stage-3, classifier, loader,
and training RNG streams, confounding allocation with a broad initialization
trajectory change.

Preserve the accepted trajectory explicitly. Construct and initialize the
accepted `[2,2,2]` module graph first under seed 42, then remove
`layer1[1]`. Construct and Kaiming-initialize the new `128 -> 128` block inside
`torch.random.fork_rng(devices=[])`, seeding only
`torch.random.default_generator` with one fixed local CPU seed; do not call
`torch.manual_seed` inside the fork because it can also alter device generators.
Append the block as `layer3[2]` and leave the restoring fork. This occurs on CPU before
`model.to(device)`. It yields the desired final `[1,2,3]` graph while ensuring:

- every surviving accepted parameter and buffer is bitwise identical to the
  accepted model constructed from the same starting RNG state;
- the new late block uses the exact accepted `PreActBlock` initialization rule;
- the global CPU RNG state after construction equals accepted construction, so
  subsequent DataLoader shuffle/worker seeding is not shifted;
- CUDA RNG state remains untouched before the unchanged mixup training path.

The local seed is an implementation constant for the new block, not a searched
seed or reroll. Use it once and preregister it in the plan. Do not try multiple
local seeds. The removed stage-1 weights are initialized and then discarded;
that one-time CPU work occurs before the counted training timer and exists only
to preserve the accepted RNG trajectory.

This initialization-compatible construction is part of experimental control,
not a different initialization treatment. If implementation simplicity is
preferred over this control, the direct constructor is unacceptable for this
experiment because it cannot satisfy the common-state/RNG preflight below.

## Predicted Metric Impact

Predict `best_test_acc` in the **94.15-94.30%** range, centered near 94.22%,
with at least 97% matched production throughput and approximately 138-143
realized dataset-equivalent passes. The formal success threshold remains
94.17%, exactly 0.10 percentage points above the 94.07% accepted baseline.

The upside comes from retaining EXP-011's full-width late transformation while
recovering its added convolution cost. More updates under the same time-based
schedule may move the +0.08 near miss across the threshold. The expected gain
is deliberately modest because removing early depth can reduce feature quality
and because the extra late block still adds many degrees of freedom.

## Unscored Semantic Preflight

Run a local evaluator-free preflight in a fresh process. Patch `prepare.Eval`
to a fail-closed stub before importing `train.py`; use synthetic
training-shaped inputs and never inspect test accuracy.

1. Validate the constructor rejects malformed stage-count inputs and accepts
   only the exact intended tuple. Confirm the production model logs widths
   `[32,64,128]` and depths `[1,2,3]`.
2. Assert exactly six `PreActBlock` modules with per-stage counts `[1,2,3]`,
   output shapes `[B,32,32,32]`, `[B,64,16,16]`, and `[B,128,8,8]`, and exactly
   three projection shortcuts with strides `[1,2,2]`.
3. Assert exactly 968,538 trainable parameters. Independently enumerate
   convolution and linear shapes and require exactly 101,106,944 MACs/image;
   require both exchanged blocks to equal 18,874,368 MACs/image.
4. Construct accepted and candidate models from identical saved CPU/CUDA RNG
   states. Require all common state-dict tensors to be bitwise identical,
   excluding only removed `layer1.1.*` and added `layer3.2.*`; require the
   global CPU and CUDA RNG states after construction to match exactly.
5. Require every candidate tensor to be finite, the new block's convolution
   weights to be nonzero and Kaiming-scaled, and all BatchNorm scales/biases to
   equal one/zero at initialization. Confirm the discarded block is absent
   from the final optimizer parameter groups.
6. With fixed synthetic data, run both mixup and hard-label forward/backward
   steps. Require finite `[256,10]` logits/loss, finite gradients for every
   retained parameter, nonzero gradients on both convolutions in the new late
   block, and successful SGD updates without OOM.
7. Confirm the accepted scalar alpha-0.2 draw, permutation rule, 65% cutoff,
   hard-label branch, LR schedule, optimizer grouping, and evaluation cadence
   are unchanged in source and behavior.

Any topology, count, MAC, shared-state, RNG, gradient, or optimizer-membership
failure rejects the implementation before scoring.

## Throughput Preflight And Gates

Benchmark accepted `[2,2,2]` and candidate `[1,2,3]` production steps on the
single H20 in a balanced, interleaved order. Use cloned initial states, pinned
host batches, nonblocking copies, real beta sampling/permutation/mixing, LR
writes, finite guard, forward, backward, matrix-only weight decay, Nesterov SGD,
optimizer step, and final CUDA synchronization. Benchmark mixup and hard-label
regimes separately, with at least 25 warmup steps followed by three windows of
at least 50 steps per regime. Report medians and population CVs, then combine
regime times at the accepted 65/35 counted-time weights.

Proceed to the single scored run only if:

- all timing-window CVs are at most 5%;
- weighted candidate throughput retention is at least **97%**;
- projected exposure from accepted 141.9 passes is at least **137.6 passes**;
- no OOM, non-finite value, unexpected synchronization, or optimizer-state
  mismatch occurs.

Equal convolutional MACs make near-baseline throughput plausible, but the
larger late parameter tensors can alter kernel efficiency and optimizer memory
traffic. The measured gate prevents the architectural hypothesis from being
silently tested with another EXP-011-like exposure penalty. Do not relax the
gate or choose between alternate block allocations based on timing.

## Why This Differs From Failed `[2,2,3]`

EXP-011 added a seventh residual block, increasing convolutional work by
18,874,368 MACs/image (18.7% over the accepted total) and reducing realized
exposure to 132.92 passes. It tested whether *more* raw low-resolution depth
could overcome less optimization exposure. Its 94.15% score was positive but
its 0.2782 test loss was materially worse than accepted, and it missed the
required threshold by 0.02 points.

This proposal keeps six blocks and removes exactly the same number of MACs
from stage 1 as it adds to stage 3. It tests a different causal claim: the
accepted transformation budget is misallocated across resolution stages. The
candidate should recover roughly nine or more passes relative to EXP-011 and
reduces high-resolution activation work, while retaining the identical dense
`128 -> 128 -> 128` late transformation that produced the positive signal.

The parameter count remains close to EXP-011 (968,538 versus 987,098), so this
is not a low-parameter regularizer. If accuracy or test loss regresses at
normal exposure, the result would indicate the removed early block provides
essential representation/generalization value or that dense late parameters
themselves cause the confidence problem. It must not be described as a retry
or rescue of `[2,2,3]`; compute matching and early-depth removal are the core
mechanism under test.

## Risks And Interpretation

- **Early feature underdevelopment:** One 32x32 block may not build sufficiently
  robust local features before downsampling. Normal throughput with accuracy
  below 94.07% would strongly favor retaining two stage-1 blocks.
- **Late overparameterization:** Equal MACs do not imply equal statistical
  complexity. The candidate adds 40% parameters and may reproduce EXP-011's
  higher test loss or overconfidence even with more updates.
- **Optimization efficiency is shape-dependent:** H20 kernels may execute the
  128-channel 8x8 convolutions or larger SGD tensors differently despite exact
  MAC equality. The matched preflight is authoritative for operational cost.
- **Block attribution is coupled:** The experiment simultaneously removes an
  early block and adds a late block by design. It identifies whether the whole
  allocation is better, not the independent contribution of either action.
- **Initialization control bug:** Accidentally constructing `[1,2,3]` directly
  shifts shared weights and subsequent RNG. A failure of bitwise common-state
  or post-construction RNG equality invalidates the implementation.
- **Small metric margin:** EXP-011 was only 0.02 points below acceptance, but
  CIFAR-10 top-1 changes in 0.01-point increments. One fixed-seed result is
  still authoritative; do not rerun a near miss.
- **Exposure is not sufficient:** EXP-012 showed that efficient low-resolution
  computation can still harm accuracy. Normal exposure cannot rescue the
  verdict if the representation mechanism is wrong.

## Full-Run Verification And Decision Rule

After all preflight gates pass, remove stale `run.log` and run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit code 0, one NVIDIA H20, a complete final summary, approximately
300 counted training seconds, total time below 600 seconds, finite losses, and
no more than one evaluation per epoch. Confirm exactly one mixup transition
near 195 counted seconds, the unchanged hard-label tail, 968,538 parameters,
stage depths `[1,2,3]`, and no source changes outside `train.py`. Record
`best_test_acc`, final accuracy/loss, steps, epochs, peak VRAM, transition
step/time, and realized passes `num_steps * 256 / 50_000`.

Accept only if the valid fixed-seed run reaches
`best_test_acc >= 94.17%`. A result below 94.17% is `no-improvement` even if it
beats 94.07%, improves loss, or exactly ties EXP-011. Do not rescue a failure by
trying `[2,1,3]`, changing the local initialization seed, adding attention, or
altering optimizer/regularization settings in EXP-016.

## Falsifiable Hypothesis

Redistributing the accepted WRN's six residual blocks from `[2,2,2]` to
`[1,2,3]`, while preserving exactly 101,106,944 convolution/linear MACs per
image, at least 97% measured throughput, the accepted shared initialization/RNG
trajectory, and every training hyperparameter, will raise fixed-seed CIFAR-10
`best_test_acc` from 94.07% to at least 94.17% within the 300-second counted
budget.

## Evidence

- `knowledge/papers/wide-residual-networks.md`: modest depth and width can use
  CIFAR compute more efficiently than indiscriminate depth.
- `experiments/010/04-analysis.md`: selective stage-3 width scored 94.11% at
  132.16 passes, supporting low-resolution capacity directionally.
- `experiments/011/04-analysis.md`: a full extra stage-3 block scored 94.15% at
  132.92 passes, the strongest unaccepted result, but worsened test loss.
- `experiments/012/04-analysis.md`: a rank-64 post-stage-3 substitute retained
  135.49 passes but fell to 93.74%, arguing for dense transformation rather than
  a generic extra residual path.
- `03-experiment-learnings.md` and `04-results.tsv`: architecture capacity is
  the only repeatedly positive neighborhood after objective, precision,
  schedule, averaging, and initialization treatments failed.
- `train.py`: accepted `PreActBlock` and WRN construction establish the exact
  parameter, shortcut, initialization, optimizer, and schedule semantics used
  in this proposal.

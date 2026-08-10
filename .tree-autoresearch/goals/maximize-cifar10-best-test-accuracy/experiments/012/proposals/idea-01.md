# Proposal: CutMix-Complementary GPU Cutout

## Reviewer Refinement (authoritative where it differs below)

Claude's adversarial review selected this proposal but correctly found that uniformly sampling only fully contained top-left boxes does not reproduce reference Cutout. The experiment will instead sample each square's **center** independently and uniformly from integer coordinates `[0,31]` on both axes, then clip the nominal 16x16 window at image edges. Precompute a 1,024-entry center-indexed mask bank. The expected one-dimensional masked length is 14 pixels, hence expected area is 196/1,024 = 19.140625%; realized min/max area are audited rather than forcing every image to 25%.

Use `centers.random_(0, 32, generator=cutout_cuda_generator)` rather than a factory call combining `out=` and `device=`. Assert the input normalization standard deviations remain `(1,1,1)` so zero in normalized space is exactly dataset-mean fill. Tighten the full-run-weighted candidate/parent median latency gate from 1.03x to **1.01x** and require at least 25,500 projected steps.

Full complement dose is deliberate: Cutout runs on every early non-CutMix batch, about 40.7% of all parent steps, which is materially below the literature's all-image Cutout dose and maximizes the chance of a stable effect large enough to clear the threshold. This leaves no unmodified early batch but does preserve all CutMix batches and the entire clean/SAM/EMA tail. The honest preregistered expectation is a 0.20-0.40-point stable lift after redundancy discounting, not the paper table's full 0.81-point gain. Formal success is best >=95.71; mechanism support additionally requires final-16 EMA mean >=95.69 versus the parent's 95.493125.

## Summary

Add vectorized per-image Cutout to EXP-011 only on early batches where the
existing CutMix gate is not selected. Each affected image receives one exact
16x16 square filled with zero in normalized tensor space, equivalent to the
dataset mean color. Use a dedicated seed-43 CUDA generator for independent
top-left coordinates, and stop Cutout at the same charged-progress 0.75
boundary as CutMix.

This preserves every parent CutMix batch and label, adds no second view, and
leaves the clean-tail SAM/EMA package unchanged. Early batch composition becomes
approximately 50% CutMix and 50% hard-label Cutout; the final 25% remains fully
clean apart from the parent's decaying drop path and periodic SAM.

Use a precomputed bank of all 17x17 valid 16x16 masks and preallocated GPU
buffers so the timed path performs one dedicated RNG fill, small index
arithmetic, one mask gather, and one in-place multiply. No Python image loop,
CPU transform, data transport, target mixing, or full model pass is added.

## Motivation and Evidence

EXP-011 is the parent and global best at 95.61%. Its 16-checkpoint late EMA
plateau averages 95.493, ends at 95.46, and spans 0.17 points. A child must
reach at least 95.71 formally and should lift the stable tail by roughly
0.20-0.30 points rather than merely produce another selected maximum.

The current system has memory and sparse-operation headroom but little room for
extra model passes. EXP-011 retains 25,798 steps and uses 1,222.4 MiB; its EMA
updates are effectively free. In contrast, CPU dual-view augmentation and
multi-launch SE failed preflight. GPU Cutout operates on the already-transferred
single view and touches only the input tensor on the parent clean branch.

`experiments/012/papers/ricap.md` reports direct WideResNet/CIFAR evidence:
16x16 Cutout reduced error from 3.89% to 3.08%, while unstructured input dropout
worsened it to 4.69%. The comparison supports coherent spatial removal, not
generic input noise. RICAP itself combines four images and area-soft labels,
which overlaps the validated CutMix mechanism. Conditional Cutout is more
differentiated: it preserves all CutMix exposure and applies hard-label spatial
occlusion only where the parent would otherwise use a clean early image.

The literature effect ceiling (+0.81 points in its controlled WRN setting) is
large enough to clear 95.71, but transfer is uncertain. EXP-011 already combines
CutMix, drop path, SAM, and EMA; CutMix itself contains an occlusion component.
The proposal should therefore be judged as a fixed complementary dose, not an
expectation that the full published gain transfers.

## Fixed Mechanism

Add constants only:

```python
CUTOUT_SIZE = 16
CUTOUT_END = CUTMIX_END  # 0.75
CUTOUT_SEED = 43
```

There is no additional Cutout probability. Every early non-CutMix batch is
Cutout, and every CutMix batch remains exactly the parent operation. This makes
the mechanism and dose auditable without searching a second gate.

### Geometry

For each image, independently sample integer top-left coordinates
`top_y, top_x` uniformly from `[0, 16]`. Zero exactly rows
`top_y:top_y+16` and columns `top_x:top_x+16`, so every mask removes exactly
256 of 1,024 pixels (25%). Unlike center sampling with edge clipping, severity
does not depend on position.

The existing input normalization subtracts the CIFAR mean with unit standard
deviation. Setting normalized pixels to zero therefore fills the square with
the dataset mean, avoiding a raw-black color artifact. Targets remain ordinary
hard labels.

### Precomputed GPU mask bank

Before charged training, create a Boolean bank of shape `[289, 1, 32, 32]` in
top-left row-major order. It can be constructed without RNG using coordinate
comparisons:

```python
coords = torch.arange(32, device=device)
tops = torch.arange(17, device=device)
inside = (coords[None, :] >= tops[:, None]) & (
    coords[None, :] < tops[:, None] + CUTOUT_SIZE
)
mask_bank = ~(inside[:, None, :, None] & inside[None, :, None, :])
mask_bank = mask_bank.reshape(17 * 17, 1, 32, 32)
```

Assert every bank entry contains exactly 256 false and 768 true pixels. Bank
construction is deterministic setup, not training, and adds about 0.3 MiB.

Preallocate on GPU:

- `cutout_positions`: int64 `[256, 2]`;
- `cutout_indices`: int64 `[256]`;
- `cutout_selected_masks`: bool `[256, 1, 32, 32]`.

Create `cutout_cuda_generator = torch.Generator(device=device).manual_seed(43)`.
The DataLoader uses `drop_last=True`, so every training batch has 256 examples.

### Timed helper

The helper must use `out=` or in-place operations with the preallocated buffers:

```python
def cutout_batch(inputs, mask_bank, positions, indices, selected, generator):
    if inputs.shape[0] != BATCH_SIZE:
        raise RuntimeError("Cutout requires the dropped-last batch size")
    torch.randint(
        0, 17, positions.shape, out=positions,
        device=inputs.device, generator=generator,
    )
    torch.mul(positions[:, 0], 17, out=indices)
    indices.add_(positions[:, 1])
    torch.index_select(mask_bank, 0, indices, out=selected)
    inputs.mul_(selected)
    return inputs
```

If the exact installed PyTorch signature rejects `device` together with
`out=`, omit `device` because the output tensor already fixes it; do not replace
the helper with a per-image loop. Validate `out=` behavior in the smoke test.

The in-place multiply is safe: inputs have already reached GPU, no second view
needs the unmasked tensor, and targets are unchanged. The selected Boolean mask
broadcasts over three channels while preserving the input's channels-last
memory format.

## Integration with the Parent Branch

Retain the existing CutMix decision first:

```python
if progress < CUTMIX_END:
    cutmix_eligible_batches += 1
    apply_cutmix = parent_cutmix_gate_draw()
    if apply_cutmix:
        inputs, targets_a, targets_b, adjusted_lam, _ = cutmix_batch(...)
        cutmix_applied_batches += 1
    else:
        inputs = cutout_batch(...)
        cutout_applied_batches += 1
```

Do not draw Cutout coordinates on selected CutMix batches or after progress
0.75. Preserve the existing CutMix CPU gate, lambda/center, CUDA permutation,
area correction, and loss exactly. `targets_b is None` continues to select the
hard-label cross-entropy path for Cutout.

Do not alter random crop/flip, model architecture, batch size, optimizer, LR,
weight decay, drop-path rates, SAM, EMA, evaluation swap, or summary keys. Add
only Cutout config and audit fields.

## RNG and Parent-Preservation Contract

The seed-43 Cutout generator is independent of:

- global CPU RNG used by DataLoader/crop/flip;
- global CUDA RNG used by six drop-path masks;
- seed-42 CutMix CPU generator used for gate/geometry;
- seed-42 CutMix CUDA generator used for permutation;
- SAM's captured/replayed global CUDA state.

Mask-bank and buffer allocation consume no RNG. For every shared batch prefix,
CutMix decisions and specs must match the parent. Cutout draws occur only after
the parent gate says clean and cannot advance any parent stream. The model
topology and initialization are unchanged, avoiding the shape-dependent
initialization confound found in EXP-010.

Cutout stops before SAM starts, so neither SAM forward ever sees a Cutout batch.
CUDA state capture/replay, second-pass BN suppression, rho 0.05, and every-even
step cadence remain unchanged. EMA still starts at 0.75, samples every 31st
post-optimizer state, averages full floating state with an 18.75-second
half-life, copies integer buffers, and performs one exact evaluation swap per
epoch. Cadence 31 remains coprime to period-two SAM.

The only unavoidable parent-dose difference is throughput: charged Cutout work
can reduce early steps, shifting the step number at progress 0.75 and the total
number of late SAM/EMA samples. The parent-relative preflight bounds this, and
the full audit reports exact phase counts.

## Timing and Memory Estimate

The selected mask is 256x1x32x32 Boolean (0.25 MiB); the bank is about 0.28
MiB; position/index buffers are negligible. The input multiply touches about
0.75M float elements per Cutout batch. There is no parameter, gradient,
optimizer, SAM-shadow, or EMA-shadow increase.

EXP-011 applies CutMix to 49.60% of its 20,857 early eligible batches. Cutout
would therefore run on roughly 10,512 batches, about 40.7% of all 25,798 steps.
Even a small per-Cutout launch penalty is visible at that frequency. The helper
uses roughly four GPU operations (RNG, index arithmetic, gather, multiply), so
nominal arithmetic is cheap but kernel launch and mask gathering require
measurement.

Expected peak VRAM remains near 1,223 MiB. Expected charged overhead is 1-3%,
projecting approximately 25,050-25,540 total steps if kernels behave normally.
The final number of evaluations may fall by a few epochs, but total runtime
should remain below 600 seconds.

## Mandatory Parent-Relative Feasibility Gate

On physical GPU 0, compare the actual EXP-011 parent and Cutout candidate in
the same BF16/channels-last harness without compilation. Use alternating,
randomized-order paired rounds after warmup.

1. Verify the helper alone for at least 1,000 calls: deterministic replay from
   seed 43, exact 16x16 zero area per image, coordinate range, no input-layout
   change, no parent RNG-state change, and no allocation growth after warmup.
2. Measure at least 500 synchronized clean parent steps and 500 otherwise
   identical Cutout steps including forward/backward/Nesterov update.
3. Measure production-faithful SAM and EMA paths unchanged to ensure the added
   generator/buffers do not affect their parity or state audits.
4. Compute full-run weighted latency with Cutout on 40.7% of steps and parent
   latency on the remaining 59.3%; report paired median and p90 ratios.

Proceed only if:

- weighted median latency is at most `1.03 * parent median`;
- weighted p90 latency is at most `1.05 * parent p90`;
- `25,798 / median_ratio >= 25,000` projected steps;
- parent CutMix/drop-path/SAM/EMA RNG and restore parity checks all pass;
- projected total runtime is below 600 seconds and peak VRAM below 1.25 GiB.

These are same-harness parent-relative gates. No absolute images/second floor
may reject a valid measured parent. If the fixed vectorized helper fails, reject
the proposal before a metric run; do not switch to shared masks or lower dose
based on prospective accuracy.

## Instrumentation and Audit

Add a terminal line containing:

- early eligible, CutMix, and complementary Cutout batch counts;
- `cutout_images = cutout_batches * 256`;
- exact `masked_pixels = cutout_images * 256` and mask fraction 0.25;
- Cutout share of early eligible and all optimizer steps;
- seed, size, coordinate support, and late-call count (must be zero).

Require `cutmix_applied + cutout_applied == cutmix_eligible` at exit. CutMix
ratio should remain near 0.5 and Cutout should be its exact complement. Preserve
the parent's SAM, EMA, evaluation-source, swap/restore, BN, RNG, distance, and
timing audit lines.

Before deleting the transient log, durably transcribe the full metric summary,
CutMix/Cutout/SAM/EMA counts, last-16 EMA accuracies with mean/range, and all
preflight ratios. This addresses the lineage's transient-evidence failure mode.

## Expected Effect and 95.71 Assessment

Formal improvement over EXP-011 requires `best_test_acc >= 95.71%`. The fixed
hypothesis is:

> Complementary 16x16 GPU Cutout will reach best accuracy of 95.75-96.00%,
> retain at least 25,000 optimizer steps, and raise the last-16 EMA mean from
> 95.493 to at least 95.69 without changing parent CutMix/SAM/EMA semantics.

It can clear 95.71 because direct WideResNet/CIFAR evidence shows an 0.81-point
Cutout effect ceiling, the candidate applies a substantial but differentiated
dose to the parent's clean early half, and the clean SAM/EMA tail can consolidate
features learned under occlusion. It may fail because CutMix already supplies
patch occlusion, drop path adds strong early regularization, and replacing every
early clean batch removes unoccluded supervision until progress 0.75. The
stable improvement expectation is therefore about 0.20-0.40 points, not the
paper's full 0.81.

Run one seed-42 parent configuration plus fixed seed-43 Cutout stream. Below
95.71 is a tree no-improvement. A best >=95.71 with last-16 EMA mean below
95.59 (less than +0.10 over the parent plateau) is a selected-maximum success
but falsifies the stable-generalization claim. Do not tune size, fill value,
conditional probability, seed, or cutoff after observing the test metric.

## Risks

- **Redundant occlusion:** CutMix already forces partial-evidence recognition;
  Cutout may add little new invariance.
- **No clean early batches:** every early batch is either CutMix or Cutout,
  potentially over-regularizing a recipe that also uses full drop path.
- **CIFAR severity:** an exact 16x16 square removes 25% of a small image and can
  erase most of the object after random crop.
- **Mean-fill boundary cues:** a constant square can become an artificial
  feature despite being less color-biased than raw black.
- **Hard-label ambiguity:** unlike CutMix/RICAP, an occluded image keeps its full
  label even when discriminative content is removed.
- **Kernel overhead:** four small operations on 40% of steps can lose enough
  image exposure and evaluations to offset accuracy gains.
- **EMA attribution:** a child metric measures Cutout plus the existing EMA
  selection protocol; last-16 mean and final accuracy are needed beside best.
- **Wall-clock dose drift:** added early cost changes the exact transition step,
  later SAM count, and EMA sample count even with fixed seeds.
- **Protocol noise:** tail variation is 0.17 and historical selected-run noise
  reaches 0.29 points, so a bare +0.10 best delta is weak evidence.

## Verification

1. Unit-test all 289 masks: shape, Boolean type, exactly 256 zeros, unique
   top-left geometry, and correct row-major index mapping.
2. For a patterned batch, verify each sampled square is exactly zero across all
   channels, all outside pixels are bit-identical, targets are unchanged, and
   channels-last layout is preserved.
3. Verify same seed-43 sequences reproduce positions and different images have
   independent positions, while CPU/CUDA global, CutMix CPU/CUDA, and DataLoader
   RNG states remain unchanged.
4. Verify selected CutMix batches are bit-identical to parent and consume no
   Cutout draw; non-CutMix early batches always Cutout; late batches never do.
5. On full WRN BF16, verify finite forward/loss/backward and no interaction with
   six drop-path masks, SAM replay/BN suppression/restore, or EMA update/swap.
6. Pass the parent-relative GPU-0 feasibility gate above.
7. Launch once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
8. Confirm physical GPU 0 is the 97,871 MiB H20, charged time approximately
   300 seconds, total below 600, one evaluation per epoch, at least 25,000
   steps, exact complementary counts, zero parent/EMA audit failures, complete
   summary, and both best and plateau hypotheses.
9. Verify only `train.py` changed, no evaluator/dependency/seed reroll occurred,
   and durable evidence is recorded before transient cleanup.

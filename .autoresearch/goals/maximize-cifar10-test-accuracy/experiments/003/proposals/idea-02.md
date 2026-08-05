# Proposal: Early-Only CutMix With an Area-Corrected Hard-Label Tail

## Recommendation

Replace the validated mixup branch during the first 65% of counted training
time with a single-rectangle, batchwise CutMix intervention, then retain the
existing final 35% hard-label cosine tail unchanged. Use `Beta(1.0, 1.0)` for
the requested retained-area coefficient, one permutation and one shared
rectangle per batch, and correct the loss coefficient from the rectangle's
actual clipped area. Do not combine CutMix and mixup in this experiment.

This is a controlled test of whether spatially local sample composition is a
better inductive bias than whole-image linear interpolation for the current
WRN. Architecture, loader, crop/flip augmentation, optimizer, LR schedule,
weight decay, seed, evaluation cadence, and the `0.65` removal point remain
identical to EXP-002. The only conceptual variable is the early regularizer.

## Limiter Diagnosis

EXP-002 reached 94.07%, with final accuracy equal to best accuracy, after 141.9
data passes. Its accuracy continued to improve in the hard-label tail and the
tail reached near-zero training loss. The remaining gap therefore looks more
like representation/generalization error than insufficient late optimization.
Early mixup already established that strong early regularization followed by
clean-label refinement is effective. CutMix tests a narrower hypothesis:
preserving natural pixels and local texture while disrupting object extent and
context may teach more useful CIFAR-10 features than interpolating every pixel.

The saved evidence directly supports convex label mixing and early removal of
regularization, but it does not directly establish CutMix or its best
hyperparameters. This proposal is consequently a medium-confidence mechanism
transfer, not a literature-backed expectation that CutMix must beat mixup.

## Exact Algorithm

Add `CUTMIX_ALPHA = 1.0` and retain the existing
`MIXUP_END_FRACTION = 0.65` under a renamed general cutoff such as
`REGULARIZATION_END_FRACTION`. At each batch whose pre-step counted progress is
below `0.65`:

1. Draw one scalar `lambda_requested ~ Beta(1.0, 1.0)` on the GPU and one
   device-local random permutation over the batch.
2. Interpret `lambda_requested` as the nominal fraction of the original image
   to retain. Set the nominal patch side ratio to
   `sqrt(1 - lambda_requested)` and convert it to integer width and height for
   the fixed `32 x 32` input.
3. Draw one patch center uniformly over `[0, 31]` for each axis. Clip the
   resulting bounds to the image. Use this one rectangle for the whole batch;
   sharing it minimizes scalar synchronization and indexing overhead while the
   shuffled donor image remains different for each example.
4. Replace that rectangular region in every image with the corresponding
   region from `inputs[permutation]`. The right-hand-side donor patch must be
   materialized before the in-place write (or assigned into an `inputs.clone()`)
   so permutation cycles cannot read already-modified pixels.
5. Compute the coefficient from the region actually pasted after clipping:

   ```python
   pasted_area = (x2 - x1) * (y2 - y1)
   lambda_effective = 1.0 - pasted_area / (inputs.size(-1) * inputs.size(-2))
   loss = (
       lambda_effective * F.cross_entropy(outputs, targets)
       + (1.0 - lambda_effective)
       * F.cross_entropy(outputs, targets[permutation])
   )
   ```

   The area correction is mandatory: center clipping usually makes the pasted
   rectangle smaller than its nominal size, so using `lambda_requested` would
   assign too much target mass to the donor class. If integer rounding yields
   zero pasted area, use the ordinary hard-label loss for that batch.

At progress `>= 0.65`, skip beta sampling, permutation, rectangle generation,
and patch assignment, and execute the current hard-label path exactly. Emit one
transition log containing counted time, progress, epoch, step, LR, and the
cumulative mean pasted-area fraction. Keep crop and horizontal flip active in
both phases; "hard-label tail" means only that sample/target mixing is removed.

## Why This Schedule and Strength

The 65% cutoff is inherited unchanged from the successful EXP-002 result. It
leaves approximately 105 counted seconds for the already-validated hard-label
refinement phase and prevents a schedule change from obscuring the CutMix
comparison. A uniform beta distribution is a deliberate first CutMix setting:
it covers small through large rectangles without the near-all-or-nothing mass
of `Beta(0.2, 0.2)`, making the intervention meaningfully spatial across more
batches. The exact realized label strength is governed by pasted area after
clipping, not the raw sample.

CutMix replaces mixup rather than augmenting it. Alternating or stacking the
two would change both regularizer diversity and average target softness and
would make a positive or negative result hard to interpret. If CutMix wins,
later work can test a fixed probability mixture; if it loses, EXP-002 remains a
clean baseline.

## Feasibility and Budget

The implementation uses only existing PyTorch operations in `train.py`. It
adds one beta sample, one `randperm`, four small coordinate samples, a donor
patch gather, and a patch assignment per early batch. It requires no extra
forward pass, dependency, CPU transform, or evaluator call. A full input clone,
if used for alias safety, is the same tensor size order as EXP-002's dense
mixup output; copying only the donor patch may be cheaper. Peak VRAM should
remain close to EXP-002's 1,094 MiB.

All intervention overhead is naturally charged to the 300-second training
timer. Before the full run, compare a short matched smoke projection against
the current EXP-002 path and require at least 95% of its projected data passes.
The full run should remain near EXP-002's 141.9 passes and 341.2 total seconds,
comfortably below 10 minutes. A drop below 95% exposure makes accuracy harder
to attribute and should be treated as an implementation-efficiency failure.

## Expected Impact and Evidence Limits

The working hypothesis is that local compositing will retain crisp, in-domain
pixels while discouraging reliance on one contiguous discriminative region,
improving the representation before the clean tail fits final margins. A
realistic successful outcome is a modest `+0.10` to `+0.35` percentage-point
gain over 94.07%, not another EXP-002-sized jump. The required improvement
threshold is therefore **94.17%**.

Confidence is medium-low. The local evidence says early regularization and
mixed targets can help, but neither supplied paper summary evaluates CutMix.
CIFAR-10 images are only `32 x 32`; clipped rectangles can remove most of a
small object, and area-proportional labels assume object evidence is spatially
uniform. Those assumptions may be worse than mixup's smooth interpolation.
One fixed-seed run can establish a benchmark improvement under the project's
rule, but a marginal `0.10-0.20` point gain should not be interpreted as a
general CutMix superiority claim without replication.

## Risks and Diagnostic Interpretation

- **Label noise from object geometry:** Area fraction may poorly approximate
  semantic contribution when a small rectangle covers the whole object. If
  throughput is preserved but both best and final accuracy fall, spatial label
  mismatch is the leading explanation; revert to early mixup rather than
  increasing CutMix strength.
- **Clipping bias:** Sampling centers at image edges shrinks the pasted area.
  Recompute `lambda_effective` from exact integer bounds and log its cumulative
  mean. Never use the uncorrected beta sample in the loss.
- **Source aliasing:** In-place replacement can corrupt donor pixels when a
  permutation contains cycles. Materialize the full donor patch before writing
  or use a cloned destination, then verify with a deterministic toy batch.
- **Over-regularization:** `Beta(1,1)` includes large patches. The unchanged 35%
  hard tail is the main recovery mechanism. A curve that is weak at the switch
  but catches up rapidly suggests a milder or shorter CutMix follow-up.
- **Throughput regression:** Scalar `.item()` calls or per-example Python loops
  could reduce exposure. Use one shared rectangle, tensor indexing, and no
  per-sample loop; apply the matched 95% smoke gate.
- **Confounded evaluation:** Keep seed 42 and exactly the existing every-fifth-
  epoch plus terminal evaluation. Do not inspect extra validation points or
  tune rectangle parameters during the run.

## Verification

1. Unit-smoke a synthetic `4 x 3 x 32 x 32` batch with identifiable constant
   images. Confirm pixels outside the rectangle remain from the original,
   pixels inside come from the intended permuted donor, the donor source is not
   corrupted, and `lambda_effective` exactly equals `1 - pasted_area / 1024`.
2. Confirm the finite-loss guard covers both CutMix and hard-label paths and a
   zero-area rectangle reduces to ordinary cross-entropy.
3. Check a matched throughput smoke run achieves at least 95% of EXP-002's
   projected data passes without per-example host loops or device syncs.
4. Run once on one H20 with `uv run train.py > run.log 2>&1`; require a complete
   summary, approximately 300 counted training seconds, and under 600 seconds
   total wall time.
5. Confirm exactly one transition near 195 counted seconds, no CutMix operations
   after it, validation at most once per epoch, seed 42, and only `train.py`
   modified.
6. Record best/final test accuracy and loss, steps, epochs, peak VRAM, realized
   data passes, transition time, and mean pasted-area fraction. Classify as an
   improvement only if `best_test_acc >= 94.17%` with all constraints passing.

## Evidence

- `experiments/002/04-analysis.md`: early alpha-0.2 mixup followed by a 35%
  hard-label tail improved 93.38% to 94.07%, with final equal to best and only
  2.8% less exposure than EXP-001.
- `knowledge/papers/mixup.md`: paired input/label interpolation improves CIFAR-10
  generalization and supplies the mixed-loss principle, but is not direct
  evidence for rectangular replacement.
- `knowledge/papers/time-matters-regularization.md`: regularization often exerts
  most of its value early and may be removed late, supporting the inherited
  65%/35% temporal structure.
- `03-experiment-learnings.md`: retain the WRN-16-2, time-aligned cosine
  schedule, persistent workers, sparse evaluation, and temporal regularization
  removal as established patterns.

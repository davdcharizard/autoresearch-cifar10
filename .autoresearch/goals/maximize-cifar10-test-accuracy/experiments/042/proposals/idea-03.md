# Proposal: Centered 2x2 Spatial-Contrast Residual Readout

## Summary

The accepted model globally averages its final `128 x 8 x 8` feature map before
the successful `128 -> 64 -> 128` pooled residual head. Add one deliberately
small readout that retains only the final map's coarse 2x2 layout:

```python
feature_map = F.relu(self.bn(out))
pooled_map = F.adaptive_avg_pool2d(feature_map, 1)
pooled = pooled_map.flatten(1)
coarse = F.adaptive_avg_pool2d(feature_map, 2) - pooled_map
spatial_correction = self.spatial_projection(coarse.flatten(1))
refined = (
    pooled
    + POOLED_HEAD_SCALE * self.pooled_head(pooled)
    + SPATIAL_HEAD_SCALE * spatial_correction
)
return self.fc(refined)
```

`spatial_projection` is exactly one bias-free `Linear(4 * 128, 128)`. The
four quadrant means are centered by the global mean channel by channel before
flattening. Thus a spatially constant feature map, or any channelwise offset
common to all four quadrants, produces exactly zero spatial correction. The
new route cannot duplicate the accepted global-average statistic; it exposes
only three independent coarse spatial contrasts per channel while leaving the
accepted global path and nonlinear pooled head intact.

Use a fixed residual scale `SPATIAL_HEAD_SCALE = 0.1`, reusing the only locally
validated residual-readout scale rather than introducing a new amplitude
sweep. Initialize the projection with Kaiming normal under one prospectively
fixed, restoring CPU RNG fork, `SPATIAL_HEAD_INIT_SEED = 42042`. There is no
bias, activation, normalization, learned gain, auxiliary loss, positional
embedding, or other spatial branch.

The hypothesis is that global average pooling discards coarse arrangement that
still helps distinguish a small fraction of CIFAR-10 images. The accepted
augmentation stack may have made such evidence robust enough that a low-cost,
subordinate quadrant-contrast correction improves the final boundary without
spending counted time on another 8x8 residual transform. This is exploratory:
there is no classwise error analysis diagnosing lost layout, and fixed
quadrants partially oppose translation invariance. It merits at most one
strictly preregistered score.

## Exact Capacity and Compute

- The projection contains exactly `512 * 128 = 65,536` trainable parameters.
  The accepted 1,003,482-parameter model therefore becomes 1,069,018
  parameters, a 6.53% increase.
- It performs 65,536 multiply-accumulates per image. The accepted spatial
  convolutions perform approximately 119,980,032 MACs/image, so the new GEMM
  is only 0.0546% of spatial convolution MACs. Computing a second adaptive
  pool also performs about 8k reductions/image; kernel launches and backward,
  not arithmetic totals, remain the credible H20 cost risk.
- Although 512 stored input weights feed each output, centering constrains the
  input to a 384-dimensional contrast subspace. Keeping the direct flattened
  2x2 representation makes ordering and verification transparent; the
  redundant common-quadrant weight direction has no data gradient and simply
  remains under ordinary matrix decay.
- The branch is evaluated only once per normal forward and adds no training
  state beyond its ordinary SGD momentum buffer. It does not alter the 8x8
  backbone, accepted pooled MLP, classifier, evaluator cadence, or data path.

Static MACs are not a feasibility verdict on this H20. Prior experiments show
that small kernels can be launch-bound and equal MAC layouts can have different
training speed. The balanced complete-update timing gate below decides whether
the treatment preserves the required exposure.

## Evidence and Relationship to Prior Results

- EXP-036 is the current 94.48% frontier: a bias-free scale-0.1
  `128 -> 64 -> 128` residual MLP after global pooling improved accuracy by
  0.16 points, improved final loss to 0.2456, and retained 130.304 passes. It
  proves that cheap readout capacity can improve this representation. It does
  **not** prove that information discarded by pooling is limiting. This
  proposal retains that exact head and adds an orthogonal input statistic.
- EXP-012's `128 -> 64 -> 64 -> 128` convolutional bottleneck on the full 8x8
  map lost 0.33 points at 135.49 passes and cost 3,407,872 MACs/image. The
  proposed readout is materially different: it never transforms or constrains
  the spatial map, preserves the entire accepted direct path, and consumes
  about 52 times fewer projection MACs. Nonetheless, EXP-012 is strong warning
  that merely adding late spatial capacity can worsen generalization.
- EXP-010/011 found small standalone signals from selective late width and an
  extra stage-3 block; EXP-027 then showed that early N1/M5 RandAugment makes
  the extra block genuinely useful. That interaction is the positive rationale
  for asking whether the accepted crop/flip/RandAugment representation can use
  a little spatial readout capacity. It is not evidence that any capacity
  addition will help.
- EXP-017's two full stage-3 SE gates gave a small positive signal, while
  EXP-018/019/024/025 established that removing a placement, input dependence,
  or global cross-channel mixing destroys it or that full gates miss the
  protected exposure regime. This candidate retains dense cross-channel mixing
  but is additive and post-spatial; it neither gates residual branches nor
  attenuates accepted channels. A success would support coarse spatial
  contrasts, not rescue or validate the closed SE family.
- The accepted crop with four-pixel padding, horizontal flip, and early
  RandAugment intentionally teaches invariance to position and transformations.
  A fixed 2x2 grid is therefore a real inductive-bias conflict: translations
  near quadrant boundaries can change the correction and arbitrary projection
  weights are not flip-equivariant. Centering, the direct accepted route, the
  scale-0.1 residual form, and the full clean tail limit the damage, while the
  augmentations can train the projection to ignore unstable contrasts. They do
  not guarantee that it will. Coarse final-map receptive fields make the branch
  less pixel-aligned than an input-space grid, but this remains the proposal's
  largest accuracy risk.

No network or remote evidence is required or used. This proposal is grounded
in the accepted source, `02-system-understanding.md`, and the recorded EXP-012,
EXP-017--025, EXP-027, and EXP-036 results.

## Exact Production Change

Add only these constants alongside the accepted pooled-head constants:

```python
SPATIAL_HEAD_SCALE = 0.1
SPATIAL_HEAD_INIT_SEED = 42042
```

After the accepted `self.apply(self._weights_init)` and after constructing the
accepted pooled head exactly as at commit `a7c42dc`, construct the new module in
its own restoring fork:

```python
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(SPATIAL_HEAD_INIT_SEED)
    self.spatial_projection = nn.Linear(
        4 * widths[2], widths[2], bias=False
    )
    init.kaiming_normal_(self.spatial_projection.weight)
```

This order is mandatory. All accepted parameters and buffers, including both
pooled-head matrices and `fc`, must remain byte-identical, and the global CPU
RNG state after construction must remain exact. Seed 42042 is fixed before any
timing or score, is used once, and has no alternative; it initializes a new
tensor rather than rerolling the experiment seed.

In `forward`, preserve the accepted backbone and final `BN -> ReLU`. Compute
the accepted 1x1 pooled map first. Compute the 2x2 pool from the same post-ReLU
feature map, subtract the broadcast 1x1 map, flatten in PyTorch's native
contiguous `N,C,H,W` order, and apply the projection. Add the correction to the
accepted refined pooled vector at fixed scale 0.1 immediately before the
accepted classifier. Do not detach any tensor or mutate it in place.

The existing optimizer comprehension must place the sole new 2D weight exactly
once in `decay_params` at `WEIGHT_DECAY = 5e-4`, with the accepted LR,
momentum, and Nesterov settings. There is no head-specific decay/LR, no
zero-decay exception, and no schedule or training-loop change.

Everything else stays byte-for-byte accepted where applicable: seed 42,
batch 256, FP32, `(2,2,3)` stages, early worker-private RandAugment N1/M5,
batch-shared alpha-0.2 mixup through 65%, exhausted-iterator augmentation
transition, 0.2-to-0.002 global time cosine, continuous matrix decay, loader,
finite-loss guard, evaluator, and once-per-epoch evaluation cadence.

## Preregistered Semantic and RNG Preflight

Create an ignored, evaluator-free `preflight.py`. It must not read test data,
modify `prepare.py`, alter tracked code after gates begin, or consume the sole
score. Fail closed before timing if any invariant fails:

1. Diff candidate production against `git show a7c42dc:train.py`. Require the
   change to be limited to the two constants, isolated branch construction,
   topology logging, and the final pooling/readout lines. Hash `prepare.py` and
   reject any training schedule, data, loss, optimizer, or evaluator change.
2. Instantiate accepted and candidate models from cloned initial CPU RNG
   states. Require every one of the 1,003,482 common parameters and buffers to
   have identical name, shape, dtype, and bytes. Require identical global CPU
   RNG state after construction and independently reconstruct seed 42042's
   Kaiming matrix byte-for-byte. Assert no constructor consumes CUDA RNG.
3. Assert exact topology `Linear(512,128,bias=False)`, scale 0.1, 65,536 new
   parameters, 1,069,018 total parameters, and no other new module, parameter,
   or buffer. Inspect optimizer groups by identity: every trainable tensor
   occurs once, the new weight is in the 5e-4 matrix-decay group, and accepted
   group membership and optimizer options are exact.
4. On deterministic finite FP32 tensors of shapes `[256,128,8,8]` and
   `[16,128,8,8]`, independently calculate the four `4x4` quadrant means,
   their channelwise global mean, native flatten order, projection, and final
   residual expression. Match the production intermediate tensors and logits
   within a preregistered FP32 tolerance. Include batch 16 because it is the
   frozen evaluator's final partial batch.
5. Verify algebraic semantics. A spatially constant map must yield zero
   centered quadrants and zero spatial correction. Each centered channel's
   four tile values must sum to zero within FP32 reduction tolerance. A
   synthetic one-quadrant impulse must land in the expected flattened slice.
   Permuting quadrants must change a deliberately non-symmetric oracle
   projection, proving the branch has not accidentally collapsed back to GAP.
6. Temporarily zero only a cloned candidate's spatial weight and require its
   logits to equal accepted logits exactly or within an explicitly justified
   one-operation FP32 tolerance; restoring the fixed initialized weight must
   produce finite, nonzero logit and representation perturbations. Record
   branch/direct representation RMS and logit RMS deltas as diagnostics only;
   do not tune or reject the fixed scale/seed from their favorable or
   unfavorable values unless they are nonfinite.
7. Prove forward consumes no CPU or CUDA RNG. From cloned stream states,
   accepted and candidate must subsequently draw identical mixup lambdas,
   permutations, crop/flip decisions, and post-forward random samples.
8. Run independent autograd oracles. The new weight, classifier, pooled head,
   and backbone must all receive finite nonzero gradients on deterministic
   mixup and hard-label losses. For a branch-only scalar objective, the summed
   gradient over all 64 spatial sites must be approximately zero per channel,
   proving the new route differentiates quadrant contrasts rather than adding
   a hidden global-mean path.
9. Execute one production-equivalent early-mixup update and one hard-label
   update from cloned candidate states. Require the new projection and accepted
   trainable tensors to move where gradients are nonzero, every momentum buffer
   to be finite, and the new weight's update to match an independent Nesterov
   plus coupled-decay calculation. Do not special-case its optimizer state.
10. Statically and dynamically re-prove the accepted alpha-0.2 batch-shared
    loss, 65% cutoff, worker-private RandAugment and exhausted-epoch cutoff,
    LR samples, seed 42, time accounting, evaluation uniqueness, and
    600-second outer timeout contract. Guard all evaluator and test-data access.

A semantic failure may be repaired only if it is an implementation or verifier
error while preserving this exact treatment. Changing pooling, centering,
scale, seed, topology, initialization, or optimizer semantics creates a new
experiment and is forbidden after preflight begins.

## Preregistered H20 Timing Gate

After semantic checks, use one NVIDIA H20 in a disposable evaluator-free
process. Benchmark accepted and treatment **complete production update bodies**
for both early mixup and hard-label regimes: zero-grad, optional batch-shared
mixup/permutation, forward, exact CE construction, backward, SGD/Nesterov step,
and synchronization. Use identical resident FP32 batches, common starting
weights, optimizer state, and cloned RNG streams.

- Warm each model/regime for at least 20 updates before measurement.
- Measure at least three balanced, interleaved windows of at least 50 updates
  per model/regime, reversing order across windows to reduce thermal drift.
- Use CUDA events with synchronization only at window boundaries.
- Require population CV at most 5% for all four series, finite loss/gradients/
  state, treatment peak allocation below 2,048 MiB, and no persistent drift or
  single-window outlier greater than 10% from its series median.

Let `a_mix`, `a_hard`, `t_mix`, and `t_hard` be median seconds/update. Compute:

```text
retention = (0.65 / t_mix + 0.35 / t_hard) \
          / (0.65 / a_mix + 0.35 / a_hard)
projected_passes = 130.304 * retention
```

Proceed to scoring only if `retention >= 0.97465`, projected passes are at
least 127.0, and the unchanged accepted-loader/evaluation cadence still
projects under 600 seconds wall. The numeric retention threshold is the
127/130.304 protected exposure ratio rounded upward. Timing instability gets
one diagnostic pass over the harness; a stable topology miss is not rerun and
must not be rescued with another projection shape or scale.

## Sole Scored Run and Verdict

If and only if every preflight gate passes, remove stale `run.log` and execute
exactly once on one confirmed H20:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Do not rerun a valid score, alter seed 42, inspect test examples or classwise
errors, add evaluation, or tune from intermediate test results. Audit exit
zero, one finite summary, 300 counted seconds, less than 600 wall seconds,
exactly 1,069,018 parameters, at least 127.0 realized data passes, one mixup
transition, one exhausted-iterator RandAugment transition, and no duplicate or
more-than-once-per-epoch evaluations.

- **Improvement:** accept only if `best_test_acc >= 94.58%`, the required
  +0.10 points over accepted 94.48%, with all integrity gates satisfied.
- **Corroboration only:** `final_test_acc >= 94.45%` and
  `final_test_loss <= 0.2456` strengthen a positive interpretation but neither
  can rescue a primary miss nor veto a valid primary success.
- **No improvement:** any valid normal-exposure score below 94.58%, regardless
  of loss or an attractive intermediate evaluation.
- **Crash/invalid:** missing/nonfinite summary, protocol violation, under-127
  realized passes, or timeout. Fix only a genuine implementation defect before
  a valid score; never reroll a completed treatment trajectory.

## Risks, Interpretation, and Closure

- The main accuracy risk is position sensitivity. Fixed quadrants can overfit
  dataset centering, conflict with random crops/translations, and encode a
  horizontal orientation despite flip augmentation. Centering makes the route
  offset-invariant, not translation- or flip-invariant.
- The branch adds 65,536 weights to a learner that already nearly interpolates
  its hard-label tail. Its residual direct path and ordinary decay permit the
  optimizer to suppress it, but more capacity can still worsen test loss.
- `adaptive_avg_pool2d(..., 2)` is a lossy four-bin summary, not preservation of
  full spatial layout. A failure cannot establish that all intermediate
  spatial evidence is useless; a success would establish only that these
  centered quadrant contrasts help this exact accepted learner.
- Any gain may arise from additional linear capacity/conditioning rather than
  interpretable object-part geometry. Centering rules out simple duplication
  of GAP, but no score can uniquely attribute which learned contrast mattered.
- Kaiming seed 42042 and scale 0.1 define one trajectory. They are prospective
  controls, not optimized values, and must not be rescued after the result.

A valid >=127-pass miss closes the exact centered `2x2 -> flatten ->
Linear(512,128,bias=False)` residual readout at scale 0.1 with Kaiming seed
42042, plus immediate seed, scale, bias, and uncentered-flatten rescues absent a
new independent diagnosis. It does **not** close translation-equivariant
spatial pooling, learned attention pooling, higher-resolution summaries,
nonlinear spatial heads, or intermediate supervision. A timing miss closes
only this exact execution topology under the protected exposure regime. A
valid success can be accepted unchanged as the new baseline.

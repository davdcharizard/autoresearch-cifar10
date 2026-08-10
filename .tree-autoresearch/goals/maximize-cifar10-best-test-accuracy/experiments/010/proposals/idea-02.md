# Proposal: PreAct WRN-22-4 with the EXP-002 CutMix Recipe

## Summary

Grow EXP-002's PreAct WRN-16-4 from two to three residual blocks per stage while retaining widths 64/128/256 and the entire validated training recipe. The candidate is the standard `6n+4` CIFAR depth with `n=3`: a PreAct WRN-22-4 containing nine two-convolution residual blocks. It changes only `train.py`, adds no auxiliary branches or small attention kernels, and preserves batch 256, BF16/channels-last, Nesterov SGD, time-normalized LR/drop-path schedules, front-loaded CutMix, dedicated CutMix RNG, one evaluation per epoch, and the fixed 300-second charged budget.

The intervention is deliberately simple but expensive. It adds six dense 3x3 convolutions on the sequential residual path. A parent-relative GPU-0 latency gate is therefore mandatory before any accuracy run.

## Limiter Diagnosis

EXP-002 reached 95.23% with 2.75M parameters, 27,950 updates, 144 epochs, and only 1,178.9 MiB peak allocation (`experiments/002/04-analysis.md`). Memory is plainly not limiting. The goal-wide bottleneck is a detectable generalization gain from an already strong recipe; sub-0.30-point effects are difficult to distinguish from observed run/tail variation (`02-system-understanding.md`; `03-experiment-learnings.md`).

Depth may improve the learned representation by inserting an additional nonlinear residual transformation at every spatial scale, extending the effective convolutional path and receptive-field composition without introducing a new operator family. Pre-activation and identity shortcuts make the deeper network optimization-safe in principle, while CutMix and linearly distributed drop path constrain its larger function class. However, EXP-001/002 already fit strongly, so raw parameter capacity is not the diagnosed limiter. The rationale is representational depth, not merely more weights.

The fixed-time cost is the strongest counterargument. EXP-009 showed that even sub-1% parameter attention can be rejected when multiple small paths add 20.7% latency. WRN-22-4 avoids those fragmented FP32 paths, but it deliberately violates the experiment-paper guidance to preserve the number of major convolutions: six large 3x3 kernels are added to the main dependency chain. Unlike launch-bound SE, these kernels should use H20 tensor cores efficiently, but their arithmetic cannot overlap away. Accuracy benefit must offset substantially fewer samples and optimizer updates.

The experiment-scoped papers support architecture topology as a meaningful capacity axis: ResNeXt shows depth/width/cardinality are not interchangeable, PyramidNet shows CIFAR accuracy depends on how capacity is distributed, and Gradually Updated Networks motivate increasing effective transformation depth without auxiliary attention (`experiments/010/papers/*.md`). None directly proves that WRN-22-4 beats WRN-16-4 under 300 seconds, so the proposal treats depth as an exploratory architecture hypothesis rather than a literature-established optimum.

## Exact Architecture

Keep `PreActWideBlock` unchanged. Replace the six-item `block_specs` with exactly:

```python
block_specs = [
    (16, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]
```

Only blocks 1, 4, and 7 in one-based indexing change shape; the latter two downsample. Those three blocks retain learned preactivated 1x1 projection shortcuts. The six identity-shape blocks use raw identity shortcuts. Stage outputs remain 64x32x32, 128x16x16, and 256x8x8. Stem, final BatchNorm/ReLU/global pool, and 256-to-10 classifier are unchanged.

Set a clear architecture label such as `PreAct WRN-22-4`, but do not rename metric keys or evaluator-facing APIs. Derive drop-path probabilities through the existing formula over `num_blocks=9`:

```text
drop_prob(block i) = 0.08 * i / 9, i=1..9
```

The maximum remains 0.08, and `drop_path_scale(progress)` still holds it through 75% then decays it to zero. This is the same depth-normalized stochastic-depth recipe, although individual shared blocks receive different probabilities because the architecture has more positions.

## Exact Parameter Count

Each added identity block contributes two 3x3 convolutions and two affine BatchNorm layers:

```text
64-channel block:   2*(64*64*3*3)   + 4*64  =    73,984
128-channel block:  2*(128*128*3*3) + 4*128 =   295,424
256-channel block:  2*(256*256*3*3) + 4*256 = 1,180,672
added total:                                        1,550,080
```

Expected candidate parameters are exactly:

```text
2,748,890 + 1,550,080 = 4,298,970
```

This is a 56.39% increase. Projection-shortcut count and classifier size do not change. Static inventory must assert 9 blocks, 19 main-path dense convolutions including the stem, 2 projection convolutions, and 4,298,970 trainable parameters.

## MACs, FLOPs, and Conv-Path Cost

Count one multiply-accumulate as one MAC and two arithmetic FLOPs. Ignoring minor BN/ReLU/pooling work, EXP-002 performs approximately 392,612,352 MACs per image. At each CIFAR stage, the added identity block has the same MAC count because doubling channels accompanies quartering spatial area:

```text
stage 64:  2 * 32*32 * 64*64   * 3*3 = 75,497,472 MACs
stage 128: 2 * 16*16 * 128*128 * 3*3 = 75,497,472 MACs
stage 256: 2 * 8*8   * 256*256 * 3*3 = 75,497,472 MACs
```

Candidate cost is therefore approximately 619,104,768 MACs or 1.238 GFLOPs per image, versus 0.785 GFLOPs for EXP-002. The MAC ratio is 1.5769 (+57.69%). At batch 256, that is about 158.49 billion MACs per training forward before backward, versus 100.51 billion for the parent.

The residual main path grows from 12 to 18 block 3x3 convolutions (+50%); including the stem, it grows from 13 to 19 sequential dense convolutions (+46.15%). Total convolution calls including two projection shortcuts grow from 15 to 21 (+40%). This critical-path increase is the binding architecture constraint even though every new operator is a familiar efficient convolution.

Using EXP-009's same-recipe parent median near 10.09 ms as context, a plausible production step is 14.6-15.9 ms (1.45-1.58x), depending on improved kernel utilization and fixed overhead. That projects approximately 17,700-19,300 updates, or 91-99 natural 195-batch epochs, versus EXP-002's 27,950 updates/144 epochs. The charged-time LR and CutMix cutoffs remain correctly aligned, but the model sees roughly one-third fewer crops, labels, and optimizer updates.

Parameters, gradients, and momentum add only about 18.6 MiB of persistent FP32 state. Additional saved activations/workspaces likely raise peak allocation into roughly the 1.4-1.8 GiB range, still below 2% of the 97,871 MiB H20. Measure rather than rely on this estimate; memory is not expected to gate the run.

## Training Recipe Preservation

Branch from EXP-002, not EXP-004. There is no SAM, EMA, ASAM, attention, extra augmentation, or altered sampler in this experiment.

Keep exactly:

- batch size 256, 195 dropped-last batches per natural epoch, and 50,000 independent-image stream;
- seed 42 and standard crop/flip/normalize transforms;
- BF16 autocast and channels-last model/input layout;
- SGD with peak LR 0.2, momentum 0.9, Nesterov, and weight decay `1e-4`;
- charged-time 5% warmup followed by cosine to LR 0.002;
- `MAX_DROP_PATH=0.08` and final-quarter decay;
- `CUTMIX_PROB=0.5`, alpha 1.0 uniform lambda, cutoff 0.75, clipped-area labels, and dedicated seed-42 CPU/CUDA generators;
- one frozen evaluation per epoch and all required summary fields.

All extra convolution work naturally remains between `t0` and CUDA synchronization, so it is charged. Do not compensate by increasing batch size, changing LR, reducing CutMix, extending the budget, or adding gradient accumulation. Those would create a different experiment.

The larger module consumes more initialization draws and more per-block drop-path draws than EXP-002. This is an unavoidable consequence of the genuine architecture change, not seed rerolling. Keep the single global seed and private CutMix generators fixed; do not search another initialization.

## Parent-Relative GPU-0 Preflight

Before any metric run, confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20. Run a production-faithful paired benchmark containing both the exact EXP-002 parent and WRN-22-4 candidate in one script/process. Use fixed synthetic batch-256 channels-last inputs and targets, BF16 autocast, the real loss, backward, Nesterov optimizer step, drop-path scale 1.0, and CUDA synchronization. Include the same fixed CutMix work in both or omit it from both; never compare mismatched paths.

Use at least 50 warmup steps and 300 measured steps per model, alternating parent/candidate measurement blocks to reduce clock/co-tenant drift. Reset gradients/optimizer state consistently, verify finite losses/gradients, and report median, p90, mean, dispersion, peak VRAM, and candidate/parent ratios. The parent must be measured in the same harness; no absolute throughput threshold may reject a candidate that the parent itself would fail, following the EXP-008 learning.

Proceed to the full run only if every fixed condition passes:

```text
candidate / parent median step latency <= 1.50
candidate / parent p90 step latency    <= 1.55
projected updates from 27,950 * parent_median / candidate_median >= 18,500
projected natural epochs >= 18,500 / 195 = 94.87
candidate peak allocated VRAM < 4,096 MiB
projected end-to-end time with one measured candidate Eval per epoch < 570 s
```

The median gate is the binding exposure gate; the p90 gate rejects unstable convolution algorithms. The VRAM limit is only a broad implementation-safety bound. Measure one warm full frozen evaluation for each model and use candidate latency plus projected epochs for the outer-time projection. Also require the parent measurement to be free of unstable co-tenant/thermal drift; if alternating parent medians differ by more than 7.5%, classify the preflight as contaminated and rerun the measurement only, without changing code or looking at accuracy.

If the fixed candidate fails a valid preflight, stop without a metric run and record a preflight reject. Do not delete a block, change batch size, alter precision, enable compilation, or tune widths on this node. If it passes, execute exactly one full training run.

## Correctness Smokes

Before latency measurement:

1. Assert the exact nine block specifications, shape transitions, three projection shortcuts, and final logits shape.
2. Reconcile every parameter tensor and exact total 4,298,970; independently assert each added block's count.
3. Register convolution hooks on one inference forward and verify 19 main/stem plus 2 projection calls with expected spatial shapes.
4. Verify calculated MAC totals for every convolution and the 619,104,768 candidate sum.
5. Run CPU and GPU BF16/channels-last forward/backward; require finite loss/gradients and nonzero gradients for all trainable parameters.
6. Verify block drop probabilities rise linearly from `0.08/9` to `0.08`, and final-quarter scale reaches zero.
7. Reuse EXP-002 CutMix orientation, clipped-area lambda, zero-area, exposure, and private-generator tests unchanged.
8. Confirm the frozen evaluator calls `model(inputs)` without API changes and evaluation mutates no training RNG/statistics beyond normal mode behavior.
9. Compile/lint `train.py`, assert only it differs from EXP-002, and verify startup config reports WRN-22-4, nine blocks, and the fixed parent recipe.

## Expected Accuracy and Falsification

The formal parent-relative threshold is 95.33% over EXP-002's 95.23%. Because the protocol has shown 0.14-0.29-point variation, the mechanism-sized target is **95.53% (+0.30 points)** and the plausible upside is 95.5-95.8% if added depth improves class-boundary representation despite lower exposure. Reaching the global best 95.40% is useful context but does not replace the parent-relative rule.

The strongest failure mode is undertraining: the candidate may complete only about 95 epochs and enter the low-LR tail after substantially fewer updates. Near-zero parent training loss also suggests capacity is not the main limiter, so deeper weights can overfit or add no generalization. A valid result below 95.33% rejects WRN-22-4 under this fixed budget. A 95.33-95.52 result is a formal improvement but below the preregistered detectable-effect target. No rerun, LR compensation, seed change, or depth tuning is permitted after observing accuracy.

Attribution is package-level. The comparison changes depth, parameters, initialization consumption, per-block drop probabilities, throughput, epoch/evaluation count, and sample exposure together. A gain supports the fixed WRN-22-4-under-300-seconds package; it does not isolate pure depth at matched updates. The exact latency and exposure audit is necessary to interpret either success or failure.

## Full-Run Verification

After a passing preflight, launch once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, no traceback/nonfinite/CUDA errors, 300-second charged budget, total runtime below 600 seconds, one evaluation per completed epoch, exact parameter/config inventory, CutMix ratio near 0.5, LR/drop-path endpoint logs, complete summary, and `best_test_acc>=95.33%`. Record actual median step behavior, updates, epochs, peak VRAM, best/final accuracy, and loss against both EXP-002 and the 95.40 global best. Remove `run.log` after analysis.

## Effort

**Low implementation, medium verification.** The code change is three block specifications plus labels/audits, but the convolution-path exposure risk requires exact inventory, FLOP accounting, and a decisive same-harness GPU-0 preflight.

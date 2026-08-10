# Proposal: Compact Pre-Activation ResNeXt-20, 8x16d

## Summary

Replace EXP-004's six-basic-block WRN-16-4 with a six-bottleneck-block,
pre-activation ResNeXt-20. Use cardinality 8, stage outputs and bottleneck
widths `[128, 256, 512]`, and two blocks per stage. The resulting group widths
are `[16, 32, 64]`, all friendly to BF16 tensor execution. Keep the complete
validated training recipe unchanged: independent-image batches, front-loaded
CutMix, six-block stochastic-depth schedule, time-based LR, and period-two SAM
in the clean final quarter.

The candidate has exactly 2,159,018 parameters and approximately 357.4M
convolution/classifier MACs per image, versus 2,748,890 parameters and about
392.6M MACs for the parent. Despite lower arithmetic, grouped convolutions and
extra kernel launches can be slower at 32x32, so an H20 microbenchmark is a
mandatory go/no-go gate before the full run.

## Rationale

EXP-004 is the global best at 95.40%, with 25,560 optimizer steps, 132 epochs,
0.1654 final loss, and 1,190.5 MiB peak VRAM. EXP-005's DLB child regressed to
95.28% because overlapping batches halved new-image introduction; it did not
show architectural saturation. This proposal preserves EXP-004's independent
image stream and changes only representation.

`experiments/006/papers/resnext.md` reports that cardinality can outperform
extra depth or width at matched complexity. Its CIFAR ResNeXt-29 8x64d result
is stronger than a similarly sized Wide ResNet, but those models have more
than 34M parameters and long conventional schedules. The compact adaptation
here is therefore an exploratory architecture bet, not a reproduction.

Cardinality offers eight parallel channel subspaces in each spatial transform.
The wider `[128, 256, 512]` stage states retain more channel information than
the parent's `[64, 128, 256]` states, while grouped 3x3 kernels and surrounding
1x1 projections keep total parameters and MACs below the parent.

## Exact Architecture

### Stem and stages

- Stem: bias-free `3x3`, 3 to 32 channels, stride 1, padding 1.
- Stage 1: two blocks, `in/out/mid = 32/128/128` then `128/128/128`, stride 1.
- Stage 2: two blocks, `128/256/256` with stride 2, then `256/256/256` stride 1.
- Stage 3: two blocks, `256/512/512` with stride 2, then `512/512/512` stride 1.
- Every grouped 3x3 uses `groups=8`; per-group widths are 16, 32, and 64.
- Tail: BN(512), ReLU, global average pool, and Linear(512, 10).

This has six residual units and 20 weighted layers under the conventional
stem + three-convolution blocks + classifier count.

### Pre-activation bottleneck

Implement `PreActResNeXtBlock(in_channels, out_channels, mid_channels,
stride, drop_prob, cardinality=8)` as:

```text
x -> BN -> ReLU -> 1x1(in, mid)
  -> BN -> ReLU -> grouped 3x3(mid, mid, groups=8, stride)
  -> BN -> ReLU -> 1x1(mid, out)
  -> existing stochastic depth -> + shortcut
```

When shape changes, apply a bias-free 1x1 projection with the block stride to
the first pre-activated tensor, matching the parent's pre-activation shortcut
semantics. Do not put a ReLU after addition. Assert that `mid_channels` is
divisible by cardinality.

Use the parent's six depth-dependent drop probabilities unchanged:
`MAX_DROP_PATH * (block_index + 1) / 6`. Apply the mask after the final 1x1 and
before addition. Retain the current late scale annealing from progress 0.75 to
1.0.

Initialize Conv2d and Linear weights with the existing Kaiming-normal helper,
BatchNorm scales to one and biases to zero, and linear bias to zero. Keep all
convolutions bias-free. Rename only the architecture classes and setup log;
the training loop should not otherwise know which residual representation it
is optimizing.

## Complexity Estimate

Counts include Conv/Linear weights, BatchNorm affine parameters, and classifier
bias. MACs include Conv/Linear multiply-accumulates for one 32x32 image but not
BN, activations, pooling, CutMix, or SAM bookkeeping.

| Model | Parameters | MACs/image | Relative MACs |
|---|---:|---:|---:|
| EXP-004 PreAct WRN-16-4 | 2,748,890 | ~392.6M | 1.00x |
| Compact ResNeXt-20 8x16d | 2,159,018 | ~357.4M | 0.91x |

The candidate uses 22 convolution calls including projections versus 16 for
the parent. Lower MACs therefore do not guarantee higher throughput. Grouped
3x3 performance, launch latency, and channels-last kernel selection dominate
the feasibility decision.

Activation widths double, so activation memory can rise even though parameter
and SAM-snapshot storage fall. A rough expectation is 1.3-1.8 GiB peak VRAM,
still negligible on the 97,871 MiB H20.

## Preserve CutMix, Drop Path, and SAM

- Keep batch size 256, global seed 42, crop/flip transforms, clean data-loader
  shuffling, and dedicated seed-42 CutMix CPU/CUDA generators.
- Keep `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, and `CUTMIX_END=0.75` exactly.
  Cardinality adds no RNG draws, so augmentation streams remain isolated.
- Keep `MAX_DROP_PATH=0.08` and one per-example residual mask per block. The six
  mask calls and their tensor shapes match the parent, preserving stochastic
  regularization structure even though initialization necessarily changes.
- Keep `SAM_RHO=0.05`, `SAM_START=0.75`, and `SAM_PERIOD=2`. The three BN layers
  in each new block must all appear in `batch_norm_modules`, so the SAM second
  pass suppresses every running-stat update. CUDA RNG replay must reproduce the
  six drop-path masks, and all new parameters must join snapshots, perturbation,
  restoration, and the sole optimizer update.
- Keep BF16 autocast, channels-last model/inputs, SGD/Nesterov, weight decay,
  and the time-indexed warmup/cosine schedule unchanged. Every architecture and
  SAM operation stays inside the charged `t0`/synchronize interval.
- Keep exactly one evaluator call per epoch and the existing summary contract.

## Mandatory H20 Microbenchmark

Before a full 300-second experiment, run a separate fixed-seed process on
physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`. Instantiate parent and candidate
in BF16-autocast/channels-last mode at batch 256. For each model:

1. Run 50 warmup iterations, then time at least 200 ordinary
   forward/backward/Nesterov iterations with synchronization.
2. Separately time at least 100 scheduled SAM iterations using the production
   two-pass RNG replay, BN suppression, snapshots, restore, and update.
3. Time at least 200 evaluation forwards at batch 256.
4. Report median and p90 latency, peak allocated VRAM, finite loss/gradients,
   and candidate/parent ratios. Do not use compilation in only one arm.

Weight the training ratio using EXP-004's observed mix: approximately 90.4% of
steps are ordinary and 9.6% are SAM steps. Project candidate steps as
`25,560 / weighted_latency_ratio`.

Proceed only if all of these hold:

- projected steps are at least 23,500;
- weighted training latency is at most 1.09x the parent;
- evaluation latency projects total runtime below 600 seconds;
- no grouped-convolution fallback, NaN/Inf, missing gradient, OOM, or SAM
  restoration/BN/RNG invariant fails.

The gate is deliberately based on measured H20 behavior, not MAC count. If it
fails, reject this configuration before a metric run rather than silently
changing cardinality or widths.

## Expected Effect

The parent-relative success threshold is 95.50%. The preregistered hypothesis
is that compact cardinality will reach `best_test_acc` of 95.50-95.90% while
retaining at least 23,500 steps. A gain would indicate that aggregated channel
subspaces are a better use of the fixed compute budget than the parent's two
dense 3x3 transforms. The effect range is uncertain because published evidence
uses much larger ResNeXt-29 models and longer training.

Run exactly one full experiment after the microbenchmark passes:
`timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
A result below 95.50% is no-improvement regardless of parameter efficiency.
Do not select a follow-up width/cardinality from the same test run.

## Failure Modes

- **Compact scaling loses capacity:** 2.16M parameters may underfit relative to
  the 2.75M WRN despite wider stage outputs.
- **Grouped kernels are launch-bound:** 22 convolutions and small CIFAR feature
  maps can erase the 9% arithmetic saving and reduce optimizer exposure.
- **Cardinality is too low or too compressed:** the paper's strongest CIFAR
  models use much larger width/depth; eight groups may not reproduce that gain.
- **More serial transforms hurt short training:** three convolutions and three
  BNs per block increase path depth from 16 to 20 weighted layers and may need
  different optimization, which this controlled experiment intentionally does
  not tune.
- **CutMix interaction:** mixed images may encourage diverse group features or
  may make separated group subspaces harder to coordinate. The clean final
  quarter provides recovery but does not guarantee it.
- **SAM cost shifts:** candidate forward/backward ratios can differ between
  ordinary and two-pass steps. The weighted microbenchmark and 23,500-step gate
  bound this risk.
- **BatchNorm correctness:** failing to include all 19 BNs (three per block plus
  final BN) in SAM suppression would double-update some statistics and invalidate
  comparison.
- **Run variance:** prior selected results moved 0.14-0.29 points on confirmation.
  Use one fixed architecture and seed 42, with no metric-driven retry.

## Verification

1. Unit-check all stage shapes, group divisibility, projection strides, six
   drop probabilities, and exact `num_params == 2_159_018`.
2. Run a BF16/channels-last forward/backward smoke and verify finite logits,
   loss, and a non-None finite gradient for every trainable parameter.
3. On a scheduled SAM smoke, verify perturbation norm 0.05, identical replayed
   drop masks, one update to each BN buffer, exact parameter restoration, and
   one optimizer/momentum update.
4. Pass the mandatory parent-vs-candidate H20 microbenchmark above.
5. In the full run, confirm GPU 0 identity, 300 charged seconds, under 600 total
   seconds, one evaluation per epoch, unchanged CutMix/SAM ratios and transition
   at 0.75, complete summary, and `best_test_acc >= 95.50%`.
6. Verify only `train.py` changed during implementation; keep `prepare.py`,
   evaluator, dependencies, and all seeds untouched, then remove transient logs.


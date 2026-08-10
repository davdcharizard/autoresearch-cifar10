# Proposal: Dense PreAct WRN-16-5 from EXP-002

## Summary

Increase only the width of EXP-002's six-basic-block PreAct WRN from stage widths `64/128/256` to `80/160/320`. Keep two blocks per stage, the same strides and projection shortcuts, batch 256, BF16/channels-last path, Nesterov optimizer, wall-clock LR/drop-path schedules, independent-image loader, and front-loaded probabilistic CutMix. Modify only `train.py` and do not inherit EXP-004's later SAM code.

The fixed architecture is WRN-16-5 in the same local naming convention as the parent's WRN-16-4: a 16-channel stem followed by stage widths `16 * 5 = 80`, `32 * 5 = 160`, and `64 * 5 = 320`. This is a single preregistered capacity point, not a width sweep.

## Why This Branch and Mechanism

EXP-002 at 95.23% retains the lineage's largest recipe gain after the WRN rewrite: CutMix added 0.61 points while preserving 27,950 steps. Forking there tests architecture capacity without the two-pass SAM cost. Its prior architecture child, EXP-009, never reached an accuracy run because four FP32 SE paths added 20.7% latency through many small launches. WRN-16-5 directly enlarges the existing dense convolutions while preserving the parent's 16 convolution calls, so it tests capacity without adding a new execution path.

The EXP-010 papers support architecture allocation as a first-class lever. Pyramidal networks show that CIFAR residual accuracy depends on channel allocation; ResNeXt shows width/cardinality/depth should be compared under actual complexity; Gradually Updated Networks motivate representation changes that reuse existing convolutional work. This proposal chooses the lowest implementation-risk interpretation: uniform dense widening with no grouped kernels, extra projections, custom channel ordering, or auxiliary modules.

The system is generalization-limited rather than memory-limited, so more width is not automatically beneficial. The hypothesis is that validated CutMix/drop-path regularization can use the added channels to learn stronger features despite fewer optimizer steps. The experiment is valuable only if a parent-relative H20 preflight confirms that dense, 16-aligned widths do not lose more exposure than preregistered.

Sources:

- `experiments/010/00-navigate.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`
- `experiments/002/04-analysis.md`
- `experiments/009/04-analysis.md`
- `experiments/010/papers/aggregated-residual-transformations.md`
- `experiments/010/papers/deep-pyramidal-residual-networks.md`
- `experiments/010/papers/gradually-updated-neural-networks.md`

## Exact Implementation

Keep `PreActWideBlock` unchanged. In `PreActWideResNet`, replace only the six block specs and tail width:

```python
block_specs = [
    (16, 80, 1),
    (80, 80, 1),
    (80, 160, 2),
    (160, 160, 1),
    (160, 320, 2),
    (320, 320, 1),
]
self.bn = nn.BatchNorm2d(320)
self.fc = nn.Linear(320, num_classes)
```

Retain the 16-channel `3x3` stem. Shape-changing blocks keep the existing preactivated bias-free `1x1` projection at strides 1, 2, and 2. Each residual branch remains two dense bias-free `3x3` convolutions. Keep six depth-scaled drop probabilities `MAX_DROP_PATH * (index+1)/6` and apply the mask at the same point.

Rename the human-readable model/config output to `PreAct WRN-16-5` and log `stage_widths=80,160,320`. The forward signature and evaluator call remain unchanged.

Do not change:

- `BATCH_SIZE=256`, crop/flip, shuffle, workers, pinning, or dropped-last epoch semantics;
- seed 42 or initialization code;
- `PEAK_LR=0.2`, warmup fraction 0.05, minimum ratio 0.01, Nesterov momentum 0.9, or weight decay `1e-4`;
- `MAX_DROP_PATH=0.08`, its final-quarter decay, or any time-based boundary;
- `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, `CUTMIX_END=0.75`, `CUTMIX_SEED=42`, helper geometry, loss weighting, or dedicated generators;
- BF16 autocast, channels-last layout, timer, evaluator, validation cadence, metric accumulation, or required final summary.

## Exact Parameter Count

Counts include all Conv/Linear weights and BatchNorm affine parameters:

| Component | Parameters |
|---|---:|
| Stem `3x3`, 3 -> 16 | 432 |
| Block 1, 16 -> 80 | 70,592 |
| Block 2, 80 -> 80 | 115,520 |
| Block 3, 80 -> 160 | 358,880 |
| Block 4, 160 -> 160 | 461,440 |
| Block 5, 160 -> 320 | 1,434,560 |
| Block 6, 320 -> 320 | 1,844,480 |
| Final BN + `320 -> 10` classifier | 3,850 |
| **Total** | **4,289,754** |

The parent has 2,748,890 parameters. WRN-16-5 adds 1,540,864 parameters, a 56.05% increase. Parameters, gradients, Nesterov momentum, and activations remain trivial relative to the 98 GB H20; expected peak allocation is roughly 1.4-1.7 GiB versus EXP-002's 1,178.9 MiB.

## Exact MAC/FLOP Estimate

For one 32x32 image, counting Conv/Linear multiply-accumulates and excluding BN/ReLU/pooling/CutMix:

| Model | MACs/image | FLOPs/image (2 per MAC) | Ratio |
|---|---:|---:|---:|
| EXP-002 WRN-16-4 | ~392.6M | ~0.785G | 1.000x |
| Candidate WRN-16-5 | 609.930M | 1.220G | 1.554x |

Candidate stage/block MACs are 72.090M, 117.965M, 91.750M, 117.965M, 91.750M, and 117.965M, plus 0.442M stem and 0.003M classifier. Widths 80/160/320 are all multiples of 16 and preserve the parent's convolution count, which should retain efficient BF16 tensor kernels. Even so, arithmetic predicts materially fewer updates.

EXP-002 averaged 10.73 ms over 27,950 charged steps. If latency scaled exactly with MACs, WRN-16-5 would take about 16.7 ms and complete about 18,000 steps. Better H20 utilization could reduce the ratio to 1.35, yielding about 20,700 steps; an unfavorable kernel choice could be slower than 1.55x. The preregistered expectation is 18,000-20,500 steps and 92-105 completed natural epochs, versus 144 parent epochs.

## Fixed-Budget Consequences

The 300-second charged schedule remains the source of truth. Widening does not receive extra epochs or LR retuning:

- warmup still lasts 15 charged seconds but likely contains only about 900-1,050 updates rather than roughly 1,400;
- CutMix remains eligible for the first 225 seconds, likely about 13,500-15,400 steps, with the same fixed 0.5 gate;
- the final clean/drop-path-decay phase remains 75 seconds, likely about 4,500-5,100 updates;
- total independent image appearances fall from about 7.15M to roughly 4.6-5.25M.

This exposure loss is the central tradeoff. Time-normalized LR/drop-path phases remain correctly aligned in wall time, but the wider model sees fewer batches at every phase. A gain must come from more useful work per update, not from extra compute or data.

## Initialization and Optimization Risks

Use the existing Kaiming-normal initialization for every changed Conv/Linear tensor, BN weights one/biases zero, and classifier bias zero. Resetting seed 42 makes the candidate deterministic, but shape changes necessarily consume a different number of random values and prevent common downstream weights from matching EXP-002; do not claim initial-function parity. The unchanged stem is created first and should remain seed-identical.

Kaiming fan-in scaling should keep activation variance stable as width increases. Batch size is unchanged, so keeping LR 0.2 is more defensible than linear width scaling, but the larger network may still need a different optimum. The fixed experiment intentionally does not tune LR, decay, drop path, or CutMix around width. A short preflight may reject NaN/divergence or grossly broken activation statistics, but may not select new hyperparameters from loss or accuracy.

Potential failure modes are:

- fewer steps/unique image views outweigh added representation capacity;
- the parent is already capacity-sufficient and extra width increases overfitting despite CutMix/drop path;
- the 15-second warmup contains too few steps for stable peak-LR entry;
- unchanged weight decay regularizes the larger parameter set differently;
- 80-channel kernels underperform the parent's power-of-two widths despite 16 alignment;
- fewer epochs reduce evaluation opportunities for the max-over-epoch metric.

## Parent-Relative GPU-0 Preflight

Run parent and candidate in separate fixed-seed processes on physical GPU 0, using the same benchmark script, batch 256, pinned CPU inputs, nonblocking transfer, channels-last, BF16 autocast, the production loss/backward/Nesterov update, and CUDA synchronization. Do not compile only one arm.

Measure five alternating parent/candidate trials to limit thermal/co-tenant drift. In each trial:

1. Reset seed and optimizer state; run at least 50 warmup steps.
2. Time at least 300 clean production steps.
3. Time at least 300 fixed CutMix production steps, including transfer, patch construction, two-term loss, backward, and update.
4. Time at least 200 evaluation forwards.
5. Report median, p90, mean, dispersion, peak VRAM, finite loss/gradients, and actual selected convolution kernels if profiler metadata is available.

Weight clean/CutMix training latency as `0.625 * clean + 0.375 * CutMix`, matching EXP-002's full-run time fractions. Compute each paired candidate/parent ratio and project steps as `27,950 / median_ratio`.

Proceed to the one metric run only if all conditions pass:

- candidate paired median weighted latency ratio `<= 1.60`;
- candidate paired p90 weighted latency ratio `<= 1.70`;
- median projected optimizer steps `>= 17,500`;
- the same harness's parent ratio is 1.0 and projected 27,950 by construction, with trial coefficient of variation `<= 5%`;
- evaluation latency plus projected epoch count keeps total runtime below 600 seconds;
- peak VRAM remains below 4 GiB;
- no nonfinite loss/gradient, OOM, layout fallback, missing gradient, worker failure, or unexpected convolution-count change.

These are parent-relative feasibility limits, not absolute floors that can reject a valid measured parent. If the fixed width fails, reject it before accuracy evaluation; do not reduce batch size, width, or change LR to rescue the same experiment.

## Smokes and Audit

1. Assert all six block specs, stage outputs, strides, projection presence, final width, six drop probabilities, and exactly 16 Conv2d modules including projections/stem.
2. Independently recompute and require `num_params == 4_289_754` and `MACs/image == 609_930_368` under the stated convention.
3. Reset seed twice and require bitwise deterministic candidate state; require the candidate stem to match a seed-reset parent stem and verify Kaiming/BN initialization statistics for changed shapes.
4. Run FP32 and BF16/channels-last forward/backward smokes at batch 256; require logits `(256,10)`, finite loss/gradients for every parameter, finite BN buffers, and preserved channels-last activations.
5. Compare activation mean/variance at stem and all block outputs against broad preregistered finite/stability bounds; reject only numerical explosion/collapse, not ordinary architecture differences.
6. Re-run deterministic CutMix geometry, source/target orientation, clipped-area lambda, zero-area, dedicated-generator isolation, and early/late cutoff tests unchanged.
7. Verify six global CUDA drop-path draws per training forward, unchanged mask shapes by block output, and no stochastic-depth draw during evaluation.
8. Verify LR/drop-path functions and charged timing code are byte-equivalent to EXP-002 apart from config output; evaluator/default forward and all summary data flow remain unchanged.
9. Pass the parent-relative GPU-0 preflight above and confirm only `train.py` differs from EXP-002.

## Full Run and Hypothesis

After preflight, run exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require the 97,871 MiB H20 on physical GPU 0, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 17,500 steps, `num_params=4,289,754`, complete summary, one evaluation per completed epoch, CutMix applied/eligible ratio near 0.5 only below progress 0.75, final LR near 0.002, final effective drop path near zero, and no NaN/Inf, traceback, CUDA error, OOM, or timeout.

The formal parent-relative threshold is `best_test_acc >= 95.33%` versus EXP-002 at 95.23%. The preregistered meaningful target is `>=95.50%`, which would also clear the current 95.40% global best by the goal's 0.10-point margin. The expected range is 95.45-95.75% if additional width compensates for the 27-36% loss of update exposure.

A score below 95.33%, preflight failure, fewer than 17,500 steps, or any scope/timing/integrity failure falsifies the fixed WRN-16-5 proposal. A 95.33-95.49 result is a formal parent improvement but does not establish a new global-best branch. Do not rerun, seed-select, or tune width/LR/decay/drop path from the outcome.

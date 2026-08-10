# Proposal: Calibrated stage-3 width-5 expansion from EXP-011

## Summary

Grow EXP-011's PreAct WRN-16-4 only at the final 8x8 stage, changing the stage widths from `64/128/256` to `64/128/320` while retaining two blocks per stage. This is a tapered `WRN-16-[4,4,5]`: the first two stages remain width factor 4 and the final stage takes the canonical width-factor-5 endpoint. It is a single preregistered architecture, not a width sweep.

The candidate preserves the full EXP-011 package: crop/flip, batch 256, CutMix, stochastic depth, time-based LR, late period-two SAM, cadence-31 full-state EMA, BF16, channels-last, fixed seed 42, and once-per-epoch evaluation. Only `train.py` changes. The intended intervention is more low-resolution semantic capacity without the high-resolution arithmetic and activation traffic of dense WRN-16-5.

The formal hypothesis is that this calibrated capacity increase raises `best_test_acc` from EXP-011's `95.61%` to at least `95.71%`. Stronger mechanism support requires a final-16 EMA mean of at least `95.69%` versus the parent's `95.493125%`, together with the preregistered training-dose and state-integrity gates below. This higher bar separates a meaningful plateau lift from another selected-maximum shift.

## Why not full WRN-16-5

Full WRN-16-5 (`80/160/320`) is not calibrated to this fixed-time parent. It raises parameters from `2,748,890` to `4,289,754` (+56.05%) and Conv/Linear MACs from `392,612,352` to `609,930,368` per image (+55.35%). If charged latency scaled with arithmetic, EXP-011's `25,798` updates would fall to about `16,606`, before accounting for fewer EMA samples and fewer max-selected evaluations. The H20 has ample memory, but optimizer exposure rather than memory is the relevant cost.

The proposed `64/128/320` taper raises parameters to `3,827,290` (+39.23%) for only `461,556,864` MACs/image (+17.56%). It keeps both 64-channel local blocks and both 128-channel middle blocks, avoiding EXP-010's unsuccessful removal of an early block. All new arithmetic is at 8x8, where EXP-010 showed that H20 latency can be materially better than equal-MAC work at 32x32: moving one block late reduced median step latency from `10.000` to `9.242 ms`. The candidate adds no module or kernel launch; it enlarges existing dense convolutions to a tensor-friendly multiple of 64.

This fixed endpoint is selected structurally, not from CIFAR-10 test tuning. `320` is the ordinary width-factor-5 final width and was proposed, but never executed, in EXP-010's architecture ideation. The experiment tests exactly this one taper once.

## Evidence and lineage fit

- EXP-011 is the base and current best: `95.61%`, with a stable final-16 EMA mean of `95.493125%`, `25,798` updates, `160` EMA samples, and only `1,222.4 MiB` peak allocation (`experiments/011/04-analysis.md`).
- The system understanding identifies stable generalization above the EMA plateau as the limiter, with memory headroom extreme and extra full forwards costly (`02-system-understanding.md`).
- EXP-010 preserved MACs while shifting a block from stage 1 to stage 3. It ran 9.3% more updates but scored `95.04%`, so deleting early processing is not supported. Its explicit unexplored avenue was to retain 2-2-2 depth and modestly widen only the final stage (`experiments/010/04-analysis.md`).
- The PyramidalNet distillation supports channel allocation as a genuine CIFAR representation lever, while also warning that exact transfer is architecture- and schedule-dependent (`knowledge/papers/deep-pyramidal-residual-networks.md`). This proposal uses a single coarse taper rather than adding a projection at every block.
- EXP-013's fixed-scale cosine classifier retained full dose but lowered the EMA plateau by `0.42` points, motivating a representation-capacity change rather than another uncalibrated classifier geometry (`experiments/013/04-analysis.md`).
- EXP-014's PolyLoss paper describes a separate low-cost loss lever, and its averaging paper supports retaining LR annealing with the already validated EMA. Neither supplies evidence for choosing a width from test results. The group-equivariant paper would require a much larger one-file architecture change and additional execution paths. Width is preferred here for implementation and attribution discipline (`experiments/014/papers/`).

The capacity premise remains unverified. EXP-001 reported near-zero late training loss before later regularizers were added, which argues that generalization rather than raw train fit may dominate. Record the candidate's terminal debiased training loss and paired early conditioning, but do not claim capacity limitation from them because no contemporaneous full parent run is authorized.

## Exact one-file implementation

Keep `PreActWideBlock` byte-for-byte unchanged. In `PreActWideResNet`, replace only the final two block specifications and tail width:

```python
block_specs = [
    (16, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 320, 2),
    (320, 320, 1),
]
...
self.bn = nn.BatchNorm2d(320)
self.fc = nn.Linear(320, num_classes)
```

Update the human-readable model/config label to `PreAct WRN-16-[4,4,5]` and report `stage_widths=64,128,320`. Do not introduce a general width switch, alternate candidate, dynamic branch, or benchmark-only code into production. The forward signature remains `forward(x, drop_scale=0.0)`, so the evaluator and SAM replay path are unchanged.

Do not change any of the following:

- `BATCH_SIZE=256`, the 50,000-image shuffled loader, crop/flip/normalization, workers, pinning, or `drop_last=True`;
- global seed 42, CutMix seed 42, CutMix probability/geometry/label weighting, or dedicated generators;
- peak LR `0.2`, warmup/minimum ratios, time-derived cosine progress, momentum `0.9`, Nesterov, or weight decay `1e-4`;
- six-block drop-path ordering, `MAX_DROP_PATH=0.08`, or final-quarter decay;
- `SAM_RHO=0.05`, start `0.75`, period 2, RNG replay, BN suppression, perturb/restore, or one optimizer update per batch;
- EMA start, cadence 31, 18.75-second half-life, full parameter/buffer state, charged placement, evaluation swap, or audits;
- BF16 autocast, channels-last layout, timer boundaries, 300-second budget, evaluation cadence, metric selection, or summary fields.

## Exact parameter inventory

Counts include convolution/linear weights, linear bias, and BatchNorm affine parameters.

| Component | Parent | Candidate |
|---|---:|---:|
| Stem `3x3`, 3 -> 16 | 432 | 432 |
| Block 1, 16 -> 64 | 47,264 | 47,264 |
| Block 2, 64 -> 64 | 73,984 | 73,984 |
| Block 3, 64 -> 128 | 229,760 | 229,760 |
| Block 4, 128 -> 128 | 295,424 | 295,424 |
| Block 5 | 918,272 | 1,332,096 |
| Block 6 | 1,180,672 | 1,844,480 |
| Final BN and classifier | 3,082 | 3,850 |
| **Total** | **2,748,890** | **3,827,290** |

The candidate adds `1,078,400` parameters without adding parameter tensors. Live parameters, gradients, momentum, SAM snapshots, and the three EMA-owned parameter shadows add roughly seven FP32 copies in the peak state inventory; the added persistent parameter storage is therefore about `28.8 MiB`. Activations and workspaces will add more, but peak allocation should remain far below 4 GiB and is not expected to constrain the 97,871 MiB H20.

## Exact MAC inventory

The convention counts Conv/Linear multiply-accumulates, excludes BN/ReLU/pooling/CutMix, and counts one multiply-accumulate as one MAC.

| Component | Candidate MACs/image |
|---|---:|
| Stem | 442,368 |
| Block 1 | 48,234,496 |
| Block 2 | 75,497,472 |
| Block 3 | 58,720,256 |
| Block 4 | 75,497,472 |
| Block 5, 128 -> 320 at 8x8 | 85,196,800 |
| Block 6, 320 -> 320 at 8x8 | 117,964,800 |
| Classifier | 3,200 |
| **Total** | **461,556,864** |

The ratio to EXP-011 is `1.1756045x`. Pure arithmetic scaling would project about `21,944` updates and 112 completed natural epochs. Because the change enlarges only low-resolution dense kernels and EXP-010 showed favorable H20 shape effects, a measured latency ratio of roughly `1.08-1.15x` is plausible, corresponding to approximately `22,433-23,887` updates and 115-122 completed epochs. This is a hypothesis for the accuracy-blind preflight, not permission to alter the width.

## Interaction with the frozen recipe

### Initialization and RNG comparability

Retain the parent's constructors and `_weights_init` exactly. Kaiming fan-in scaling is appropriate for the wider convolutions; BN remains one/zero initialized and classifier bias remains zero. Resetting seed 42 twice must reproduce the candidate bit-for-bit.

Do not require parent/candidate shared weights or post-construction RNG states to match. EXP-010 established that shape-dependent constructor draws occur before the model-wide `apply`, so changing late tensor shapes changes the RNG state from which even the stem is reinitialized. Burning draws, copying overlapping weights, using per-layer seeds, or preserving a parent submatrix would be a second intervention and could amount to seed engineering. The result must be interpreted as the fixed-seed architecture package.

The architecture does not add stochastic calls in forward: there are still six drop-path masks in the same order and with the same probabilities. Dedicated CutMix generators remain isolated. Parent and candidate smokes should replay the same scripted CutMix and drop-path decisions, but numerical outputs need not match.

### LR, weight decay, CutMix, and drop path

Batch size and peak LR remain fixed. Width is not a data-parallel batch increase, so linear LR scaling has no basis. Kaiming initialization should preserve broad activation scale, while a paired early trace can detect only numerical collapse, not tune LR.

Per-weight decay remains `1e-4`; total regularization energy and model capacity necessarily change with parameter count. CutMix and stochastic depth are plausible reasons the added capacity may generalize instead of merely fitting harder, but no strength is retuned. Six blocks preserve the parent's exact depth-indexed drop probabilities, unlike EXP-010's stage reassignment.

### SAM

SAM remains every second eligible late step with global Euclidean radius `0.05`. The perturbation norm is still exactly rho, but its energy can redistribute across the larger tensors, so width and SAM geometry are not causally separable. The preflight must exercise the exact two-pass path, RNG replay, one BN update, and exact restore. No rho or cadence compensation is allowed.

### EMA and evaluation

Charged-time EMA retains its 18.75-second horizon, so its decay remains meaningful if steps slow. Cadence 31 remains odd and continues alternating period-two ordinary/SAM states; fewer steps will reduce sample count from the parent's 160 to an estimated 135-150. Larger tensor copies add some charged cost and must be included in the timing measurement. Full-state swaps, BN buffers, restore identity, RNG audits, and once-per-epoch routing remain unchanged.

Fewer natural epochs mean fewer evaluation checkpoints for the max-selected primary metric. This is a real fixed-budget consequence, not something to compensate with extra evaluations. Report the final-16 EMA mean/range when at least 16 EMA evaluations exist.

## Deterministic correctness gates

Before timing or metric evaluation:

1. Materialize exact parent commit `d68f73a` under `/tmp`; import parent and candidate under distinct module names without invoking either `main`.
2. Assert the six candidate block tuples, stage shapes `64/128/320`, stride pattern `1,1,2,1,2,1`, three projection shortcuts, 16 Conv2d modules, output `(256,10)`, and unchanged six drop probabilities.
3. Independently reconcile every tensor and require exactly `3,827,290` parameters and `461,556,864` MACs/image. Require the same parameter key set and parameter-tensor count as the parent, with shape differences confined to blocks 5-6, final BN, and classifier.
4. Construct the candidate twice after seed reset and require bitwise candidate self-determinism. Record, but do not gate on, parent/candidate equality of common-shaped tensors and final RNG states.
5. Run CPU FP32 plus GPU-0 BF16/channels-last forward/backward and one Nesterov step. Require finite logits/loss/BN buffers and finite nonzero gradients for every trainable tensor.
6. Reuse the parent's deterministic CutMix geometry/orientation/lambda/generator checks and verify exactly six drop-path RNG draws per training forward and none in evaluation.
7. Execute production-faithful ordinary and SAM steps, including RNG replay, BN suppression, perturbation norm, exact parameter restore, and one optimizer update.
   Record `||epsilon||/||w||` for parent and candidate during the paired SAM smoke; this is report-only and cannot retune `SAM_RHO`.
8. Execute at least 30 cadence-31 EMA samples covering both ordinary and SAM states, followed by an EMA evaluation swap/restore. Require exact state restoration, complete state coverage, preserved optimizer identities/RNG/module modes, finite nonzero distances, and zero audit failures.
9. Run `py_compile`, lint/diff checks, and require that only `train.py` differs from `d68f73a`; no change to `prepare.py`, dependencies, evaluator, seed, or protocol is permitted.

Any straightforward harness defect may be corrected only before a complete gate result exists. A production correctness failure after implementation correction is a pre-metric failure, not a reason to tune the architecture.

## One-shot parent-relative GPU-0 preflight

Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 before every GPU command, and expose only it with `CUDA_VISIBLE_DEVICES=0`. The first complete accuracy-blind measurement is decisive.

Use exact parent/candidate modules in one process with fixed materialized CIFAR batches shared across arms. Run five alternating-order timing rounds after warmup. Each arm and round must measure enough production-faithful early ordinary, early CutMix, late ordinary, and late SAM steps to estimate stable medians; a suitable minimum is `100/40/20/20` respectively per round, plus 40 evaluation forwards. Include input transfer, loss, backward, Nesterov update, synchronization, and the actual cadence-31 EMA work. Use identical scripted augmentation and replayable drop-path decisions between arms.

Weight the four charged paths by EXP-011's observed counts:

```text
early ordinary: 10,512 / 25,798
early CutMix:   10,345 / 25,798
late ordinary:   2,470 / 25,798
late SAM:        2,471 / 25,798
```

Report every round's parent/candidate weighted median, paired ratios, ratio median and MAD, p90, evaluation latency, peak allocation, projected updates/epochs/EMA samples, and projected total runtime. The parent-relative gates are:

- parent round drift `(max-min)/median <= 0.03`;
- paired-ratio `MAD/median <= 0.01`;
- candidate/parent weighted median latency ratio `<= 1.15`;
- p90 ratio `<= 1.20`;
- projected updates from `25,798 / ratio >= 22,000`;
- projected complete epochs `>=112` and projected cadence-31 EMA samples `>=130`;
- projected end-to-end runtime, using measured candidate evaluation latency, `<600s`;
- candidate peak allocated memory `<4,096 MiB`;
- all losses/gradients/state audits finite and all SAM/EMA restore/RNG/coverage checks exact.

There is no fallback width. If width 320 produces a valid ratio above `1.15`, record a pre-metric failure for this exact package. Do not resize to 288, rerun timing, or convert the experiment into an implicit width sweep.

These limits deliberately accept meaningful exposure loss while excluding full-WRN-16-5-like undertraining. If a valid first measurement fails, record EXP-014 as a pre-metric `crash`/`NaN` leaf under the loop's conventions. Do not reduce width, change batch size, retune LR/decay/SAM/EMA, rerun timing, or inspect test accuracy.

An optional paired 200-step real-data trace from deterministic candidate/parent initializations may record finite loss and activation/gradient norms. It is informational except for nonfinite/collapse/integrity failure and may not select an architecture, hyperparameter, or stopping decision.

## Metric run and verification

After every correctness and preflight gate passes, execute exactly one fixed-seed metric run:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

Do not stop on loss, training accuracy, intermediate test accuracy, or ordinary finite diagnostics. Abort only for process failure, nonfinite/integrity error, CUDA/OOM, 120 seconds without process/GPU/log progress, or the outer timeout. Do not rerun the metric or try another width after observing accuracy.

Require exit 0, charged training `[299.5,301.0]s`, total `<600s`, one evaluation per completed epoch, complete summary, `num_params=3,827,290`, no error signature, and only `train.py` changed. Audit exact CutMix/SAM eligibility and application counts, LR/drop-path endpoints, EMA cadence/parity/decays/state coverage/swaps/restores, and final-16 EMA values.

The formal tree verdict is:

- **improvement** only if `best_test_acc >=95.71%` and every integrity condition passes;
- **no-improvement** for a valid complete result below `95.71%`;
- **crash/NaN or invalid**, as appropriate, for failed execution or integrity.

Mechanism-supporting evidence additionally requires realized updates `>=22,000`, EMA samples `>=130`, balanced ordinary/SAM EMA sampling within one, and final-16 EMA mean `>=95.69%`. Report terminal debiased training loss, realized evaluation count, `best_test_acc - final16_mean`, and paired parent/candidate relative SAM perturbation. Falling below a dose target limits the architecture interpretation but cannot turn sub-threshold accuracy into improvement or authorize a retry.

## Expected effect and falsification

The realistic best-accuracy range is `95.60-95.90%`. The lower end reflects the risk that EXP-011 is already representation-sufficient and that losing 7-15% of updates, image views, EMA samples, and evaluations outweighs added capacity. The upper end assumes CutMix/drop path/SAM can exploit 39% more final-stage parameters and lift the stable EMA plateau by roughly 0.15-0.30 points. The proposal does not claim that width alone has a known effect size; the evidence supports the compute allocation, not the accuracy outcome.

At the latency gate the candidate is expected to complete roughly 115 rather than 133 evaluations, reducing max-selection opportunities. The final report must therefore treat a threshold-clearing maximum with tail mean below `95.69%` as formal improvement but weak scientific evidence, and must report the maximum-minus-tail premium explicitly.

Important failure risks are:

- less data and optimizer exposure overwhelms the capacity benefit;
- fixed LR or per-weight decay is suboptimal for the wider state;
- global SAM rho redistributes perturbation energy unfavorably;
- fewer cadence samples increase EMA variance despite the unchanged time horizon;
- 320-channel kernels or workspaces run slower than MAC arithmetic suggests;
- the parent is generalization-limited by data/objective rather than final-stage capacity;
- fixed-seed initialization divergence creates package-level uncertainty;
- fewer once-per-epoch evaluations reduce the opportunity for a selected maximum.

A valid result below `95.71%` falsifies this exact `64/128/320` package under the fixed 300-second protocol. It does not justify full WRN-16-5, a neighboring width, LR scaling, or another seed. A passing formal result with a weak tail mean would establish the tree metric but should be described as max-selected package evidence, not a stable representation gain.

## Effort and risk

Implementation effort is low: three architecture/tail lines plus truthful labels. Verification effort is medium-high because width changes initialization, throughput, SAM geometry, EMA state size, and evaluation count together. Execution risk is medium; memory risk is low, latency/dose risk is medium, and accuracy risk is medium-high. The stage-3 taper is nevertheless the most disciplined width test available because it preserves validated early/middle processing and spends added compute only in the H20-favorable low-resolution stage.

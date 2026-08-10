# Proposal 02: Balanced Mixup/CutMix Geometry

## Proposal

Keep exactly 50% of strong-phase batches hard in expectation, but replace exactly half of the accepted CutMix decision interval with Mixup:

| Worker-local draw `u` | Batch geometry | Probability | Alpha |
| --- | --- | ---: | ---: |
| `0.00 <= u < 0.25` | CutMix | 25% | 1.0 |
| `0.25 <= u < 0.50` | Mixup | 25% | 0.2 |
| `0.50 <= u < 1.00` | Hard | 50% | N/A |

This preserves the accepted hard-versus-soft gate `u < 0.5`. Retained `u < 0.25` CutMix calls begin from the same RNG position as EXP-010, while the accepted would-be CutMix events in `[0.25,0.5)` become Mixup. Total mixed-target probability does not increase, directly avoiding EXP-011's p=0.75 over-regularization failure.

Keep the weak tail exactly hard and Mixup-free after the 80% loader switch. Do not tune alpha/probabilities, add both transforms to one batch, alternate by epoch, change the switch, or fall back to pure CutMix after any failure.

## Alpha and Mechanism

Retain accepted CutMix `alpha=1.0`: its uniform Beta distribution already delivered the 94.15% frontier by matching target mass to visible donor area. Fix Mixup at `alpha=0.2`, within the original paper's common CIFAR range of 0.1-0.4. Beta(0.2,0.2) is endpoint-heavy, so global interpolation usually preserves a dominant original example rather than imposing the stronger central blends of alpha 1. This is important under the already difficult N1/M7 phase.

The predicted complement is geometric, not probabilistic. CutMix teaches regional class-bearing occlusion/localization; Mixup teaches approximately linear behavior between whole-image examples. At 25% each, the model sees both invariances without increasing soft-label frequency or reducing the 50% hard anchor. Sources: `knowledge/papers/mixup.md`, `knowledge/papers/cutmix.md`, and EXP-010/011.

## Forkserver-Safe Collator and RNG

Use installed torchvision v2 only:

```python
MIXUP_ALPHA = 0.2
CUTMIX_PROBABILITY = 0.25
MIXUP_PROBABILITY = 0.25

cutmix = v2.CutMix(alpha=1.0, num_classes=NUM_CLASSES)
mixup = v2.MixUp(alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES)

HARD, CUTMIX, MIXUP = 0, 1, 2

def mixed_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        u = torch.rand(()).item()
        if u < CUTMIX_PROBABILITY:
            inputs, targets = cutmix(inputs, targets)
            kind = CUTMIX
        elif u < CUTMIX_PROBABILITY + MIXUP_PROBABILITY:
            inputs, targets = mixup(inputs, targets)
            kind = MIXUP
        else:
            kind = HARD
    return inputs, targets, kind
```

The function and transform objects remain module-level/picklable for `multiprocessing_context="forkserver"`. A single categorical draw is mandatory; two Bernoulli gates would change probabilities and RNG consumption. `fork_rng(devices=[])` restores each worker's CPU torch RNG after the gate and chosen transform, so collation cannot perturb later RandomCrop/flip/RandAugment draws. Do not use Python `random`, NumPy, CUDA RNG, shared counters, or worker reseeding.

The strong loop unpacks `(inputs, targets, kind)` before `t0`, validates kind/target shape, and increments hard/CutMix/Mixup counters. The rebuilt weak loader keeps default two-item collation; it must return int64 one-dimensional targets and never a provenance code. Logging adds only one switch summary with all three counts.

## Target Semantics and Structural Gates

Hard batches return FP32 `[B,3,32,32]` inputs and int64 `[B]` targets. Both mixed branches return FP32 inputs and FP32 `[B,10]` probability targets; every target row must be finite, nonnegative, and sum to one. The unchanged `F.cross_entropy(outputs, targets)` handles both accepted target forms.

Before timing, require:

- 20,000 collator calls across all eight forkserver workers realize 48.5-51.5% total mixed and 23.5-26.5% each CutMix/Mixup, with no invalid provenance;
- saved/restored worker RNG states are bitwise equal around every hard/CutMix/Mixup collate call;
- for identical seeded materialized batches, candidate `u >=0.5` hard outputs and `u <0.25` CutMix outputs are bitwise accepted, including targets;
- controlled unique-label inputs prove Mixup pixels and targets use the same lambda/pairing, and CutMix target mass matches realized rectangle area;
- hard, CutMix, and Mixup losses/gradients are finite, all parameters receive finite gradients, and one SGD step creates finite FP32 momentum;
- exact model, optimizer, CPU/CUDA RNG, and BN state alignment before the first production batch; parameter count remains 1,073,962.

## Production-Batch Safety Gate

Use distinct real N1/M7 batches, not Gaussian tensors. Run 200 paired accepted/candidate steps from aligned model/optimizer state and the same materialized source batches. The accepted arm uses the original `u <0.5` alpha-1 CutMix rule; candidate uses the fixed three-way rule. Require:

- no non-finite input, target, loss, gradient, parameter, BN buffer, or momentum;
- candidate loss EMA no more than 1.5x accepted and no candidate-only one-class concentration above 95%;
- all three geometries occur at least 35 times, total mixed counts differ by no more than one when driven by the same registered `u` stream, and no soft target crosses into a weak batch;
- Mixup target/pixel interpolation remains nondegenerate on real batches; report lambda quantiles and same-class pair frequency without tuning from them.

These are collapse/integrity gates, not accuracy surrogates. EXP-015 showed short-fit checks can pass while the full strong phase underfits.

## Throughput, Loader, and Wall Gates

GPU work is unchanged for CutMix versus Mixup because both produce the same dense-target cross-entropy shape, but full-image interpolation can cost more worker CPU time than rectangular replacement. On the sole idle H20, run five alternating fresh-process accepted/candidate trials with the exact eight-worker loader and at least 1,000 real synchronized training steps after warmup.

Require:

- candidate/control median synchronized GPU-step ratio at most 1.01 and projected exposure at least **26,629 steps** (99% of EXP-010's 26,898);
- warmed strong-loader delivery at least 1.20x candidate GPU consumption, median iterator wait below 10% of GPU step time, and p95 below 20%;
- integrated wall/count ratio no more than 1.07 and no more than 0.02 above accepted paired control;
- stable per-trial CV below 3%, correct three-way proportions, peak allocation below 650 MiB, and no worker/memory growth;
- exact shutdown of all eight strong workers, weak-loader rebuild under five seconds, and first weak batch hard/int64;
- projected total runtime below 540 seconds with the unchanged evaluator cadence.

If any gate fails, do not move Mixup to GPU, change workers/prefetch, lower alpha, reduce its probability, or relax accounting.

## Underfit and Evaluation Risks

Although total soft frequency is fixed, Mixup changes regularization strength. Global interpolation can blur spatial evidence already altered by RandAugment, and its soft targets may weaken strong clean fit more than endpoint-heavy alpha suggests. Conversely, replacing regional occlusion may reduce the localization benefit that made CutMix succeed.

Pre-register 87.08% as the recurring strong-underfit marker and 89.0% as a healthier switch expectation. These are diagnostic only; they cannot trigger tuning or a rerun. Compare first weak checkpoint, tail slope, final/best gap, and NLL to EXP-010/011.

Evaluation remains untouched FP32 `Eval.evaluate`, at most once per epoch on the accepted schedule. No geometry-specific validation, extra terminal look, or checkpoint selection is allowed. A small epoch-count change from wall behavior receives no compensating test pass.

## One-Run Hypothesis and Verification

**Hypothesis:** the 50/25/25 hard/CutMix/Mixup plateau preserves accepted exposure and strong fit while complementary regional/global vicinal geometry raises `best_test_acc` from 94.15% to at least 94.25%.

If every preflight passes, change only `train.py`, confirm one idle 97,871 MiB H20, remove stale `run.log`, and run seed 42 exactly once as `uv run train.py > run.log 2>&1` under the 600-second supervisor. No valid-run retry is allowed.

Require exit zero, approximately 300 counted seconds, total below 600, finite ten-field summary, 1,073,962 parameters, at least 26,629 steps, one 80% switch, eight stopped workers, hard weak targets, and unique at-most-once-per-epoch evaluations. Strong realized proportions must fall within 48.5-51.5% hard and 23.5-26.5% each CutMix/Mixup.

Accept only at `best_test_acc >=94.25%`. Report geometry counts, target checks, switch/first-weak/best/final/NLL, exposure, VRAM, and wall time. A valid miss rejects exactly alpha-0.2 Mixup replacing half of accepted CutMix events; do not retry another alpha, ratio, schedule, or seed.

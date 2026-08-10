# EXP-006 Proposal: Plateau-Only Fixed-Square Cutout with Weak Refinement Tail

## Summary

Replace, rather than stack with, EXP-004's `RandAugment(num_ops=1, magnitude=7)` during the first 80% of counted training time with one fixed 16x16 mean-valued Cutout patch on every training image. At the existing 80% learning-rate boundary, shut down the Cutout loader and rebuild the accepted weak crop/flip loader for the final `0.01 -> 1e-4` cosine refinement tail.

Implement Cutout using the existing `torchvision.transforms.RandomErasing` tensor operation:

```python
transforms.RandomErasing(
    p=1.0,
    scale=(0.25, 0.25),
    ratio=(1.0, 1.0),
    value=0,
    inplace=True,
)
```

Applied after normalization to a 32x32 image, this selects one contained 16x16 square and fills it with normalized zero. Since the repository uses unit standard deviation and subtracts the CIFAR mean, zero corresponds to the per-channel training mean rather than an unnatural black extreme. The mask covers exactly 25% of pixels and is resampled independently for each view.

This is an orthogonal input-space regularizer on top of the accepted EXP-004 optimizer/model recipe. It retains crop and horizontal flip, but removes all RandAugment color/geometric operations so EXP-006 directly compares learned occlusion robustness against the accepted broad transformation policy.

## Local Evidence and Diagnosis

The moving baseline is EXP-004 at 92.30%, with 38,358 steps, 99 epochs, 300.0 counted training seconds, 340.7 total seconds, 330.1 MB peak VRAM, and 269,722 parameters. Its successful composition was:

- Standard crop/flip plus one-operation magnitude-7 RandAugment during the 80% `lr=0.1` plateau.
- A deterministic worker shutdown and rebuild exactly at the LR transition.
- Standard crop/flip only during the final 20% `0.01 -> 1e-4` cosine tail.

EXP-004 improved the previous 91.83% baseline by 0.47 points while retaining 99.3% of its optimizer steps. This establishes two important constraints for a replacement regularizer:

1. It should run in DataLoader workers, outside the synchronized GPU step, and preserve approximately 38k optimizer updates.
2. It should remain active until the 80% LR boundary. EXP-005 moved the RandAugment switch to 75% and regressed from 92.30% to 92.12%, despite unchanged throughput. Five percent of weak high-LR training displaced useful strong-view exploration.

Cutout attacks a different invariance gap from RandAugment. Rather than perturbing color, contrast, rotation, shear, or translation, it prevents the model from depending on one contiguous discriminative patch. CIFAR-10 objects often occupy much of a 32x32 frame; a 16x16 mask forces the shallow ResNet to combine evidence from multiple parts and surrounding context. The weak tail then optimizes the actual clean-image objective and refreshes BatchNorm statistics without occlusion.

## Primary Evidence

- DeVries and Taylor, *Improved Regularization of Convolutional Neural Networks with Cutout*, define Cutout as a fixed-size zero mask at a random input location and report improved CIFAR-10 performance. They emphasize zero-centered normalization and identify mask size as the main hyperparameter: <https://arxiv.org/abs/1708.04552>.
- Zhong et al., *Random Erasing Data Augmentation* (AAAI 2020), show that randomly erased rectangles generate occluded views, reduce overfitting, and complement crop/flip across image-classification tasks: <https://ojs.aaai.org/index.php/AAAI/article/view/7000>.
- The strongest repository-specific evidence is EXP-004, not the external headline numbers: worker-side strong augmentation can improve this exact model without sacrificing synchronized optimizer exposure, and a weak 20% tail converts difficult-view representations to the clean evaluator.

The external papers use different models, schedules, and training lengths; they support the mechanism and CIFAR mask scale, not an expected numerical gain over RandAugment.

## Exact Transform Pipelines

Define the plateau transform as:

```python
cutout_train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(
            p=1.0,
            scale=(0.25, 0.25),
            ratio=(1.0, 1.0),
            value=0,
            inplace=True,
        ),
    ]
)
```

Retain the accepted tail transform exactly:

```python
weak_train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
```

Do not include RandAugment in either pipeline. Do not apply Cutout before crop: cropping a pre-masked image would vary or remove the intended mask. Do not apply it before normalization with `value=0`, because that would create a normalized `-mean` black patch rather than a zero-centered mean patch. Do not use random fill in the first experiment; mean fill matches Cutout's zero-centered rationale and avoids introducing a second color-noise mechanism.

`scale=(0.25, 0.25)` and `ratio=(1.0, 1.0)` make torchvision's sampled target area exactly 256 pixels and the rounded height/width exactly 16. Because the implementation selects a fully contained rectangle, every accepted sample masks exactly 25% rather than clipping masks centered near an edge. `p=1.0` matches canonical Cutout's one-mask-per-view behavior and makes the intervention strength deterministic at the distribution level.

## Preserved Accepted Recipe

Keep all non-augmentation behavior from EXP-004 unchanged:

- ResNet-20 with widths 16/32/64 and 269,722 parameters.
- Batch size 128.
- Hard-label `F.cross_entropy`.
- SGD with `lr=0.1`, momentum 0.9, weight decay `1e-4`, and no Nesterov.
- Hold `lr=0.1` through 80% of counted time, then step to `0.01` and cosine-decay to `1e-4`.
- Eight persistent forkserver workers, pinned memory, shuffle, and `drop_last=True`.
- Synchronized per-step timing and the 300-second counted budget.
- The accepted early checkpoints, dense tail evaluation, and terminal evaluation.
- Seed 42, fixed evaluator, output schema, one-H20 protocol, and 600-second wall timeout.

The phase switch must reuse EXP-004's proven lifecycle:

1. Start with `make_train_loader(cutout_train_tf)` and `cutout_enabled = True`.
2. After the first batch crosses `0.8 * TIME_BUDGET_S`, break the current epoch immediately.
3. Perform the due evaluation.
4. Call `shutdown_train_loader`, verify all eight Cutout workers exit, delete the loader, and collect garbage.
5. Build `make_train_loader(weak_train_tf)`, set `cutout_enabled = False`, and log one `augmentation_switch: cutout->base` record.
6. Continue the unchanged weak low-LR tail; never switch back.

The crossing batch should still use `lr=0.1`, matching EXP-004. The next batch from the new loader should enter the scaled tail because elapsed progress is above 80%.

## Why Replace Rather Than Stack

Stacking Cutout after RandAugment would obscure attribution and could make plateau images too distorted for a 0.27M-parameter network. It would also combine two worker costs and potentially remove the throughput property that made EXP-004 successful. Replacement asks the useful comparative question: is structured occlusion a better high-LR representation regularizer than one random magnitude-7 color/geometric operation under the same time, optimizer, and weak-tail protocol?

An improvement would justify Cutout as the accepted strong-view family. A regression would not contradict Cutout's general usefulness; it would show that EXP-004's broader RandAugment invariances are better for this small model and short fixed budget.

## Worker Throughput and Fixed-Time Feasibility

EXP-004 measured the control loader at 329-383 batches/s, RandAugment at 165-176 batches/s, and the actual training loop consumed roughly `38,358 / 300 = 127.9` batches/s. RandAugment therefore had about 29-37% host-throughput headroom and preserved 99.3% of EXP-002's synchronized steps.

Fixed-square in-place RandomErasing consists of a few scalar random draws and one tensor slice assignment after `ToTensor`/normalization. With fixed valid area/aspect, its first placement attempt is valid; it does not invoke PIL geometry or color conversion. It should be closer to the weak loader than to RandAugment and is expected to remain above 200 batches/s.

Before the full run, repeat EXP-004's disposable `/tmp` loader diagnostic in a fresh process:

- Measure at least 1,000 warmed batches for weak crop/flip and Cutout pipelines with the real batch size, eight forkserver workers, pinning, and persistent workers.
- Require Cutout throughput of at least 160 batches/s, giving 25% headroom over the observed 128 batches/s GPU consumption.
- Exercise one real Cutout-to-weak loader switch, require all old workers to terminate, and require transition time below five seconds.
- Verify batch shape, finite normalized values, and that every sampled Cutout image contains a 16x16 normalized-zero rectangle. Use a separate diagnostic process so its random draws cannot perturb the fixed-seed training run.
- Project total time using measured loader and switch overhead; require comfortably below 600 seconds.

The full-run expectation is 37,900-38,600 steps, 98-100 reported epochs, about 330 MB VRAM, 300 counted seconds, and 340-350 total seconds. Since data loading occurs before the timed `t0`, loader slowdown primarily affects total wall time rather than `training_seconds`; the 600-second supervisor still makes preflight essential.

## Hypothesis and Expected Benefit

**Hypothesis:** one 16x16 mean-valued Cutout patch per plateau image will learn stronger part-distributed representations than RandAugment N1 M7 while retaining at least 98.5% of EXP-004's synchronized steps; the unchanged weak tail will convert those representations to `best_test_acc >= 92.40%`.

The expected accuracy range is 92.40-92.70%, a +0.10 to +0.40 point gain over the 92.30% moving baseline. Cutout is cheaper and its occlusion mechanism is well matched to CIFAR, but this is a replacement for an already successful augmentation rather than an addition to a weak baseline. The probability of a small regression is therefore material.

Useful intermediate signals are:

- Plateau checkpoints may be higher than EXP-004's final strong checkpoint of 84.60% because Cutout does not alter color/geometric statistics as broadly. This is not itself success; the clean weak-tail peak decides.
- The immediate post-switch jump may be smaller than EXP-004's 6.83 points because normalized-zero occlusion causes less BatchNorm/domain mismatch than RandAugment.
- A lower train-loss EMA with unchanged test accuracy would indicate Cutout is weaker than RandAugment at the chosen strength, not that it is computationally infeasible.

## Risks and Mitigations

- **Mask is too strong for a small 32x32 model.** A contained 16x16 patch removes 25% of every image. This is evidence-backed for CIFAR Cutout and is limited to the plateau; the 20% weak tail mitigates objective mismatch. Do not weaken it adaptively in the same experiment.
- **Mask is weaker than RandAugment.** Cutout targets only occlusion robustness. If train loss is materially lower and accuracy regresses with full throughput, future work may test a larger/variable mask, but EXP-006 should reject this fixed strength.
- **Mean fill is distinguishable as a hard-edged artifact.** Random location and crop/flip reduce reliance on the boundary, while zero-centered fill minimizes BatchNorm mean shift. Random fill would add another mechanism and is deferred.
- **In-place transform corrupts shared data.** `ToTensor()` creates a per-sample tensor before RandomErasing, so in-place mutation cannot modify the underlying CIFAR numpy image or another sample. Keep the transform order exact.
- **Worker RNG and augmentation stream differ from EXP-004.** This is inherent to replacing augmentation, not seed hacking. Seed remains 42 and there is one run with no reroll. Avoid claims of exact causal effect size from a single stochastic path.
- **Loader stalls reduce wall-clock feasibility.** Preflight against the 160 batches/s gate and verify worker teardown. If it fails before the experiment, the proposal is infeasible as implemented; do not move Cutout into the timed GPU loop as an unplanned workaround.
- **Switch implementation leaks workers.** Reuse the already verified explicit shutdown helper, retain worker PID logging, and require one switch record with eight stopped workers.
- **Sparse best-metric sampling.** Retain the accepted evaluation cadence exactly. EXP-002 found only a 0.01-point best-final gap, and EXP-004 completed safely with 25 unique evaluations.
- **Private DataLoader iterator internals are version-sensitive.** The accepted EXP-004 implementation already verified this exact PyTorch environment. Static and lifecycle preflight must still run before the single full experiment.

## Implementation Sketch

Starting from accepted EXP-004 `train.py`:

1. Replace `strong_train_tf` with `cutout_train_tf` as specified above.
2. Initialize the first loader from `cutout_train_tf`.
3. Rename `randaugment_enabled` to `cutout_enabled` and update the two switch predicates without changing their timing condition.
4. Change only the switch log label from `randaugment->base` to `cutout->base`.
5. Leave model, optimizer, scheduler, loop, evaluator calls, and summary untouched.

The intended tracked diff should contain no `RandAugment` call. It should contain exactly one `RandomErasing` call with the declared fixed parameters and only mechanical state/log renaming elsewhere.

## Verification Plan

1. Confirm the moving baseline from `04-results.tsv` is 92.30%; a valid improvement requires at least 92.40%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is visible.
3. Run the isolated worker-throughput and lifecycle preflight above; save only concise measurements in the plan/report, not a tracked diagnostic script.
4. Verify the tracked diff modifies only `train.py` and matches the replacement scope exactly.
5. Run syntax compilation, Ruff, pre-commit, and a static transform construction check using the installed torchvision.
6. Assert model parameter count remains 269,722 and train-loader length remains 390.
7. Remove any stale `run.log`, then execute exactly `uv run train.py > run.log 2>&1` under the 600-second supervisor.
8. Require exit code zero, one complete finite ten-field summary, about 300 counted seconds, and total time below 600 seconds.
9. Require exactly one `cutout->base` switch at approximately 80.0%, eight stopped workers, and no RandAugment switch text.
10. Require all evaluation epoch numbers unique, no more than one validation per epoch, and terminal evaluation aligned with the summary epoch.
11. Require at least 37,783 steps (98.5% of EXP-004's 38,358) for throughput equivalence. Record step delta, epochs, total time, and peak VRAM.
12. Require `best_test_acc >= 92.40%` for an improvement verdict. Compare best/final loss and accuracy with EXP-004's 92.30%/92.23% trajectory.
13. Remove `run.log` after analysis as required.

## Decision Rules

- **Accept:** accuracy at least 92.40%, all integrity conditions pass, and at least 37,783 steps complete. Cutout replaces RandAugment in the moving recipe.
- **Accuracy no-improvement with throughput equivalence:** reject fixed 16x16 Cutout as a replacement for N1 M7 RandAugment at this phase boundary. Revert to EXP-004; do not reroll or stack augmentations post hoc.
- **Accuracy gain with step shortfall:** the metric is formally improved, but attribute cautiously and investigate loader timing before accepting the implementation as a reusable recipe.
- **Throughput failure:** below 37,783 steps or preflight below 160 batches/s. Treat the worker implementation as infeasible under this protocol; do not infer that the statistical regularizer itself fails.
- **Lifecycle or timeout failure:** invalid experiment. Revert to accepted EXP-004 and diagnose the loader mechanics in a new predeclared experiment.

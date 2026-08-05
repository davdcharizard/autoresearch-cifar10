# Low-Magnitude RandAugment on the Accepted Mixup Schedule

## Thesis

Add one mild, standard torchvision RandAugment operation to each training image while leaving the accepted WRN-16-2, alpha-0.2 mixup-through-65%, hard-label tail, optimizer, and time-based schedule unchanged. This tests whether input-space invariances that crop, flip, and convex interpolation do not provide can move the 94.07% baseline, without repeating the failed interventions at their tested strengths. The policy is deliberately much weaker than the conventional `N=2, M=9` starting point: use exactly `num_ops=1` and `magnitude=5` of 31 bins, with no sweep or result-dependent retry.

## Exact Intervention

Construct one `transforms.RandAugment` in the training transform pipeline:

```python
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.RandAugment(
    num_ops=1,
    magnitude=5,
    num_magnitude_bins=31,
    interpolation=transforms.InterpolationMode.BILINEAR,
    fill=[125, 123, 114],
),
transforms.ToTensor(),
transforms.Normalize(mean, std),
```

Placement after crop and flip makes every RandAugment geometric magnitude refer to the final 32x32 field, and placement before `ToTensor` follows the transform's native PIL/uint8 path. Bilinear interpolation avoids nearest-neighbor aliasing for rotations and shears. The fill values are the rounded CIFAR-10 channel means in uint8 space, so newly exposed borders become approximately zero after the existing mean subtraction instead of creating black edge artifacts.

For `M=5/30`, the magnitude-bearing operations are modest: about 5 degrees rotation, 0.05 shear, 2.42 pixels translation, and 0.15 brightness/color/contrast/sharpness adjustment; posterization retains 7 bits and solarization uses a threshold near 212.5. `N=1` applies at most one operation per image, and the 14-operation torchvision space also contains identity and magnitude-independent operations. Do not add a probability wrapper, another augmentation, or tune `N/M` after observing the result.

## Temporal Policy

Keep RandAugment active for the full run. Preserve the accepted label policy exactly: batchwise alpha-0.2 mixup remains active before 65% counted training time and hard-label cross entropy remains active afterward. Thus the final 35% is a hard-label tail, but not an unaugmented-input tail; the already accepted random crop and flip also remain active there.

This full-run policy is intentional. Time-gating a PIL transform inside persistent DataLoader workers would require shared mutable state or loader reconstruction, and prefetched batches would make the cutoff lagged or ambiguous. Keeping one immutable worker transform makes the intervention reproducible and isolates the question "does this mild augmentation distribution help the accepted learner?" It also avoids changing epoch boundaries, shuffling, evaluation cadence, or the exact 65% mixup switch.

## Why This Is Not an Unchanged Retry of Failed Regularization

EXP-005 strengthened label interpolation from alpha 0.2 to 0.4 and lost 0.50 points. EXP-006 added p=0.10 feature masking on all six residual branches and lost 0.55 points. EXP-003 replaced the successful mixup path with shared-rectangle CutMix averaging 31% pasted area and lost 0.35 points. This proposal does none of those things: it retains alpha 0.2 and its cutoff, never masks learned features, never introduces a second target, and does not replace mixup. A single low-magnitude photometric or geometric transform targets local nuisance invariance rather than more target softness or information removal.

The additive-regularization warning remains material. RandAugment and mixup coexist for the first 65%, and RandAugment continues during late hard-label refinement. `N=1, M=5` is therefore a deliberately conservative one-shot test. A regression at normal exposure should be interpreted as evidence that the accepted WRN is already regularized enough, not as an invitation to retry a stronger `N=2, M=9` policy.

## Hypothesis and Decision Rule

The hypothesis is that mild image-space diversity improves learned CIFAR invariances enough to reach at least **94.17% `best_test_acc`**, the preregistered +0.10 percentage-point threshold over the accepted 94.07%. The policy should retain at least 95% of EXP-002's optimizer-step exposure (at least 26,348 of 27,735 steps) and finish below the 600-second hard wall.

One fixed-seed scored run is sufficient. Accept only if all integrity checks pass and `best_test_acc >= 94.17%`. A valid result below 94.17% falsifies the actionable hypothesis. If exposure remains normal and final test loss is at or above the accepted 0.2432, attribute the result to unhelpful or excessive augmentation rather than throughput. A lower loss without the accuracy threshold is still `no-improvement`, though it may indicate calibration rather than invariance was affected.

## CPU and Wall-Time Preflight

RandAugment runs in DataLoader workers and adds no model FLOPs or VRAM beyond the unchanged image batch. In the current loop, the counted step timer starts only after the loader yields a batch, so worker stalls primarily increase `total_seconds`, not `training_seconds`; this makes the 600-second total-wall limit the relevant operational risk. The accepted run used 341.2 total seconds for 27,735 steps, leaving 258.8 seconds of hard-wall margin, or about 9.33 ms additional wall time per step if every other cost stays fixed.

Before the scored run, perform an order-balanced local loader benchmark of the baseline transform and the proposed transform using the real CIFAR-10 training set, `BATCH_SIZE=256`, `NUM_WORKERS`, pinning, and persistent workers. Warm each loader, consume multiple complete iterator cycles, and emulate the accepted roughly 10.8 ms GPU consumer cadence so prefetch overlap is represented. Record steady-state wall milliseconds per yielded batch for base and RandAugment. Project:

`projected_total = 341.2s + max(0, ra_ms_per_batch - base_ms_per_batch) * 27,735 / 1000`.

Proceed only if projected total is at most 500 seconds, providing about 100 seconds of margin for worker jitter, validation, and startup, and if the loader exits cleanly with correctly shaped finite tensors. Otherwise reject this implementation as operationally unsuitable without weakening the policy or running a scored fallback. This preflight touches training data only and must not construct or inspect the evaluator/test set.

## RNG and Reproducibility

Keep `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`, shuffle behavior, and every existing model-side random operation unchanged. Torchvision RandAugment draws its operation, sign, and magnitude choice from the CPU torch RNG inside each seeded DataLoader worker. Because it is placed after crop and flip, the current sample's crop/flip draws occur before the new draws, but RandAugment advances that worker's stream and therefore changes crop/flip outcomes for later samples assigned to the worker. This is part of the fixed RandAugment treatment, not a seed reroll. Main-process shuffle and CUDA mixup/Beta/permutation streams are not directly consumed by the PIL transform. Log the fixed `N`, `M`, interpolation, fill, and placement so the stochastic policy is auditable.

## Evaluator and Constraint Integrity

Only `train.py` may change. The transform must be passed only to `datasets.CIFAR10(..., train=True, transform=train_tf)`. Do not alter `prepare.py`, `Eval`, the test transform, dataset split, labels, download source, seed, or evaluation schedule. Keep evaluation at the existing every-five-epochs cadence plus the budget-exhaustion evaluation, never more than once per epoch. Run on one H20 with the existing 300-second counted budget and 600-second external timeout; delete `run.log` after analysis.

## Failure Modes and Interpretation

- **Accuracy and loss regress with normal steps:** mild RandAugment still compounds the accepted mixup/crop/flip regularization; stop pursuing stronger or multi-op variants.
- **Early evaluations lag but the hard-label tail recovers to at least 94.17%:** supports complementary invariance learning despite slower fitting; accept on the primary metric.
- **Steps fall below 26,348 despite a passed loader preflight:** inspect whether transform work entered the counted region or GPU starvation changed synchronization timing; do not attribute the metric cleanly until accounted for.
- **Total runtime projection exceeds 500 seconds or the scored run exceeds 600 seconds:** classify the CPU PIL implementation as infeasible under the local wall constraint, not as an accuracy verdict.
- **Only magnitude-independent operations appear effective in spot checks:** do not alter the torchvision operation distribution; a frequency or ablation study would be a separate experiment.

## Estimated Effort

Low implementation effort and medium experimental risk. The code change is one transform insertion plus a configuration log line, while the principal uncertainty is additive regularization rather than correctness. The matched loader preflight and one full fixed-seed run are required; no online search, dependency installation, or remote service is involved.

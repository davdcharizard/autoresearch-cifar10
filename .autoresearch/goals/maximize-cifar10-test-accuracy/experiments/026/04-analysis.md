# Report EXP-026: Worker-Safe Early-Only RandAugment
- **Created**: 2026-07-26

## Goal

Maximize fixed-budget CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, requiring at least 94.17%. This experiment tested whether missing photometric/geometric invariances could improve the accepted WRN/mixup learner without sacrificing counted optimizer exposure or the late clean refinement phase.

## Idea & Hypothesis

Apply exactly one standard torchvision RandAugment operation at magnitude 5 after crop/flip during early training, then disable it after the first exhausted epoch ending at or beyond the 65% mixup cutoff. The hypothesis predicted normal exposure, wall time below 600 seconds, and at least 94.17% from useful input invariance plus a RandAugment-free hard-label tail.

## Approach

`train.py` gained a fixed `EarlyRandAugment` wrapper controlled by a shared forkserver byte and used by the existing eight persistent workers. RandAugment used the installed 14-operation policy, bilinear interpolation, and CIFAR mean-color fill. A worker-local RNG stream was swapped only around RandAugment and the accepted worker RNG restored in `finally`, preserving exact subsequent crop/flip and clean-tail trajectories without adding a seed. The flag changed only after normal iterator exhaustion. Evaluator, model, optimizer, schedule, mixup, seed, and evaluation cadence remained unchanged.

## Execution

The first semantic preflight attempt had a harness-only inference-tensor backward error; a normal backward probe fixed it before scoring. The retry proved model/RNG identity, exact clean-tail replay, forkserver flag propagation, and no next-epoch prefetch leakage. Balanced loader timing projected 346.735 and 426.101 seconds by two conservative models. One scored H20 run completed without retry: mixup stopped at step 17,859 / 195.0 seconds, RandAugment stopped after epoch 92 at step 17,940 / 195.9 seconds, and training finished normally.

## Results

- **Primary metric**: 94.12% (baseline: 94.07%, delta: +0.05 percentage points, +0.05%)
- **Observations**: The run completed 27,822 steps / 142.44864 passes in 300.0 counted and 345.2 total seconds. Best and final accuracy were both 94.12%; final loss was 0.2574 versus accepted 0.2432. Peak VRAM and parameters remained 1,094.0 MiB and 691,674. The RandAugment lag after mixup was 81 steps, below one 195-batch epoch.
- **Analysis**: The operational mechanism worked completely: image augmentation added no counted exposure tax, the clean-tail data/RNG trajectory was isolated from RandAugment draws, and the run retained slightly more exposure than accepted. Accuracy moved directionally by +0.05, suggesting mild image invariance is not as harmful as prior same-axis regularizers, but the gain was only half the required margin and test loss worsened. Because crop/flip RNG was exactly preserved, the delta cannot be dismissed as an augmentation-stream reroll. The exact `N=1,M=5` early-through-boundary policy is insufficient alone; its positive but small signal does not justify result-conditioned strength, operation-space, or cutoff tuning.
- **Key Learning**: Early RNG-isolated RandAugment preserves 142.45 passes and gains 0.05 points, but its mild invariance signal is insufficient as a standalone treatment.

## Verification

- **Conditions**: Device, scope, semantics, cutoff, loader feasibility, completion, wall time, exposure, transitions, and cadence passed; `best_test_acc >=94.17%` failed at 94.12%.
- **Review Notes**: Results are trustworthy: one fixed-seed run, exact accepted clean-tail replay, one H20, one complete summary, 27 unique evaluation epochs, no error markers, and only `train.py` changed in production.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run improved 0.05 points but missed the required +0.10-point margin by 0.05; no rerun is permitted.

## Unexplored Avenues

- Always-on, stronger, weaker, custom-filtered, or differently timed RandAugment remains unmeasured, but this result provides no principled setting choice and the accepted hard-label-tail evidence opposes immediate tuning. Treat those as closed absent a new mechanism.
- Composing this fixed image-invariance mechanism with a separately motivated overconfident capacity treatment is distinct from tuning RandAugment itself; it would need a new interaction hypothesis and exposure gate.

## Next Steps

- **Extra 8x8 block plus fixed early RandAugment (medium confidence)**: combine EXP-011's 94.15% capacity signal and high test loss with EXP-026's operationally cheap, directionally positive invariance signal; require the composition to retain the extra-block exposure regime.
- **Batch-128 with proportionally scaled LR (medium-low confidence)**: test update granularity and smaller-batch statistics as the strongest remaining standalone fallback, explicitly not as an increased-noise claim.
- **Alpha-0.1 mixup (low confidence)**: complete the isolated strength bracket only if the complementary composition fails review.

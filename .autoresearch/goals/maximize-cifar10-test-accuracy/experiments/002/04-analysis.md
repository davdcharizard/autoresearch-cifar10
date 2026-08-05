# Report EXP-002: Early Mixup With a Hard-Label Tail
- **Created**: 2026-07-24

## Goal

Increase CIFAR-10 `best_test_acc` above the 93.38% moving baseline by at least 0.10 percentage points while preserving the fixed 300-second training budget and all scope constraints.

## Idea & Hypothesis

Apply mild batchwise mixup only during the first 65% of counted training, then restore hard-label cross entropy for the final 35%. This directly targets the WRN baseline's near-zero-loss generalization gap while reserving a long low-LR tail for clean-label margin refinement. The hypothesis was that this would reach at least 93.48% without materially reducing training exposure.

## Approach

`train.py` now samples one device-resident beta coefficient with alpha 0.2 per batch, mixes images and labels through a single forward pass, and switches once to the unchanged hard-label path at 65% counted time. Architecture, optimizer, time-based schedule, seed, data augmentation, loader, and evaluation cadence remain identical to EXP-001. The finite-loss guard covers both loss paths.

## Execution

One full run completed successfully. Before it, the smoke-test exposure gate was changed from an invalid absolute projection to a matched relative threshold: mixup projected 120.3 passes versus 122.2 for EXP-001's code path, or 98.4%, above the 95% gate. The run switched mixup off exactly once at epoch 92, step 17,790, and 195.0 counted seconds. It completed 143 epochs and 27,735 steps in 341.2 seconds total.

## Results

- **Primary metric**: 94.07% (baseline: 93.38%, delta: +0.69 percentage points, +0.74%)
- **Observations**: Final accuracy equaled the best accuracy. The run realized 141.9 dataset passes versus about 146 for EXP-001, only 2.8% fewer, while peak VRAM remained low at 1,094 MiB.
- **Analysis**: The gain is well above the required 0.10-point margin and the pre-registered noise-sensitive band. Accuracy continued improving throughout the hard-label tail, supporting the intended division between early regularization and late clean-label convergence.
- **Key Learning**: Early alpha-0.2 mixup followed by a 35% hard-label tail improves the converged WRN baseline with minimal throughput loss.

## Verification

- **Conditions**: all passed
- **Review Notes**: Results confirmed trustworthy; the run completed with exit code 0, used one H20, counted exactly 300.0 training seconds, stayed below 600 seconds total, evaluated at most once per epoch, and modified only `train.py`.
- **Verdict**: improvement
- **Verdict Basis**: All hard constraints and verification conditions passed, and 94.07% exceeds the 93.38% baseline by 0.69 percentage points.

## Unexplored Avenues

- Tune the mixup cutoff around 50% or 75% to test whether more clean-label refinement or more regularization improves the balance.
- Tune alpha near 0.1-0.3, or combine the validated temporal mixup schedule with a low-overhead late weight average.

## Next Steps

Retain the early-mixup WRN as the new baseline. Next, prefer a controlled single-variable refinement such as late EMA or mixup cutoff/strength tuning; confidence is medium because both mechanisms have plausible but smaller remaining headroom.

## Exit Action Results

No exit actions were defined for this local-only run.

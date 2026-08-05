# Report EXP-027: Extra Final Block Plus Early RandAugment
- **Created**: 2026-07-26

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the 94.07% baseline under the fixed 300-second counted training budget, while satisfying the 0.10-point acceptance margin and all local-only integrity constraints.

## Idea & Hypothesis

Compose two distinct standalone near-misses exactly: EXP-011's `(2,2,3)` WRN and EXP-026's RNG-isolated early `RandAugment(num_ops=1, magnitude=5)`. The hypothesis was that early image invariance would regularize the deeper model's extra low-resolution capacity, producing at least 94.17% best and final accuracy while retaining at least 130 data passes. A final loss below EXP-011's 0.2782 would support that interaction mechanism.

## Approach

`train.py` now uses strict per-stage block counts `(2,2,3)`, yielding 987,098 parameters. It applies one bilinear mean-filled RandAugment operation after crop/flip through the first exhausted epoch ending at or after 65% counted time. A worker-private Torch RNG stream isolates RandAugment from accepted crop/flip and sampler randomness; the later clean tail, FP32 SGD, batch 256, time-based LR schedule, alpha-0.2 batch-shared mixup, seed, and evaluator cadence remain unchanged.

## Execution

The semantic harness initially failed because dynamically loaded EXP-026 multiprocessing classes were not importable by forkserver children. Three harness-only corrections established an importable physical module name; production code did not change during those retries. Semantic identity then passed, balanced GPU timing projected 130.651 passes, and real-loader timing projected 351.951 seconds by historical differential and 426.298 seconds absolute. The sole scored run completed with exit 0; no valid score was rerun.

## Results

- **Primary metric**: 94.32% (baseline: 94.07%, delta: +0.25 points, +0.266%)
- **Observations**: Final accuracy was 94.22%; final loss was 0.2523, 0.0259 below EXP-011. The run completed 25,978 steps, 133.007 passes, and 134 epochs in 300.0 counted / 345.3 total seconds with 1096.3 MiB peak VRAM. Mixup stopped at step 16,622 and RandAugment stopped after iterator exhaustion at step 16,770.
- **Analysis**: Both preregistered endpoint accuracy and loss evidence support the interaction hypothesis. The composition exceeded the acceptance threshold even though each exact component missed it alone, while retaining the deeper model's required exposure regime and the augmentation's exact clean-tail replay. This indicates early input invariance and added low-resolution capacity are complementary in this fixed-time learner rather than independently sufficient.
- **Key Learning**: RNG-isolated early image invariance unlocks useful extra low-resolution WRN capacity, converting two standalone near-misses into a +0.25-point improvement.

## Verification

- **Conditions**: all passed
- **Review Notes**: Results confirmed trustworthy: one local H20, frozen evaluator and `prepare.py`, one tracked production file, one finite summary, unique once-per-epoch evaluations, correct transitions, 133.007 passes, and 345.3 seconds wall time.
- **Verdict**: improvement
- **Verdict Basis**: all hard constraints and necessary conditions passed; 94.32% exceeds 94.07% by 0.25 points and predetermined final accuracy 94.22% also exceeds 94.17%.

## Unexplored Avenues

- Re-evaluate inexpensive conditioning or attention atop the accepted deeper-plus-invariance baseline; prior standalone effects may change under the now-better representation regime, but exact closed treatments should not be blindly replayed.
- Test a compute-neutral operating-point change such as batch 128 with a proportionally scaled LR curve, using the new 94.32% baseline and a strict exposure gate.

## Next Steps

- **Medium confidence**: profile the new accepted model's stage-wise cost and optimization behavior before selecting another capacity or conditioning intervention.
- **Medium confidence**: test batch 128 with the fully scaled LR curve as the strongest remaining standalone operating-point hypothesis.
- **Low confidence**: consider a strictly cheaper conditional gate only if it has a new interaction rationale and preserves the 133-pass regime.


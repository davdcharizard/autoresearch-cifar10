# Report EXP-001: Time-Aware Pre-Activation WRN-16-4
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` (%), where higher is better, under the frozen 300-second charged training budget. EXP-001 grew from BASE at 91.51%; after insertion, the global best is EXP-001 at 94.62%.

## Idea & Hypothesis

The chosen idea replaced the tiny 2016 ResNet-20 with a shallow-wide pre-activation WRN-16-4 while tying optimization phases to charged wall-clock progress. The hypothesis was that batch-256 BF16 channels-last execution would make the larger model affordable on H20, restrained stochastic depth would regularize it early, and time-normalized warmup/cosine decay would guarantee low-LR refinement. The predicted pass threshold was 91.61%, with a likely result above 93%.

## Approach

Only `train.py` changed. The implementation introduced a 2,748,890-parameter six-block pre-activation WRN with widths 64/128/256, learned projection shortcuts for shape changes, and per-example expectation-preserving residual drop path increasing to 0.08 by depth. Training used batch 256, BF16 autocast, channels-last tensors, Nesterov SGD, baseline weight decay `1e-4`, and a schedule driven by charged-time progress: LR 0.02 to 0.20 over 5%, then cosine to 0.002; drop path decayed to zero over the final 25%. Seed 42, crop/flip augmentation, input scaling, evaluator, charged timer, and summary keys stayed unchanged. A runtime-only preflight retained every-epoch evaluation.

## Execution

One run was launched on physical GPU 0 with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`; no retries or accuracy-driven adjustments were made. Preflight measured 11.597 ms median synthetic step latency and 0.870 seconds for a full evaluation, projecting about 446 total seconds. The actual run completed cleanly in 471.9 seconds with no NaN/Inf, traceback, CUDA, or memory errors. It completed 28,790 steps and 148 epochs, with one evaluation per epoch.

## Results

- **Primary metric**: 94.62% (parent: 91.51%, delta vs parent: +3.11 points, +3.40%; global best: 94.62%)
- **Observations**: Warmup reached LR 0.2000 at 5.0%, the final schedule reached LR 0.0020, and effective maximum drop path reached 0.000 by 99.9%. Best accuracy occurred at epoch 147 (94.62%) and final accuracy was close at 94.52%, so the result was not a transient early spike. Peak VRAM was only 1,178.9 MiB, leaving substantial memory headroom. Actual warm evaluation latency near the end was about 0.51 seconds, lower than the conservative preflight value.
- **Analysis**: The hypothesis was validated. The package exceeded its >93% expectation and improved BASE by 3.11 points while staying well inside both time and memory limits. The result supports the joint mechanism that the baseline was capacity- and schedule-limited, but this bundled experiment cannot isolate the contribution of pre-activation width, BF16/layout, Nesterov, time-cosine scheduling, or drop path individually. The near-zero late training loss alongside a 94.6% test ceiling suggests that additional generalization methods may offer more immediate upside than simply extending optimization.
- **Key Learning**: A time-scheduled pre-activation WRN-16-4 converts H20 headroom into a 3.11-point CIFAR-10 gain within the fixed budget.

## Verification

- **Conditions**: All passed. The run exceeded the parent-relative threshold, completed with a full summary, respected the 300-second charged budget and 600-second outer limit, used GPU 0, changed only `train.py`, and validated no more than once per epoch.
- **Review Notes**: Results are trustworthy. The metric came from the frozen `Eval.evaluate` harness on the local CIFAR-10 test set; configuration and parameter count matched the reviewed plan; the run started from a clean log and exited 0. No seed selection, evaluator modification, stale output, or scope violation was observed.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed and 94.62% exceeded the 91.51% parent by 3.11 points, well above the required 0.10-point margin.

## Unexplored Avenues

- Remove or reduce stochastic depth while holding the winning architecture and time schedule fixed; the late near-zero training loss suggests it did not prevent fitting, but an ablation would reveal whether it helped test accuracy.
- Add front-loaded CutMix or Mixup with a clean late phase on top of EXP-001. The high-capacity model now fits training extremely well, so literature-backed input regularization could raise the generalization ceiling.
- Add a carefully handled EMA of weights to reduce the remaining 0.10-point best-to-final gap without changing the evaluator or validation cadence.
- Explore WRN width/depth operating points only after regularization, because EXP-001 still uses little VRAM but attribution and generalization are now more important than raw capacity alone.

## Next Steps

- **High confidence**: Extend EXP-001 with front-loaded CutMix and retain the winning architecture/time schedule; it directly targets the observed train/test generalization gap.
- **Medium confidence**: Test sparse EMA on the EXP-001 recipe with correct BatchNorm buffer handling; the small best/final gap leaves limited but plausible upside.
- **Medium confidence**: Ablate stochastic depth or tune its maximum from 0.08 to 0.04 to determine whether the winning branch can gain from less early regularization.

## Exit Action Results

No exit actions were defined for this goal.

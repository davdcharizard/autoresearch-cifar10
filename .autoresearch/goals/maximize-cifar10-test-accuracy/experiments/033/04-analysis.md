# Report EXP-033: Three-Point Terminal Parameter Average
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` from the accepted 94.32% baseline to at least 94.42% within the fixed 300-second counted-training budget and 600-second wall limit. This experiment tested whether a short predetermined late parameter average could improve the terminal boundary representative without changing accepted live training.

## Idea & Hypothesis

Capture trainable parameters at the first post-update states whose pre-step counted times reach 95% and 97.5%, uniformly average them with terminal live parameters, retain terminal live BN buffers, and use that view only for the existing final evaluation. Unlike EXP013's 65%-start decay-0.999 whole-state EMA, this rule uses exactly three nearby clean-tail parameter states, gives the terminal state one-third weight, and never averages buffers. The hypothesis predicted at least 131.677 passes and an averaged endpoint of at least 94.42%.

## Approach

`train.py` added pure snapshot scheduling, two detached device-resident parameter snapshots, and a terminal averaging helper. Clone work remained inside the existing step timer. At budget exhaustion, the helper cloned terminal live parameters, fully materialized and finite-checked the fixed-order FP32 average, temporarily copied it into the existing parameter objects, invoked the frozen evaluator once, and restored every terminal parameter in `finally` with elementwise `torch.equal` verification. Model, optimizer, data, augmentation, schedule, seed, earlier evaluations, and BN buffers remained accepted.

The ignored preflight used an independent `git show 67c8e98:train.py` oracle and fake evaluators to prove initial state/RNG identity, exact thresholds, complete parameter coverage, optimizer and buffer isolation, evaluator RNG semantics, finite guards, partial-failure restoration, and exact live restoration. A first harness run failed only because it compared `cuda:0` against unspecialized `cuda`; the check was corrected to device type without changing production.

## Execution

H20 timing measured 0.682 ms total counted overhead for both snapshots, 99.9998% retention, 133.00706 projected passes, at most one lost terminal step, and a maximum LR offset of `1.225e-7`. The full excluded terminal state sequence added 5.40 ms, projecting 345.305 seconds wall; all timing CVs were below 2.3%.

The sole score completed once with exit 0. Mixup disabled at step 16,587 / 195.0 seconds and RandAugment after iterator exhaustion at step 16,770 / 197.1 seconds. Snapshots occurred once at step 24,549 / 285.002 seconds and step 25,211 / 292.509 seconds. The averaged terminal evaluation occurred once at epoch 133, and every live parameter restored exactly.

## Results

- **Primary metric**: 93.87% (baseline: 94.32%, delta: -0.45 points, -0.48%)
- **Observations**: Final averaged accuracy was 93.87% and final loss was 0.2606, versus accepted 94.22% and 0.2523. The last live evaluation at epoch 130 was also 93.87% with lower 0.2560 loss. The run completed 25,873 steps / 132.46976 passes / 133 epochs in 300.0 counted / 342.8 wall seconds, with 1,096.3 MiB peak VRAM and 27 unique evaluations.
- **Analysis**: The treatment achieved its local mechanism and retained normal exposure, but supplied no endpoint top-1 gain and worsened loss relative to both the accepted endpoint and this run's epoch-130 live evaluation. Because the protocol forbade evaluating both live and averaged states in the terminal epoch, it cannot isolate parameter-space curvature from terminal-BN mismatch or ordinary run variation. It does establish that the exact uniform `[95%,97.5%,100%]` trainable-parameter average with terminal live buffers is not a useful standalone refinement. Together with EXP013, two materially different late averaging policies have now failed to improve top-1.
- **Key Learning**: Short parameter-only terminal averaging preserves exposure and state integrity but does not improve top-1; late averaging is not the missing boundary-control mechanism.

## Verification

- **Conditions**: Run integrity passed; `best_test_acc=93.87%` failed the required 94.42%; averaged endpoint and loss corroboration also failed.
- **Review Notes**: The result is trustworthy: one local H20, fixed seed 42, frozen `prepare.py`/evaluator, only `train.py` changed, one score, one finite summary, correct transitions and snapshots, 132.46976 passes, 342.8 seconds wall, unique once-per-epoch evaluations, and exact restoration. No stale output or infrastructure failure was present.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid fixed-seed result completed within all hard constraints but scored 0.45 points below the current baseline and 0.55 below the acceptance threshold.

## Unexplored Avenues

- BN recalibration for averaged parameters is materially different, but it requires an extra training-data traversal outside the accepted learner and lacks evidence strong enough to justify immediate pursuit.
- Logit or prediction averaging would require extra evaluator forwards and a new fairness/accounting rationale; it is not an adjacent rescue for this result.

## Next Steps

- **Medium confidence**: Test batch 512 with the fully scaled `0.4 -> 0.004` LR curve only if direct H20 timing earns the preregistered 1.10x image-rate gate.
- **Low confidence**: Close the remaining mixup-strength bracket with alpha 0.1 as a clean one-line experiment, recognizing its negative local prior.
- **Low confidence**: Develop a new non-masking conditioning mechanism that preserves full high-resolution gradients and the accepted early-invariance/depth interaction.

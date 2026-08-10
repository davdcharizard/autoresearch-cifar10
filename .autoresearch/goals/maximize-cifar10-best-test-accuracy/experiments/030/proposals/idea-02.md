# Proposal: Raise Only the Weak-Tail Cosine Start LR to 0.02

## Decision

Change exactly one scalar in the accepted EXP010 recipe:

```python
ANNEAL_START_LR = 0.02  # accepted: 0.01
```

Preserve `LR=0.1`, `LR_HOLD_FRACTION=0.8`, `MIN_LR=1e-4`, and the existing elapsed-time cosine expression. Preserve every other model, initialization, batch, data, CutMix, optimizer, decay, timer, evaluator, worker, seed, precision, and logging choice. This isolates whether the accepted 60-second weak hard-label tail begins too conservatively.

## Mechanism and hypothesis

The accepted run keeps LR 0.1 with N1/M7 and probability-0.5 alpha-1 CutMix through 80% of counted time, then simultaneously switches to weak crop/flip, hard labels, and a cosine tail beginning near 0.01. The candidate keeps the complete protected strong phase bitwise and structurally unchanged, but starts weak-tail refinement near 0.02. This supplies larger early hard-label parameter movement while still dropping fivefold from the preceding 0.1 regime and converging to the same `1e-4` endpoint.

For tail coordinate `q=(progress-0.8)/0.2`, the candidate is

`lr(q) = 1e-4 + 0.5 * (0.02 - 1e-4) * (1 + cos(pi*q))`.

The accepted/candidate LRs are approximately 0.010/0.020 at `q=0`, 0.00505/0.01005 at `q=0.5` (90% total progress), 0.00155/0.00301 at `q=0.75` (95%), and both 0.0001 at completion. Thus mean tail LR rises from 0.00505 to 0.01005 and integrated tail learning-rate exposure is almost doubled, while the 80% high-LR exploration horizon is untouched.

The mechanistic prediction is faster adaptation from composite strong targets to clean hard labels and more useful motion before the terminal LR becomes tiny. The point prediction is `best_test_acc=94.30%`, with the formal hypothesis `best_test_acc >=94.25%` versus the 94.15% frontier, at at least 99% of accepted exposure.

This is not a free increase in only the loss-gradient step. PyTorch SGD first updates the ordinary momentum buffer from `g + 1e-4*w`, then multiplies that direction by LR; the first matched candidate weak update is therefore approximately twice the accepted update, and the effective coupled-decay displacement is also nearly doubled through most of the tail. Momentum coefficient/state semantics remain unchanged.

## Why this point is justified

- EXP002 established that an 80% LR-0.1 plateau followed by a 0.01-to-`1e-4` cosine tail was productive: it gained 0.16 points and ended within 0.01 of its best. It did not establish 0.01 as optimal.
- EXP010 carried the same schedule into the accepted width-2/CutMix recipe. It entered the weak tail at 89.73%, immediately reached 93.16%, continued rising, and finished at its best of 94.15% with NLL 0.1934. A rising terminal trajectory leaves a credible refinement-amplitude gap.
- EXP012 and EXP026 each recovered rapidly from deeper strong-phase underfit and peaked at 94.22%, only 0.03 below the acceptance gate. Their results suggest weak-tail conversion is powerful enough that a modest optimization change could matter, while their worse NLL warns that more aggressive movement need not improve calibration.
- The intervention has no additional operators, tensors, random draws, loader work, or architectural changes. EXP029's 1.97% overhead failure therefore motivates this zero-overhead scalar more strongly than another per-step helper.
- EXP005 and EXP027 protect the simultaneous data/LR boundary: hard or weaker-label training at LR 0.1 before 80% hurt. This proposal does not move that boundary or expose hard weak batches to LR 0.1.

## Exact invariants and implementation checks

The diff must be exactly the one constant change above. In particular, do not change the hold fraction, endpoint, cosine formula, comparison operator, CutMix probability/alpha, transforms, worker rebuild, SGD momentum, weight decay, evaluation cadence, or summary. Do not add warmup, a smooth 0.1-to-0.02 bridge, gradient clipping, a tail-only decay exception, or a fallback LR.

Before production, use a no-test disposable schedule audit:

1. Evaluate accepted and candidate LR functions at progress `0`, `0.2`, `0.4`, `0.6`, `0.7`, immediately below `0.8`, exactly `0.8`, immediately above `0.8`, `0.85`, `0.9`, `0.95`, and `1.0`.
2. Require exact equality at LR 0.1 through and including progress 0.8, candidate LR near 0.02 only after the strict `progress > 0.8` branch, monotonic non-increasing tail values, and exact common endpoint `1e-4`.
3. Simulate the current loop ordering and prove the strong loader is shut down after the step that crosses 80%; no strong N1/M7/CutMix step may use 0.02, and the first 0.02-like step must use the rebuilt weak hard-label loader.
4. Construct seed-42 accepted/candidate models and loaders without stepping. Require identical parameter/buffer/RNG hashes, parameter count 1,073,962, loader/collator configuration, optimizer groups/state, and backend flags. The scalar must not affect initialization or the data stream.
5. On a copied model/optimizer state and one fixed weak hard batch, manually verify the installed SGD recurrence at LR 0.01 and 0.02. Momentum-buffer values must initially match, candidate parameter displacement must be twice accepted to FP32 tolerance on that first matched update, and all state must remain finite.
6. Run compile, Ruff/pre-commit, scope, static evaluator-call, target-format, and worker-lifecycle checks. No timing benchmark is needed: both arms execute the identical instruction graph, and adding a timing apparatus would not inform the LR mechanism.

These checks establish exact scope and boundary semantics, not accuracy. They must not invoke `Eval.evaluate()` or use test labels.

## Production execution and expected diagnostics

After the checks pass, confirm one idle 97,871 MiB H20, remove stale completed logs, and execute exactly once at seed 42 with `uv run train.py > run.log 2>&1`. Do not retry, reroll, or change the LR after observing any checkpoint.

Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, peak VRAM near the accepted 598.7 MiB, at least 26,629 steps (99% of 26,898), one switch near 80%, all eight strong workers stopped, 45-55% strong CutMix, hard weak targets, no validation more than once per epoch, and 19 unique evaluations including the terminal epoch. Because only arithmetic values change, not operation count, exposure and evaluation cadence should match EXP010 closely; a material step loss is unexpected and must be explained rather than credited to the LR idea.

The key trajectory diagnostics are:

- **Strong phase:** switch accuracy should remain near EXP010's 89.73%, with unchanged CutMix fraction and about 21,446 strong steps. A substantial pre-switch difference indicates stream/environment drift or an unintended scope change, because `ANNEAL_START_LR` is not read on this branch.
- **First weak checkpoint:** compare against 93.16%. The hypothesis predicts at least comparable and preferably faster recovery; a lower value plus elevated loss is evidence of overshoot.
- **Tail path:** compare every common evaluation epoch, the epoch of the best checkpoint, best-final gap, and final NLL against 0.1934. Record whether larger early motion produces an earlier/higher peak or a late regression.
- **Optimization:** log/derive LR at switch, first weak step, 90%, 95%, and terminal progress; require finite loss and no one-class prediction pathology visible in evaluations. Preserve ordinary momentum throughout.
- **Primary:** require `best_test_acc >=94.25%`; final accuracy and NLL remain explanatory rather than acceptance substitutes.

## Failure risks

- **Hard-label overshoot:** the first weak update is about 2x accepted and inherits momentum accumulated under strong composite targets. Even though 0.02 is far below 0.1, the abrupt data/target change can transiently destabilize the representation.
- **Reduced convergence time:** nearly doubling mean tail LR leaves fewer effective seconds in the very-low-LR basin. The run may peak earlier, regress, or finish with worse NLL even if first-weak accuracy improves.
- **Coupled-decay confound:** fixed `weight_decay=1e-4` produces greater tail shrinkage when LR is larger. A result is the net schedule change, not evidence that gradient amplitude alone helped.
- **Protected-balance disruption:** EXP010's 0.01 drop may deliberately quench high-LR/CutMix momentum at the phase boundary. A 0.02 start could retain too much of the previous trajectory and undo clean-label refinement.
- **Single-point uncertainty:** 0.02 is a preregistered round point, not a sweep result. Failure does not imply every value between 0.01 and 0.02 fails, and success does not authorize post hoc tuning.
- **Single-seed resolution:** the acceptance margin is ten CIFAR-10 examples. A bare 94.25 pass is protocol-valid but weak effect-size evidence.

## Verdict and no-rescue rules

- **Improvement:** all integrity/runtime conditions pass and `best_test_acc >=94.25%`. Accept the scalar change; report switch, first-weak, tail path, final NLL, and exposure so the refinement mechanism is distinguishable from a lucky isolated checkpoint.
- **No improvement:** the run is valid but best accuracy is below 94.25%. Revert to 0.01 without reroll. Better first-weak accuracy, lower NLL, or a 94.22-like near miss cannot override the primary threshold.
- **Invalid:** scope/invariant failure, nonfinite state, missing summary, wrong hardware, fewer than 26,629 steps without a justified infrastructure cause, evaluation-cadence violation, crash, or total time over 600 seconds. Fix only a demonstrable implementation/infrastructure defect while retaining exactly 0.02; otherwise close the experiment.

Do not rescue the experiment with 0.015/0.025, a different hold fraction, gradient clipping, modified decay, extra evaluations, a second seed-42 run, or combination with another candidate. Those are new hypotheses.

## Evidence consulted

- `experiments/002/04-analysis.md`: original long-plateau, 0.01-start cosine-tail improvement.
- `experiments/010/04-analysis.md`: accepted CutMix frontier and rising weak-tail trajectory.
- `experiments/012/04-analysis.md` and `experiments/026/04-analysis.md`: strong underfit followed by fast weak-tail recovery and 94.22% near misses.
- Goal definition, system understanding, experiment learnings/results through EXP029, current `train.py`, and EXP030 brainstorm.

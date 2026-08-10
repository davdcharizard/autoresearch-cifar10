# Report EXP-026: Exact-Corpus Balanced Mixup/CutMix Retry
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of `94.15%` at `7c1e7d8`. A valid improvement required at least `94.25%` under the fixed seed-42, one-H20, 300-counted-second, `train.py`-only protocol. This experiment asked whether replacing half of the accepted CutMix events with Mixup could improve generalization without changing the total mixed-target rate.

## Idea & Hypothesis

Retry EXP019's unresolved balanced geometry with a causal evidence protocol: keep 50% hard strong batches, split the mixed half into 25% alpha-1 CutMix and 25% alpha-0.4 Mixup, and preserve the accepted hard weak tail. The idea was selected because whole-image interpolation supplies a distinct invariance from CutMix while holding aggregate regularization frequency constant. The hypothesis predicted at least `94.25%` and `26,629` optimizer steps after exact pre-policy replay cleared EXP019's non-replayable safety uncertainty.

## Approach

Only tracked `train.py` changed. One worker-local CPU draw selected hard, CutMix, or Mixup; explicit integer provenance distinguished the two soft-target geometries, and the production loop conditionally unpacked strong triples versus weak pairs. Model, optimizer, LR schedule, evaluator, seed, transforms, weak tail, timer, and 1,073,962 parameters remained fixed.

The experiment persisted the first 200 natural post-N1/M7, pre-policy source batches and their exact worker RNG states. Explicit accepted and candidate arms replayed those immutable sources; common hard and CutMix branches were bitwise equal. A 20,000-collation lifecycle gate and five alternating real-loader timing pairs preceded the sole production run.

## Execution

The semantic and safety controller passed on corpus SHA-256 `4386e6915d0bf3bb1f1f5dfdc6c36581758308d23df89eef2dd41202bb2e41e3`: candidate counts were 103 hard, 47 CutMix, and 50 Mixup; there was no candidate-only concentration; and candidate/control terminal loss-EMA ratio was `0.934313`. The 20,000-collation gate observed 50.295%/25.190%/24.515%, stopped all strong and weak workers, and rebuilt the weak loader in 2.912 seconds.

Two initial preflight wrappers were too short for the registered 20,000-batch CPU exercise; extending only the external diagnostic bound allowed the unchanged gate to complete. The first timing wrapper ended after nine of ten measured arms. A complete fresh retry persisted all five pairs. Its first analysis incorrectly concentrated the one-time loader transition into 13.7 seconds of probe work; production-horizon reanalysis preserved the raw trials and thresholds and passed. Mean candidate/control counted ratio was `1.003966`, worst pair `1.027406`, and projected exposure was 26,791 steps.

One scored run started at 2026-08-06T16:57:46Z, exited zero without retry, and completed at 2026-08-06T17:03:25Z. Its log SHA-256 is `8c44f6afff741e904ba5b6727e4d63e9b56f56d2e9b168efa9a67aec3732ea82`.

## Results

- **Primary metric**: `94.22%` (baseline: `94.15%`, delta: `+0.07` percentage points, `+0.07%` relative)
- **Observations**: The run completed 27,268 steps in 300.0 counted seconds and 332.4 total seconds, exceeding both the registered 26,629 floor and accepted EXP010's 26,898 steps. The 80.0% switch stopped all eight workers and reported 10,963 hard, 5,450 CutMix, and 5,423 Mixup batches (50.206%/24.958%/24.836%). Strong accuracy reached 89.68% at 60% but fell to 88.13% at the switch, 1.60 points below EXP010's 89.73%. The first weak evaluation recovered to 93.37%, 0.21 above EXP010's 93.16%, then peaked at 94.22% in epoch 68 and regressed to 94.05%. Final NLL was `0.1975`, worse than EXP010's `0.1934`; peak VRAM remained 598.7 MiB.
- **Analysis**: The exact-corpus evidence resolves EXP019's uncertainty: this 50/25/25 alpha-0.4 operating point is safe, semantically correct, and exposure-neutral, but it does not improve enough. Its lower short-probe loss did not predict phase-scale fit. Replacing half of CutMix with Mixup deepened late strong-phase underfit, while the hard tail converted that representation into a slightly better immediate weak checkpoint. The tail nearly matched the gate but retained worse calibration and a late regression. Extra exposure rules out throughput as the limiting factor; the result instead suggests whole-image interpolation traded away some useful CutMix-localization fit. The exact point is discredited under this fixed protocol, although milder geometry substitutions remain distinct hypotheses.
- **Key Learning**: Balanced Mixup recovered quickly in the hard tail but deeper switch underfit and worse NLL left its 94.22% peak 0.03 below the gate.

## Verification

- **Conditions**: The primary accuracy condition failed (`94.22% < 94.25%`). All integrity, scope, corpus, semantic, safety, lifecycle, timing, exposure, geometry, runtime, summary, and evaluator-cadence conditions passed.
- **Review Notes**: Results are trustworthy. Exactly one idle H20 and one scored seed-42 run were used; only `train.py` differed from `7c1e7d8`; the process exited zero with ten finite summary fields; training/total time were 300.0/332.4 seconds; all 19 evaluation epochs were unique; and no threshold, corpus, alpha, ratio, or seed was changed after observing data. Diagnostic timeout/accounting corrections did not rerun or alter the scored candidate.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid and improved the numeric baseline by 0.07 points, but the goal requires a margin of at least 0.10 points; it missed that gate by 0.03.

## Unexplored Avenues

- Replace a smaller fraction of CutMix events with a milder Mixup operating point. That could preserve more of the accepted switch fit, but alpha and probability are compound changes requiring a new reviewed experiment rather than a rescue of EXP026.
- End Mixup before the existing 80% strong-to-weak boundary while retaining RandAugment and CutMix. The strong decline and immediate weak recovery motivate an earlier geometry-only transition, but an extra persistent-worker phase and EXP005's earlier weak-tail failure make this a separate, higher-complexity hypothesis.
- Use feature-space interpolation only with an independently bounded mechanism. It could avoid global pixel ghosts, but it changes model/data coupling and has less direct evidence than the now-resolved input-space result.

## Next Steps

- **High confidence**: retain the accepted 50% alpha-1 CutMix recipe; the balanced alpha-0.4 replacement did not justify displacing it.
- **Medium confidence**: prioritize a mechanism that preserves strong fit, such as a bounded readout/capacity change, with exact recruitment and full-phase diagnostics.
- **Medium-low confidence**: investigate a new, milder geometry substitution only if it has a pre-registered reason to avoid the observed 1.60-point switch deficit.

## Exit Action Results

- None defined.

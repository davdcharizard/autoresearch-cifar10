# Report EXP-005: Stronger Alpha-0.4 Mixup
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with at least 94.17% required for a meaningful improvement, by testing whether stronger early mixup improves generalization.

## Idea & Hypothesis

Increase the symmetric mixup Beta parameter from 0.2 to 0.4 while retaining the validated 65% cutoff. Alpha 0.4 produces more materially interpolated batches, so the hypothesis was that stronger early regularization would reduce test loss and improve accuracy without sacrificing the long hard-label tail.

## Approach

Changed only `MIXUP_ALPHA` in `train.py` from 0.2 to 0.4. Architecture, batch size, optimizer, schedule, seed, mixup cutoff, evaluator cadence, and all other behavior remained unchanged. A CUDA preflight measured 35.53% of alpha-0.4 lambdas in `[0.2, 0.8]`, versus 21.34% for alpha 0.2, with both means near 0.5. Beta rejection sampling consumes an alpha-dependent number of CUDA random draws, so later data permutations are not bit-identical to EXP-002; this experiment measures the alpha-defined fixed-seed stochastic process.

## Execution

One preregistered run completed without retry or adjustment. Mixup was disabled exactly once at epoch 92, step 17,872, 195.0 counted seconds (65.0%). The run completed 27,875 steps across 143 epochs in 300.0 training seconds and 339.8 total seconds, with 1,094 MiB peak VRAM and no errors.

## Results

- **Primary metric**: 93.57% (baseline: 94.07%, delta: -0.50 percentage points, -0.53%)
- **Observations**: Final accuracy equaled best accuracy at 93.57%. Realized exposure was 142.72 data passes, close to EXP-002's 141.9, while final test loss worsened from 0.2432 to 0.2737.
- **Analysis**: Normal exposure and a stable but lower endpoint rule out throughput or an incomplete run as the cause. Stronger alpha-0.4 mixup over-regularized this WRN and degraded both the primary metric and test loss.
- **Key Learning**: Alpha 0.2 is better calibrated than alpha 0.4 for batchwise mixup through 65% of this fixed-time WRN training run.

## Verification

- **Conditions**: All structural and resource conditions passed; the 94.17% improvement threshold failed.
- **Review Notes**: Results confirmed trustworthy. The diff was one allowed constant change, the run used one NVIDIA H20, counted training was 300.0 seconds, total runtime was below 600 seconds, evaluation cadence was compliant, and parameter count remained 691,674.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid completed run scored 0.50 percentage points below baseline and 0.60 points below the required threshold.

## Unexplored Avenues

- An intermediate alpha such as 0.3 could test whether the optimum lies narrowly above 0.2, although the clear alpha-0.4 regression lowers its priority.
- Per-sample rather than batchwise lambda sampling could change regularization diversity, but it is a distinct implementation and would need isolated verification.

## Next Steps

Retain alpha 0.2 and the 65% cutoff. With high confidence, move to an orthogonal capacity or optimization lever rather than further strengthening mixup; a deconfounded WRN width increase is the leading candidate if its fixed-time throughput cost is explicitly accounted for.

## Exit Action Results

- Restore the accepted alpha-0.2 implementation: completed after analysis; EXP-005 is not merged.
